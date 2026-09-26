"""Stage 2 (eval): measure blocking on the validation split.

Indexes the full train S2/S3 pool, retrieves candidates for the validation
S1 records, and reports recall / candidate volume / incremental contribution
per strategy. Also writes a breakdown and a sample of missed true matches to
data/intermediate/blocking/ to guide which strategy to add next.

Run from code/business_entity_resolution/:
    python3 src/pipeline/run_stage2_blocking_eval.py [--sets baseline_global full]
"""

import argparse
import functools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from blocking import (  # noqa: E402
    AddressNumberKeys,
    AddressRarePairKeys,
    NameAddressRareKeys,
    CoreNameKey,
    ExactKey,
    KeyBlock,
    NamePairKeys,
    SpacelessNameKey,
    TokenKeys,
    block_to_disk,
    iter_union,
)
from blocking_config import default_strategies  # noqa: E402
from blocking_eval import evaluate_blocking, format_report  # noqa: E402
from config import INTERMEDIATE_DIR, STAGING_DIR  # noqa: E402
from normalize import consonant_skeleton, phonetic_skeleton  # noqa: E402
from split import load_split  # noqa: E402
from translit_dict import MappedPhonetic, TranslitMap  # noqa: E402

from config import TRANSLIT_DICT_PATH  # noqa: E402

ADDRESS = "business_address_translit"

COLUMNS = [
    "entity_id",
    "business_name",
    "business_name_norm",
    "business_name_translit",
    "business_address",
    "business_address_translit",
    "country",
]
NORM, TRANSLIT = "business_name_norm", "business_name_translit"
RAW_COLUMNS = ["business_name", "business_address"]
# raw text is only needed for the missed-pair report, so the pool is loaded
# without it (~1.5 GB) and it is read back afterwards
POOL_COLUMNS = [c for c in COLUMNS if c not in RAW_COLUMNS]


def strategy_sets(max_df, token_max_df, addr_max_df, pair_max_df):
    """Named, ordered strategy lists. Order sets incremental attribution."""
    return {
        # the v1 baseline: global (not country-scoped) keys
        "baseline_global": [
            KeyBlock("exact_name", ExactKey(NORM, by_country=False)),
            KeyBlock(f"name_token[{max_df}]", TokenKeys(NORM, by_country=False), max_df),
        ],
        # the production configuration (blocking_config.py); --max-df sets the
        # core-name and spaceless caps. Pruned by leave-one-out on validation:
        # - name_token over translit / skeleton fields: +0.03 pts recall for
        #   ~53 cands/S1 and ~3.5 GB RAM
        # - exact_name, core_name, core_name_translit: each lost <= 16 true
        #   matches when removed; subsumed by the skeleton core key
        "full": default_strategies(
            token_max_df=token_max_df, core_max_df=max_df, addr_max_df=addr_max_df,
            pair_max_df=pair_max_df, spaceless_max_df=max_df,
        ),
        # before/after for the cheap fixes: each new variant sits right after
        # the version it replaces, so its new_true is exactly what the fix adds
        "cheap_fixes": [
            KeyBlock(
                f"name_token|country[{token_max_df}]", TokenKeys(NORM), token_max_df
            ),
            KeyBlock(
                f"core_name_skeleton|country[{max_df}]",
                CoreNameKey(TRANSLIT, transform=consonant_skeleton),
                max_df,
            ),
            KeyBlock(
                f"core_name_skeleton+indic|country[{max_df}]",
                CoreNameKey(
                    TRANSLIT, transform=functools.partial(consonant_skeleton, indic_folds=True)
                ),
                max_df,
            ),
            KeyBlock(
                f"core_name_skeleton+indic+digits|country[{max_df}]",
                CoreNameKey(TRANSLIT, transform=phonetic_skeleton),
                max_df,
            ),
            KeyBlock(
                f"addr_number_x_word|country[{addr_max_df}]",
                AddressNumberKeys("business_address_translit"),
                addr_max_df,
            ),
            KeyBlock(
                f"addr_number_x_word_norm|country[{addr_max_df}]",
                AddressNumberKeys("business_address_translit", normalize_numbers=True),
                addr_max_df,
            ),
            KeyBlock(f"name_pair|country[{pair_max_df}]", NamePairKeys(NORM), pair_max_df),
            KeyBlock(f"spaceless_name|country[{max_df}]", SpacelessNameKey(TRANSLIT), max_df),
        ],
        # production + every proposed addition, for incremental / leave-one-out
        # measurement in one run (see blocking_v2_candidates below)
        "blocking_v2_candidates": blocking_v2_candidates(token_max_df, max_df, addr_max_df,
                                                         pair_max_df),
    }


def blocking_v2_candidates(token_max_df, max_df, addr_max_df, pair_max_df):
    """Current production strategies first, then the proposed additions.

    Additions: word / word-pair keys on the accent-folded field, the learned
    transliteration dictionary (core-name and word keys; only if
    translit_dict.json exists), rarest-word address pairs and rarest name x
    address word keys.
    """
    strategies = default_strategies(
        token_max_df=token_max_df, core_max_df=max_df, addr_max_df=addr_max_df,
        pair_max_df=pair_max_df, spaceless_max_df=max_df,
    ) + [
        KeyBlock(f"name_token_translit|country[{token_max_df}]", TokenKeys(TRANSLIT),
                 token_max_df),
        KeyBlock(f"name_pair_translit|country[{pair_max_df}]", NamePairKeys(TRANSLIT),
                 pair_max_df),
    ]
    if TRANSLIT_DICT_PATH.exists():
        tmap = TranslitMap.load(TRANSLIT_DICT_PATH)
        strategies += [
            KeyBlock(f"core_name_dict_phonetic|country[{max_df}]",
                     CoreNameKey(TRANSLIT, transform=MappedPhonetic(tmap, phonetic_skeleton)),
                     max_df),
            KeyBlock(f"name_token_dict|country[{token_max_df}]",
                     TokenKeys(TRANSLIT, transform=tmap), token_max_df),
        ]
    strategies += [
        KeyBlock(f"addr_rare_pairs|country[{addr_max_df}]", AddressRarePairKeys(ADDRESS),
                 addr_max_df),
        KeyBlock(f"name_x_addr_rare|country[{addr_max_df}]",
                 NameAddressRareKeys(TRANSLIT, ADDRESS), addr_max_df),
    ]
    return strategies


OUT_DIR = INTERMEDIATE_DIR / "blocking"


def load_inputs():
    split = load_split()
    val_ids = pa.array(split.loc[split["split"] == "val", "entity_id"])
    s1 = pq.read_table(STAGING_DIR / "train" / "s1.parquet", columns=COLUMNS)
    s1 = s1.filter(pc.is_in(s1["entity_id"], value_set=val_ids))
    pool = pa.concat_tables(
        pq.read_table(STAGING_DIR / "train" / f"{s}.parquet", columns=POOL_COLUMNS)
        for s in ("s2", "s3")
    ).combine_chunks()

    gt = pq.read_table(STAGING_DIR / "train" / "ground_truth.parquet").to_pandas()
    gt = gt[gt["source1_entity_id"].isin(set(val_ids.to_pylist()))]
    gt = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")
    gt = gt[gt["m"].notna() & (gt["m"] != "")]
    gt_s1 = pc.index_in(pa.array(gt["source1_entity_id"]), value_set=s1["entity_id"])
    gt_pool = pc.index_in(pa.array(gt["m"]), value_set=pool["entity_id"])
    if gt_s1.null_count or gt_pool.null_count:
        raise ValueError("ground-truth IDs missing from staging tables")
    return s1, pool, gt_s1.to_numpy(), gt_pool.to_numpy()


def same_country_space(s1, pool) -> int:
    """Pairs a brute-force same-country comparison would evaluate."""
    pool_counts = pd.Series(pool["country"].to_numpy(zero_copy_only=False)).value_counts()
    s1_countries = pd.Series(s1["country"].to_numpy(zero_copy_only=False))
    return int(s1_countries.map(pool_counts).fillna(0).sum())


def precision_view(report: pd.DataFrame) -> str:
    cols = ["strategy", "recall", "pair_precision", "macro_precision",
            "macro_f05_all_cands", "new_pair_precision", "solo_precision",
            "avg_cands", "reduction_ratio"]
    return format_report(report, cols)


def miss_breakdown(s1, pool, miss_s1, miss_pool, field):
    """Why were true matches missed? Joins missed pairs back to their text."""
    a = s1.take(pa.array(miss_s1)).to_pandas()
    raw = pa.concat_tables(  # same row order as the pool (S2 then S3)
        pq.read_table(STAGING_DIR / "train" / f"{s}.parquet", columns=RAW_COLUMNS)
        for s in ("s2", "s3")
    )
    b = pool.select([c for c in pool.column_names if c not in RAW_COLUMNS]).take(
        pa.array(miss_pool)).to_pandas()
    for col in RAW_COLUMNS:
        b[col] = raw[col].take(pa.array(miss_pool)).to_numpy(zero_copy_only=False)
    del raw
    df = pd.DataFrame({
        "s1_id": a["entity_id"], "s1_name": a["business_name"],
        "cand_id": b["entity_id"], "cand_name": b["business_name"],
        "country": a["country"],
        "source": b["entity_id"].str[:2],
        "cand_non_latin": b["business_name"].str.contains(r"[^\x00-ɏ]"),
        "cand_empty_name": b[field] == "",
        "shares_any_token": [
            bool(set(x.split()) & set(y.split())) for x, y in zip(a[field], b[field])
        ],
    })
    return df


def sweep_cap(strategies, target, caps, s1, pool, gt_s1, gt_pool, args):
    """Cap sweep for one strategy, measured on the full stack.

    Each row is the whole stack's final recall / volume at that cap, plus the
    target's leave-one-out contribution (what it alone adds), which doesn't
    depend on where it sits in the stack.
    """
    k = next(i for i, s in enumerate(strategies) if s.name.startswith(target))
    strat = strategies[k]
    base_name = strat.name.split("[")[0]
    rows = []
    for cap in sorted(caps):
        strat.max_df = cap
        strat.name = f"{base_name}[{cap}]"
        report, _, _, elapsed = evaluate_blocking(
            strategies, s1, pool.num_rows, gt_s1, gt_pool, args.chunk_size
        )
        final, own = report.iloc[-1], report.iloc[k]
        rows.append({
            "cap": cap, "recall": final["recall"],
            "s1_fully_covered": final["s1_fully_covered"],
            "avg_cands": final["avg_cands"], "median": final["median"],
            "p95": final["p95"], "p99": final["p99"],
            "target_only_true": round(own["loo_recall_loss"] * len(gt_s1)),
            "target_only_pairs": own["loo_pairs_saved"],
        })
        print(f"  cap {cap}: blocking {elapsed:.1f}s")
    sweep = pd.DataFrame(rows)
    # the stack without the target: any run minus the target's exclusive share
    ref = sweep.iloc[-1]
    without = {
        "cap": 0, "recall": ref["recall"] - ref["target_only_true"] / len(gt_s1),
        "avg_cands": ref["avg_cands"] - ref["target_only_pairs"] / s1.num_rows,
        "target_only_true": 0, "target_only_pairs": 0,
    }
    sweep = pd.concat([pd.DataFrame([without]), sweep], ignore_index=True)
    sweep["pairs_per_target_only_true"] = sweep["target_only_pairs"] / sweep["target_only_true"]
    sweep.to_csv(OUT_DIR / f"sweep_{base_name.replace('|', '_')}.tsv", sep="\t", index=False)
    print(f"\nsweep of {base_name} (cap 0 = strategy removed)")
    print(sweep.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["baseline_global", "full"])
    ap.add_argument("--max-df", type=int, default=1000)
    ap.add_argument("--token-max-df", type=int, default=100)
    ap.add_argument("--addr-max-df", type=int, default=100)
    ap.add_argument("--pair-max-df", type=int, default=100)
    ap.add_argument("--chunk-size", type=int, default=20_000)
    ap.add_argument("--sweep-target", help="name prefix of the strategy to cap-sweep")
    ap.add_argument(
        "--sweep-caps", type=int, nargs="+",
        help="max_df values for --sweep-target (its index is built at the cap "
             "set by its --*-max-df flag, so sweep at or below it)",
    )
    args = ap.parse_args()

    t0 = time.time()
    s1, pool, gt_s1, gt_pool = load_inputs()
    print(
        f"val S1: {s1.num_rows:,} | pool S2+S3: {pool.num_rows:,} | "
        f"true pairs: {len(gt_s1):,} | load {time.time() - t0:.1f}s"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sets = strategy_sets(
        args.max_df, args.token_max_df, args.addr_max_df, args.pair_max_df
    )
    for set_name in args.sets:
        strategies = sets[set_name]
        print(f"\n=== {set_name}")
        if args.sweep_target:
            for strat in strategies:
                t0 = time.time()
                strat.fit(pool)
                idx = strat.index
                print(f"  fit {strat.name}: {time.time() - t0:.1f}s | kept "
                      f"{len(idx.vocab):,}/{idx.n_keys_total:,} keys, "
                      f"{len(idx.postings):,}/{idx.n_postings_total:,} postings")
            sweep_cap(strategies, args.sweep_target, args.sweep_caps,
                      s1, pool, gt_s1, gt_pool, args)
            for strat in strategies:
                del strat.index
            continue
        # low memory: one strategy index at a time, candidates via disk
        workdir = OUT_DIR / "tmp_candidates"
        block_to_disk(strategies, {"val": s1}, pool, workdir, args.chunk_size, log=print)
        report, miss_s1, miss_pool, elapsed = evaluate_blocking(
            strategies, s1, pool.num_rows, gt_s1, gt_pool, args.chunk_size,
            comparison_space=same_country_space(s1, pool),
            candidates=iter_union("val", s1, pool.num_rows, len(strategies), workdir,
                                  args.chunk_size),
        )
        print(f"blocking {elapsed:.1f}s")
        print(format_report(report))
        print("\nprecision view:")
        print(precision_view(report))
        report.to_csv(OUT_DIR / f"report_{set_name}.tsv", sep="\t", index=False)

        miss = miss_breakdown(s1, pool, miss_s1, miss_pool, NORM)
        print(f"\nmissed true pairs: {len(miss):,}")
        for col in ("source", "country", "cand_non_latin", "shares_any_token"):
            share = miss[col].value_counts(normalize=True).mul(100).round(1)
            print(f"  by {col}: {share.to_dict()}")
        miss.sample(min(len(miss), 2000), random_state=0).to_csv(
            OUT_DIR / f"missed_sample_{set_name}.tsv", sep="\t", index=False
        )
        for strat in strategies:
            del strat.index


if __name__ == "__main__":
    main()
