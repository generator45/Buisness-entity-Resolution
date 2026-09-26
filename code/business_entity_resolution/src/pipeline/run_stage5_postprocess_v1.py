"""Stage 5 (V1): diagnostics and S1-level post-processing on validation.

Uses the saved V1 validation probabilities (no retraining, no test data).

Diagnostics (all on the leaderboard metric: macro F0.5 per S1 over every
matcher_val S1, singletons included, blocker misses counted as misses):
- ceilings: a perfect matcher on the current candidates ("oracle"), and the
  gains from removing all false positives / recovering all matcher misses
- loss breakdown by S1 type, country and number of true matches
- samples of the most confident false positives and false negatives, with
  names and addresses, for manual review

Post-processing rule families, each tuned for macro F0.5:
- R0 global threshold t
- R1 R0 + one S1 per candidate (a candidate may only be predicted for the
     S1 that gives it the highest probability)
- R2 R1 + separate thresholds for each S1's top candidate (t_top, which
     also decides singleton vs non-singleton) and its other candidates (t_rest)
- R3 R2 + relative margin: other candidates also need p >= r * top_p

Honest estimates: every rule is tuned on one half of the validation S1s and
scored on the other half, then the halves swap (2-fold cross-fitting at S1
level); "oof" numbers are these out-of-fold scores. The same-set optimum is
reported as "in_sample" for reference only.

Caveat: only matcher_val S1s (10% of all S1s) compete for candidates here,
so R1's one-S1-per-candidate rule fires far less often than it will on the
test set, where every S1 is present.

Run from code/business_entity_resolution/:
    python3 src/pipeline/run_stage5_postprocess_v1.py
"""

import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from config import MODELS_DIR, SEED, STAGING_DIR  # noqa: E402
from metrics import per_entity_fbeta  # noqa: E402
from split import load_matcher_split  # noqa: E402

PRED_PATH = MODELS_DIR / "matcher_v1_lgbm_val_predictions.parquet"
OUT_DIR = MODELS_DIR / "matcher_v1_analysis"
# pairs below this probability are never predicted by any rule in the grids
MIN_PROB = 0.02
FINE = np.round(np.arange(0.05, 0.991, 0.01), 2)
MEDIUM = np.round(np.arange(0.10, 0.991, 0.02), 2)
COARSE = np.round(np.arange(0.10, 0.991, 0.05), 2)
MARGINS = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95]
N_SAMPLE = 300


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class ValPredictions:
    """Validation pairs with per-S1 bookkeeping for the macro metric."""

    def __init__(self):
        split = load_matcher_split()
        self.val_ids = split.loc[split["split"] == "matcher_val", "entity_id"].to_numpy()
        self.n = len(self.val_ids)

        gt = pq.read_table(STAGING_DIR / "train" / "ground_truth.parquet").to_pandas()
        gt = gt.set_index("source1_entity_id").reindex(self.val_ids)["matched_entity_ids"]
        self.n_true = gt.fillna("").map(lambda s: len(s.split(",")) if s else 0).to_numpy()

        tbl = pq.read_table(PRED_PATH)
        enc = pc.dictionary_encode(tbl["source1_entity_id"].combine_chunks())
        pos = pd.Index(self.val_ids).get_indexer(enc.dictionary.to_numpy(zero_copy_only=False))
        s1_all = pos[enc.indices.to_numpy()]
        y_all = tbl["target"].to_numpy().astype(bool)
        p_all = tbl["pred_prob"].to_numpy()
        # true matches inside each S1's candidate set (a perfect matcher's output)
        self.cand_pos = np.bincount(s1_all[y_all], minlength=self.n)
        self.n_pairs, self.n_pos_pairs = len(y_all), int(y_all.sum())

        keep = p_all >= MIN_PROB
        self.s1, self.y, self.p = s1_all[keep], y_all[keep], p_all[keep]
        cand = pc.dictionary_encode(tbl["candidate_entity_id"].combine_chunks().filter(pa.array(keep)))
        self.cand = cand.indices.to_numpy()
        self.cand_ids = cand.dictionary
        self.missed_below_min = int((y_all & ~keep).sum())
        del tbl, enc, s1_all, y_all, p_all

        # candidate -> best S1 (highest probability) for the one-S1-per-candidate rule
        order = np.lexsort((-self.p, self.cand))
        first = np.r_[True, self.cand[order][1:] != self.cand[order][:-1]]
        self.winner = np.zeros(len(self.p), bool)
        self.winner[order[first]] = True
        self.top = {False: self._top(np.ones(len(self.p), bool)), True: self._top(self.winner)}

        rng = np.random.default_rng(SEED)
        self.fold = rng.permutation(self.n) % 2  # S1-level 2-fold assignment

    def _top(self, eligible):
        """(is_top flag per row, top probability per S1) among eligible rows."""
        idx = np.flatnonzero(eligible)
        order = idx[np.lexsort((-self.p[idx], self.s1[idx]))]
        first = np.r_[True, self.s1[order][1:] != self.s1[order][:-1]]
        is_top = np.zeros(len(self.p), bool)
        is_top[order[first]] = True
        top_p = np.zeros(self.n)
        top_p[self.s1[order[first]]] = self.p[order[first]]
        return is_top, top_p

    def predict(self, t_top, t_rest, margin=0.0, dedupe=False):
        eligible = self.winner if dedupe else np.ones(len(self.p), bool)
        is_top, top_p = self.top[dedupe]
        tp = top_p[self.s1]
        rest = (~is_top) & (self.p >= t_rest) & (tp >= t_top) & (self.p >= margin * tp)
        return eligible & ((is_top & (self.p >= t_top)) | rest)

    def per_s1(self, pred):
        n_pred = np.bincount(self.s1[pred], minlength=self.n)
        n_correct = np.bincount(self.s1[pred & self.y], minlength=self.n)
        return n_pred, n_correct

    def f_per_s1(self, pred):
        n_pred, n_correct = self.per_s1(pred)
        return per_entity_fbeta(n_pred, n_correct, self.n_true)


def search(vp: ValPredictions, family: str, grid):
    """Evaluate every parameter set; return per-set fold / overall means."""
    rows = []
    for params in grid:
        f = vp.f_per_s1(vp.predict(**params))
        rows.append({**params, "all": f.mean(),
                     "fold0": f[vp.fold == 0].mean(), "fold1": f[vp.fold == 1].mean()})
    res = pd.DataFrame(rows)
    res.insert(0, "family", family)
    return res


def cross_fit(vp: ValPredictions, res: pd.DataFrame):
    """Out-of-fold macro F0.5: tune on one fold, score on the other."""
    param_cols = [c for c in ("t_top", "t_rest", "margin", "dedupe") if c in res]
    f_oof = np.zeros(vp.n)
    chosen = {}
    for k in (0, 1):
        best = res.loc[res[f"fold{1 - k}"].idxmax(), param_cols].to_dict()
        chosen[f"tuned_on_fold{1 - k}"] = best
        f = vp.f_per_s1(vp.predict(**best))
        f_oof[vp.fold == k] = f[vp.fold == k]
    in_sample = res.loc[res["all"].idxmax()]
    return {
        "oof_macro_f0_5": float(f_oof.mean()),
        "in_sample_macro_f0_5": float(in_sample["all"]),
        "in_sample_params": in_sample[param_cols].to_dict(),
        "fold_params": chosen,
    }


def decomposition(vp: ValPredictions, pred):
    """Macro F0.5 under counterfactual fixes, starting from ``pred``."""
    n_pred, n_correct = vp.per_s1(pred)
    fp = n_pred - n_correct
    f = lambda a, b: float(per_entity_fbeta(a, b, vp.n_true).mean())  # noqa: E731
    return {
        "actual": f(n_pred, n_correct),
        "remove_all_false_positives": f(n_correct, n_correct),
        "recover_all_matcher_misses": f(vp.cand_pos + fp, vp.cand_pos),
        "oracle_perfect_matcher": f(vp.cand_pos, vp.cand_pos),
        "perfect_blocking_and_matcher": 1.0,
    }


def loss_breakdown(vp: ValPredictions, pred, s1_country):
    n_pred, n_correct = vp.per_s1(pred)
    f = per_entity_fbeta(n_pred, n_correct, vp.n_true)
    fp = n_pred - n_correct
    matcher_miss = vp.cand_pos - n_correct
    blocker_miss = vp.n_true - vp.cand_pos
    kind = np.select(
        [
            (vp.n_true == 0) & (n_pred == 0),
            vp.n_true == 0,
            f == 1.0,
            (n_pred == 0),
            (fp > 0) & ((matcher_miss + blocker_miss) > 0),
            fp > 0,
        ],
        [
            "singleton, correctly empty",
            "singleton, false prediction",
            "non-singleton, perfect",
            "non-singleton, nothing predicted",
            "non-singleton, false positives and misses",
            "non-singleton, false positives only",
        ],
        default="non-singleton, misses only",
    )
    n_true_bucket = np.where(vp.n_true >= 6, "6+", vp.n_true.astype(str))
    df = pd.DataFrame({"kind": kind, "country": s1_country, "n_true": n_true_bucket,
                       "f": f, "loss": 1 - f,
                       "has_blocker_miss": blocker_miss > 0})
    total_loss = df["loss"].sum()

    def table(col):
        g = df.groupby(col).agg(s1=("f", "size"), mean_f0_5=("f", "mean"), loss=("loss", "sum"))
        g["share_of_s1"] = g["s1"] / len(df)
        g["share_of_total_loss"] = g["loss"] / total_loss
        return g.drop(columns="loss").sort_values("share_of_total_loss", ascending=False)

    return {c: table(c) for c in ("kind", "country", "n_true", "has_blocker_miss")}


def error_samples(vp: ValPredictions, pred):
    """Most confident false positives and false negatives, joined to text."""
    fp_rows = np.flatnonzero(pred & ~vp.y)
    fp_rows = fp_rows[np.argsort(-vp.p[fp_rows])][:N_SAMPLE]
    fn_rows = np.flatnonzero(~pred & vp.y)
    fn_rows = fn_rows[np.argsort(vp.p[fn_rows])][:N_SAMPLE]
    cols = ["entity_id", "business_name", "business_address", "country"]
    s1 = pq.read_table(STAGING_DIR / "train" / "s1.parquet", columns=cols)
    pool = pa.concat_tables(
        pq.read_table(STAGING_DIR / "train" / f"{s}.parquet", columns=cols) for s in ("s2", "s3")
    )
    out = {}
    for kind, rows in (("false_positives", fp_rows), ("false_negatives", fn_rows)):
        s1_ids = pa.array(vp.val_ids[vp.s1[rows]])
        cand_ids = vp.cand_ids.take(pa.array(vp.cand[rows]))
        a = s1.take(pc.index_in(s1_ids, value_set=s1["entity_id"])).to_pandas()
        b = pool.take(pc.index_in(cand_ids, value_set=pool["entity_id"])).to_pandas()
        out[kind] = pd.DataFrame({
            "prob": vp.p[rows], "s1_id": a["entity_id"], "cand_id": b["entity_id"],
            "s1_name": a["business_name"], "cand_name": b["business_name"],
            "s1_address": a["business_address"], "cand_address": b["business_address"],
            "country": a["country"],
        })
    return out


def s1_countries(vp: ValPredictions):
    s1 = pq.read_table(STAGING_DIR / "train" / "s1.parquet", columns=["entity_id", "country"])
    idx = pc.index_in(pa.array(vp.val_ids), value_set=s1["entity_id"])
    return s1["country"].take(idx).to_numpy(zero_copy_only=False)


def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vp = ValPredictions()
    log(f"{vp.n:,} val S1 | {vp.n_pairs:,} pairs ({len(vp.p):,} with p >= {MIN_PROB}) "
        f"| true pairs in candidates {vp.n_pos_pairs:,} of {int(vp.n_true.sum()):,}")

    # ---- rule families
    families = {
        "R0 global threshold": [dict(t_top=t, t_rest=t) for t in FINE],
        "R1 + one S1 per candidate": [dict(t_top=t, t_rest=t, dedupe=True) for t in FINE],
        "R2 + top/rest thresholds": [
            dict(t_top=a, t_rest=b, dedupe=True) for a, b in itertools.product(MEDIUM, MEDIUM)
        ],
        "R3 + relative margin": [
            dict(t_top=a, t_rest=b, margin=m, dedupe=True)
            for a, b, m in itertools.product(COARSE, COARSE, MARGINS)
        ],
    }
    all_res, summary = [], {}
    for name, grid in families.items():
        t0 = time.time()
        res = search(vp, name, grid)
        all_res.append(res)
        summary[name] = cross_fit(vp, res)
        log(f"{name}: {len(grid)} settings in {time.time() - t0:.0f}s -> "
            f"oof {summary[name]['oof_macro_f0_5']:.4f}")
    pd.concat(all_res).to_csv(OUT_DIR / "postprocess_grid.tsv", sep="\t", index=False)

    # ---- diagnostics at the V1 reference point (t=0.60) and at the best rule
    base_pred = vp.predict(0.60, 0.60)
    best_name = max(summary, key=lambda k: summary[k]["oof_macro_f0_5"])
    best_params = summary[best_name]["in_sample_params"]
    best_pred = vp.predict(**best_params)
    decomp = {"V1 t=0.60": decomposition(vp, base_pred), best_name: decomposition(vp, best_pred)}

    both = base_pred & ~vp.winner
    dedupe_stats = {
        "pairs_removed_by_one_s1_per_candidate_at_t0.60": int(both.sum()),
        "of_which_true_matches": int((both & vp.y).sum()),
        "predicted_pairs_at_t0.60": int(base_pred.sum()),
    }

    country = s1_countries(vp)
    breakdown = loss_breakdown(vp, best_pred, country)
    samples = error_samples(vp, best_pred)
    for kind, df in samples.items():
        df.to_csv(OUT_DIR / f"{kind}_sample.tsv", sep="\t", index=False)
    for col, df in breakdown.items():
        df.to_csv(OUT_DIR / f"loss_by_{col}.tsv", sep="\t")

    report = {
        "val_s1": vp.n, "val_pairs": vp.n_pairs, "true_pairs_in_candidates": vp.n_pos_pairs,
        "true_pairs_total": int(vp.n_true.sum()),
        "true_pairs_below_min_prob": vp.missed_below_min,
        "rule_families": summary,
        "decomposition": decomp,
        "one_s1_per_candidate": dedupe_stats,
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    with open(OUT_DIR / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)

    # ---- console summary
    fam = pd.DataFrame({k: {"oof_macro_f0_5": v["oof_macro_f0_5"],
                            "in_sample_macro_f0_5": v["in_sample_macro_f0_5"],
                            "in_sample_params": v["in_sample_params"]}
                        for k, v in summary.items()}).T
    print("\n=== post-processing (macro F0.5, leaderboard metric)\n" + fam.to_string())
    print("\n=== ceilings / counterfactuals\n" + pd.DataFrame(decomp).round(4).to_string())
    print("\n=== one S1 per candidate at t=0.60:", dedupe_stats)
    for col, df in breakdown.items():
        print(f"\n=== loss by {col} (best rule)\n" + df.round(4).to_string())
    pd.set_option("display.max_colwidth", 45)
    pd.set_option("display.width", 250)
    for kind, df in samples.items():
        print(f"\n=== top {kind} (best rule)\n"
              + df.head(12)[["prob", "s1_name", "cand_name", "s1_address", "cand_address"]]
              .to_string(index=False))
    log(f"done in {time.time() - t_start:.0f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
