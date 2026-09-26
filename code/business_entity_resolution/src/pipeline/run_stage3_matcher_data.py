"""Stage 3: build matcher training / validation datasets (no model yet).

1. Sample two disjoint ~10% sets of train S1 entities (split.py,
   ``make_matcher_split``): matcher_train and matcher_val.
2. Run the production blocker (blocking_config.default_strategies) against
   the full train S2/S3 pool, independently for each sample, and write every
   (S1, candidate) pair with its ground-truth target to
   data/intermediate/matcher/<sample>_candidates.parquet. Only blocker
   candidates are kept: true matches the blocker missed are counted in the
   summary but never added as pairs.
3. Free the blocking indexes, then stream the candidate pairs in chunks and
   compute pairwise features (pair_features.py), writing
   data/marts/matcher/<sample>_pairs.parquet with columns:
   source1_entity_id, candidate_entity_id, candidate_source, <features>, target
4. Validate the outputs and write data/marts/matcher/summary.json.

Run from code/business_entity_resolution/:
    python3 src/pipeline/run_stage3_matcher_data.py
"""

import gc
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

from blocking import generate_candidates  # noqa: E402
from blocking_config import BLOCKING_COLUMNS, default_strategies  # noqa: E402
from config import (  # noqa: E402
    INTERMEDIATE_DIR,
    MATCHER_DIR,
    PAIR_CHUNK_SIZE,
    STAGING_DIR,
    STRONG_ADDRESS_JACCARD,
    STRONG_NAME_JACCARD,
)
from io_utils import ParquetChunkWriter  # noqa: E402
from pair_features import (  # noqa: E402
    ADDRESS_FIELD,
    BINARY_FEATURES,
    FEATURE_COLUMNS,
    NAME_FIELD,
    UNIT_INTERVAL_FEATURES,
    FieldIndex,
    compute_features,
)
from split import load_matcher_split  # noqa: E402

SAMPLES = ("matcher_train", "matcher_val")
CANDIDATE_DIR = INTERMEDIATE_DIR / "matcher"
COLUMNS = sorted({"entity_id", NAME_FIELD, ADDRESS_FIELD, *BLOCKING_COLUMNS})


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- inputs

def load_s1_sample(ids: set, columns=COLUMNS):
    """S1 rows of one sample, plus their row numbers in train s1.parquet."""
    s1 = pq.read_table(STAGING_DIR / "train" / "s1.parquet", columns=columns)
    mask = pc.is_in(s1["entity_id"], value_set=pa.array(sorted(ids)))
    rows = np.flatnonzero(mask.to_numpy(zero_copy_only=False))
    return s1.filter(mask).combine_chunks(), rows


def load_pool(columns=COLUMNS):
    """Train S2 then S3 rows concatenated; also returns the S2 row count."""
    s2 = pq.read_table(STAGING_DIR / "train" / "s2.parquet", columns=columns)
    s3 = pq.read_table(STAGING_DIR / "train" / "s3.parquet", columns=columns)
    return pa.concat_tables([s2, s3]).combine_chunks(), s2.num_rows


def true_pairs(s1: pa.Table, pool: pa.Table):
    """Ground-truth (S1 row, pool row) pairs for the S1 records in ``s1``."""
    gt = pq.read_table(STAGING_DIR / "train" / "ground_truth.parquet").to_pandas()
    gt = gt[gt["source1_entity_id"].isin(set(s1["entity_id"].to_pylist()))]
    gt = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")
    gt = gt[gt["m"].notna() & (gt["m"] != "")]
    s1_rows = pc.index_in(pa.array(gt["source1_entity_id"]), value_set=s1["entity_id"])
    pool_rows = pc.index_in(pa.array(gt["m"]), value_set=pool["entity_id"])
    if s1_rows.null_count or pool_rows.null_count:
        raise ValueError("ground-truth IDs missing from staging tables")
    return s1_rows.to_numpy(), pool_rows.to_numpy()


# ------------------------------------------------------ candidates + target

def write_candidates(name, s1, s1_rows, pool, strategies):
    """Block one sample and write (s1 row, pool row, target) pairs."""
    gt_s1, gt_pool = true_pairs(s1, pool)
    n_pool = pool.num_rows
    gt_keys = np.sort(gt_s1.astype(np.int64) * n_pool + gt_pool)
    cands_per_s1 = np.zeros(s1.num_rows, np.int64)
    pos_per_s1 = np.zeros(s1.num_rows, np.int64)
    path = CANDIDATE_DIR / f"{name}_candidates.parquet"
    with ParquetChunkWriter(path) as writer:
        for _, q, c, _ in generate_candidates(strategies, s1, n_pool):
            target = np.isin(q * n_pool + c, gt_keys, assume_unique=True)
            cands_per_s1 += np.bincount(q, minlength=s1.num_rows)
            pos_per_s1 += np.bincount(q[target], minlength=s1.num_rows)
            writer.write(pa.table({
                "s1_row": pa.array(s1_rows[q].astype(np.int32)),  # row in train s1.parquet
                "pool_row": pa.array(c.astype(np.int32)),  # row in train s2+s3 concat
                "target": pa.array(target.astype(np.int8)),
            }))
    gt_per_s1 = np.bincount(gt_s1, minlength=s1.num_rows)
    return path, {
        "cands_per_s1": cands_per_s1,
        "pos_per_s1": pos_per_s1,
        "gt_per_s1": gt_per_s1,
    }


# ----------------------------------------------------------------- features

class FeatureChecks:
    """Streaming data-quality checks and per-class feature means."""

    def __init__(self, columns=FEATURE_COLUMNS, binary=BINARY_FEATURES,
                 unit_interval=UNIT_INTERVAL_FEATURES):
        self.columns, self.binary, self.unit_interval = columns, binary, unit_interval
        self.rows = 0
        self.problems = []
        self.sums = {t: np.zeros(len(columns)) for t in (0, 1)}
        self.counts = {0: 0, 1: 0}
        self.country_mismatch = 0

    def update(self, feats, target, pool_rows, source, n_s2):
        self.rows += len(target)
        mat = np.column_stack([feats[c].astype(np.float64) for c in self.columns])
        if not np.isfinite(mat).all():
            self.problems.append("non-finite feature values")
        for c in self.binary:
            if not np.isin(feats[c], (0, 1)).all():
                self.problems.append(f"{c} not binary")
        for c in self.unit_interval:
            if ((feats[c] < 0) | (feats[c] > 1)).any():
                self.problems.append(f"{c} outside [0, 1]")
        for pre in ("name", "address"):
            if (feats[f"{pre}_shared_meaningful_token_count"]
                    > feats[f"{pre}_token_overlap_count"]).any():
                self.problems.append(f"{pre} meaningful overlap > overlap")
        # pool rows [0, n_s2) come from the S2 file, the rest from S3
        expected = np.where(pool_rows < n_s2, "S2", "S3")
        if not (source.to_numpy(zero_copy_only=False) == expected).all():
            self.problems.append("candidate_source inconsistent with source file")
        self.country_mismatch += int((feats["country_exact_match"] == 0).sum())
        for t in (0, 1):
            sel = target == t
            self.sums[t] += mat[sel].sum(axis=0)
            self.counts[t] += int(sel.sum())

    def means(self):
        return {
            col: {
                "mean_target_1": self.sums[1][i] / max(self.counts[1], 1),
                "mean_target_0": self.sums[0][i] / max(self.counts[0], 1),
            }
            for i, col in enumerate(self.columns)
        }


def write_features(name, cand_path, s1, s1_rows, pool, pool_index, n_s2):
    s1_index = FieldIndex(s1, "s1")
    checks = FeatureChecks()
    out_path = MATCHER_DIR / f"{name}_pairs.parquet"
    cand_file = pq.ParquetFile(cand_path)
    n_chunks = -(-cand_file.metadata.num_rows // PAIR_CHUNK_SIZE)
    with ParquetChunkWriter(out_path) as writer:
        for i, batch in enumerate(cand_file.iter_batches(batch_size=PAIR_CHUNK_SIZE)):
            qa = np.searchsorted(s1_rows, batch["s1_row"].to_numpy())  # sample-local row
            qb = batch["pool_row"].to_numpy().astype(np.int64)
            target = batch["target"].to_numpy()
            feats = compute_features(s1_index, pool_index, qa, qb)
            cand_ids = pool["entity_id"].take(pa.array(qb))
            source = pc.utf8_slice_codeunits(cand_ids, 0, 2)
            table = pa.table({
                "source1_entity_id": s1["entity_id"].take(pa.array(qa)),
                "candidate_entity_id": cand_ids,
                "candidate_source": source,
                **{c: pa.array(v) for c, v in feats.items()},
                "target": pa.array(target),
            })
            checks.update(feats, target, qb, source, n_s2)
            writer.write(table)
            log(f"  {name}: features chunk {i + 1}/{n_chunks} ({checks.rows:,} pairs)")
    return out_path, checks


# ------------------------------------------------------------------ summary

def sample_summary(stats, checks, n_sampled):
    c, p, g = stats["cands_per_s1"], stats["pos_per_s1"], stats["gt_per_s1"]
    n_pairs, n_pos = int(c.sum()), int(p.sum())
    return {
        "s1_entities_sampled": n_sampled,
        "s1_entities_with_candidates": int((c > 0).sum()),
        "s1_entities_without_candidates": int((c == 0).sum()),
        "s1_singletons_in_ground_truth": int((g == 0).sum()),
        "candidate_pairs": n_pairs,
        "positive_pairs": n_pos,
        "negative_pairs": n_pairs - n_pos,
        "positive_rate": n_pos / max(n_pairs, 1),
        "avg_candidates_per_s1": float(c.mean()),
        "median_candidates_per_s1": float(np.median(c)),
        "p95_candidates_per_s1": float(np.percentile(c, 95)),
        "p99_candidates_per_s1": float(np.percentile(c, 99)),
        "max_candidates_per_s1": int(c.max()),
        "ground_truth_pairs": int(g.sum()),
        "blocker_missed_true_pairs": int(g.sum() - n_pos),
        "blocking_recall": n_pos / max(int(g.sum()), 1),
        "rows_written": checks.rows,
        "country_mismatch_pairs": checks.country_mismatch,
        "problems": sorted(set(checks.problems)),
        "feature_means_by_target": checks.means(),
    }


def validate_outputs(paths, summaries):
    """Cross-dataset checks on the written files."""
    ids, issues = {}, []
    for name, path in paths.items():
        tbl = pq.read_table(path, columns=["source1_entity_id", "candidate_entity_id", "target"])
        ids[name] = set(pc.unique(tbl["source1_entity_id"]).to_pylist())
        key = pc.binary_join_element_wise(tbl["source1_entity_id"], tbl["candidate_entity_id"], "|")
        if pc.count_distinct(key).as_py() != tbl.num_rows:
            issues.append(f"{name}: duplicate (S1, candidate) pairs")
        if int(pc.sum(tbl["target"]).as_py()) != summaries[name]["positive_pairs"]:
            issues.append(f"{name}: positive count differs from candidate stage")
        if tbl.num_rows != summaries[name]["candidate_pairs"]:
            issues.append(f"{name}: row count differs from candidate stage")
        del tbl, key
    overlap = len(ids["matcher_train"] & ids["matcher_val"])
    if overlap:
        issues.append(f"{overlap} S1 IDs appear in both datasets")
    return overlap, issues


def main():
    t_start = time.time()
    split = load_matcher_split()
    samples = {n: set(split.loc[split["split"] == n, "entity_id"]) for n in SAMPLES}
    split_overlap = len(samples["matcher_train"] & samples["matcher_val"])
    if split_overlap:
        raise ValueError(f"matcher samples overlap on {split_overlap} S1 IDs")
    log("samples: " + ", ".join(f"{n}={len(v):,}" for n, v in samples.items())
        + f" | overlap={split_overlap}")

    pool, n_s2 = load_pool()
    s1_tables = {n: load_s1_sample(ids) for n, ids in samples.items()}
    log(f"pool S2+S3: {pool.num_rows:,}")

    # --- blocking: indexes built once, queried per sample
    strategies = default_strategies()
    for strat in strategies:
        t0 = time.time()
        strat.fit(pool)
        log(f"fit {strat.name}: {time.time() - t0:.1f}s")
    cand_paths, stats = {}, {}
    for name in SAMPLES:
        s1, s1_rows = s1_tables[name]
        cand_paths[name], stats[name] = write_candidates(name, s1, s1_rows, pool, strategies)
        log(f"{name}: {int(stats[name]['cands_per_s1'].sum()):,} candidate pairs -> {cand_paths[name]}")
    del strategies
    gc.collect()

    # --- features: pool-side lookups built once, S1 side per sample
    t0 = time.time()
    pool_index = FieldIndex(pool, "pool")
    log(f"pool feature index built in {time.time() - t0:.1f}s")
    out_paths, summaries = {}, {}
    for name in SAMPLES:
        s1, s1_rows = s1_tables[name]
        out_paths[name], checks = write_features(
            name, cand_paths[name], s1, s1_rows, pool, pool_index, n_s2
        )
        summaries[name] = sample_summary(stats[name], checks, len(samples[name]))
    del pool_index
    gc.collect()

    overlap, issues = validate_outputs(out_paths, summaries)
    summary = {
        "matcher_split_overlap_s1_ids": split_overlap,
        "dataset_overlap_s1_ids": overlap,
        "cross_dataset_issues": issues,
        "strong_overlap_thresholds": {
            "name_jaccard": STRONG_NAME_JACCARD, "address_jaccard": STRONG_ADDRESS_JACCARD,
        },
        "outputs": {n: str(p) for n, p in out_paths.items()},
        "samples": summaries,
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    MATCHER_DIR.mkdir(parents=True, exist_ok=True)
    with open(MATCHER_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    keys = ["s1_entities_sampled", "s1_entities_with_candidates", "candidate_pairs",
            "positive_pairs", "negative_pairs", "positive_rate", "avg_candidates_per_s1",
            "median_candidates_per_s1", "p95_candidates_per_s1", "p99_candidates_per_s1",
            "s1_singletons_in_ground_truth", "ground_truth_pairs",
            "blocker_missed_true_pairs", "blocking_recall", "country_mismatch_pairs"]
    print("\n" + pd.DataFrame({n: {k: summaries[n][k] for k in keys} for n in SAMPLES}).to_string())
    print(f"\nS1 IDs in both samples: split={split_overlap}, datasets={overlap}")
    print("problems:", {n: summaries[n]["problems"] for n in SAMPLES}, "| cross-dataset:", issues)
    means = pd.DataFrame({
        (n, t): {c: v[f"mean_target_{t}"] for c, v in summaries[n]["feature_means_by_target"].items()}
        for n in SAMPLES for t in (1, 0)
    })
    print("\nfeature means by target:\n" + means.round(4).to_string())
    log(f"done in {time.time() - t_start:.0f}s -> {MATCHER_DIR}")


if __name__ == "__main__":
    main()
