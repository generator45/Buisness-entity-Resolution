"""Stage 3b: V2 matcher datasets (V1 features + V2 features).

Reuses the candidate pairs written by stage 3
(data/intermediate/matcher/<sample>_candidates.parquet), so no blocking is
re-run and the pairs / targets are identical to V1. Computes all
pair_features_v2.V2_FEATURE_COLUMNS in chunks that always hold complete S1
candidate groups (needed for the per-S1 context features) and writes
data/marts/matcher/<sample>_pairs_v2.parquet with columns:
    source1_entity_id, candidate_entity_id, candidate_source, <features>, target
The V1 datasets are left untouched.

Run from code/business_entity_resolution/ after stage 3:
    python3 src/pipeline/run_stage3b_matcher_features_v2.py
"""

import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from config import MATCHER_DIR, PAIR_CHUNK_SIZE  # noqa: E402
from io_utils import ParquetChunkWriter  # noqa: E402
from pair_features_v2 import (  # noqa: E402
    V2_BINARY_FEATURES,
    V2_FEATURE_COLUMNS,
    V2_UNIT_INTERVAL_FEATURES,
    FieldIndexV2,
    compute_features_v2,
)
from run_stage3_matcher_data import (  # noqa: E402
    CANDIDATE_DIR,
    COLUMNS,
    SAMPLES,
    FeatureChecks,
    load_pool,
    load_s1_sample,
    log,
)
from split import load_matcher_split  # noqa: E402

V2_COLUMNS = sorted({*COLUMNS, "postal_proxy"})
SUFFIX = "v2"


def aligned_chunks(s1_rows: np.ndarray, size: int):
    """(start, end) slices of ~``size`` rows that never split an S1 group."""
    if (np.diff(s1_rows) < 0).any():
        raise ValueError("candidate pairs are not grouped by S1")
    group_starts = np.flatnonzero(np.r_[True, s1_rows[1:] != s1_rows[:-1]])
    n, start = len(s1_rows), 0
    while start < n:
        i = np.searchsorted(group_starts, start + size)
        end = int(group_starts[i]) if i < len(group_starts) else n
        yield start, end
        start = end


class ContextChecks:
    problems = []

    @classmethod
    def check(cls, feats):
        if (feats["sim_rank_in_s1"] < 1).any() or (
            feats["sim_rank_in_s1"] > feats["s1_candidate_count"]
        ).any():
            cls.problems.append("sim_rank_in_s1 outside [1, s1_candidate_count]")


def write_features_v2(name, s1, s1_rows, pool, pool_index, n_s2):
    s1_index = FieldIndexV2(s1, "s1", pool_index)
    cands = pq.read_table(CANDIDATE_DIR / f"{name}_candidates.parquet")
    cand_s1 = cands["s1_row"].to_numpy()
    chunks = list(aligned_chunks(cand_s1, PAIR_CHUNK_SIZE))
    checks = FeatureChecks(V2_FEATURE_COLUMNS, V2_BINARY_FEATURES, V2_UNIT_INTERVAL_FEATURES)
    out_path = MATCHER_DIR / f"{name}_pairs_{SUFFIX}.parquet"
    with ParquetChunkWriter(out_path) as writer:
        for i, (start, end) in enumerate(chunks):
            batch = cands.slice(start, end - start)
            qa = np.searchsorted(s1_rows, batch["s1_row"].to_numpy())
            qb = batch["pool_row"].to_numpy().astype(np.int64)
            target = batch["target"].to_numpy()
            feats = compute_features_v2(s1_index, pool_index, qa, qb)
            cand_ids = pool["entity_id"].take(pa.array(qb))
            source = pc.utf8_slice_codeunits(cand_ids, 0, 2)
            writer.write(pa.table({
                "source1_entity_id": s1["entity_id"].take(pa.array(qa)),
                "candidate_entity_id": cand_ids,
                "candidate_source": source,
                **{c: pa.array(v) for c, v in feats.items()},
                "target": pa.array(target),
            }))
            checks.update(feats, target, qb, source, n_s2)
            ContextChecks.check(feats)
            log(f"  {name}: chunk {i + 1}/{len(chunks)} ({checks.rows:,} pairs)")
    positives = int(pc.sum(cands["target"]).as_py())
    return out_path, checks, cands.num_rows, positives


def main():
    t_start = time.time()
    split = load_matcher_split()
    samples = {n: set(split.loc[split["split"] == n, "entity_id"]) for n in SAMPLES}
    overlap = len(samples["matcher_train"] & samples["matcher_val"])
    if overlap:
        raise ValueError(f"matcher samples overlap on {overlap} S1 IDs")

    pool, n_s2 = load_pool(V2_COLUMNS)
    t0 = time.time()
    pool_index = FieldIndexV2(pool, "pool")
    log(f"pool V2 feature index built in {time.time() - t0:.1f}s")

    summary = {}
    for name in SAMPLES:
        s1, s1_rows = load_s1_sample(samples[name], V2_COLUMNS)
        path, checks, n_pairs, n_pos = write_features_v2(name, s1, s1_rows, pool, pool_index, n_s2)
        summary[name] = {
            "output": str(path),
            "s1_entities": len(samples[name]),
            "candidate_pairs": n_pairs,
            "positive_pairs": n_pos,
            "rows_written": checks.rows,
            "problems": sorted(set(checks.problems)),
            "feature_means_by_target": checks.means(),
        }
        gc.collect()
    summary["context_problems"] = sorted(set(ContextChecks.problems))
    summary["matcher_split_overlap_s1_ids"] = overlap
    summary["runtime_seconds"] = round(time.time() - t_start, 1)
    with open(MATCHER_DIR / f"summary_{SUFFIX}.json", "w") as f:
        json.dump(summary, f, indent=2)

    for name in SAMPLES:
        s = summary[name]
        print(f"{name}: {s['rows_written']:,} rows ({s['candidate_pairs']:,} candidates, "
              f"{s['positive_pairs']:,} positive) | problems: {s['problems']}")
    print("context problems:", summary["context_problems"])
    means = pd.DataFrame({
        (n, t): {c: v[f"mean_target_{t}"] for c, v in summary[n]["feature_means_by_target"].items()}
        for n in SAMPLES for t in (1, 0)
    })
    print("\nfeature means by target (V2 additions):\n"
          + means.iloc[16:].round(4).to_string())
    log(f"done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
