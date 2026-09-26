"""Stage 3c: V3a add-on features, row-aligned with the V2 datasets.

For each matcher sample, reads the stage-3 candidate pairs in S1-aligned
chunks (the same order stage 3b wrote the V2 file), takes the V2 inputs it
needs from <sample>_pairs_v2.parquet, computes pair_features_v3a.V3A_COLUMNS
and writes data/marts/matcher/<sample>_pairs_v3a.parquet (features + target).

Row alignment with the V2 file is verified on every row: S1 and candidate
IDs recomputed from the candidate pairs must equal the V2 file's IDs, and the
targets must agree. Training joins the two files by row position.

Run from code/business_entity_resolution/ after stage 3b:
    python3 src/pipeline/run_stage3c_matcher_features_v3a.py
"""

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
from pair_features import ADDRESS_FIELD, NAME_FIELD  # noqa: E402
from pair_features_v3a import (  # noqa: E402
    RAW_ADDRESS,
    RAW_NAME,
    V2_INPUTS,
    V3A_BINARY,
    V3A_COLUMNS,
    V3A_UNIT_INTERVAL,
    FieldIndexV3a,
    compute_features_v3a,
)
from run_stage3_matcher_data import CANDIDATE_DIR, SAMPLES, load_pool, load_s1_sample, log  # noqa: E402
from run_stage3b_matcher_features_v2 import aligned_chunks  # noqa: E402
from split import load_matcher_split  # noqa: E402

SUFFIX = "v3a"
COLUMNS = ["entity_id", RAW_NAME, RAW_ADDRESS, NAME_FIELD, ADDRESS_FIELD]


class Checks:
    def __init__(self):
        self.rows, self.problems = 0, []
        self.sums = {t: np.zeros(len(V3A_COLUMNS)) for t in (0, 1)}
        self.counts = {0: 0, 1: 0}

    def update(self, feats, target):
        self.rows += len(target)
        mat = np.column_stack([feats[c].astype(np.float64) for c in V3A_COLUMNS])
        if not np.isfinite(mat).all():
            self.problems.append("non-finite values")
        for c in V3A_BINARY:
            if not np.isin(feats[c], (0, 1)).all():
                self.problems.append(f"{c} not binary")
        for c in V3A_UNIT_INTERVAL:
            if ((feats[c] < 0) | (feats[c] > 1 + 1e-6)).any():
                self.problems.append(f"{c} outside [0, 1]")
        if (feats["name_rank_in_s1"] < 1).any():
            self.problems.append("name_rank_in_s1 < 1")
        for t in (0, 1):
            sel = target == t
            self.sums[t] += mat[sel].sum(axis=0)
            self.counts[t] += int(sel.sum())

    def means(self):
        return {c: {f"mean_target_{t}": self.sums[t][i] / max(self.counts[t], 1) for t in (1, 0)}
                for i, c in enumerate(V3A_COLUMNS)}


def build(name, ids, pool, pool_index):
    s1, s1_rows = load_s1_sample(ids, COLUMNS)
    s1_index = FieldIndexV3a(s1, "s1", pool_index)
    cands = pq.read_table(CANDIDATE_DIR / f"{name}_candidates.parquet")
    v2 = pq.read_table(MATCHER_DIR / f"{name}_pairs_v2.parquet",
                       columns=["source1_entity_id", "candidate_entity_id", "target", *V2_INPUTS])
    if v2.num_rows != cands.num_rows:
        raise ValueError(f"{name}: V2 file has {v2.num_rows} rows, candidates {cands.num_rows}")
    chunks = list(aligned_chunks(cands["s1_row"].to_numpy(), PAIR_CHUNK_SIZE))
    checks = Checks()
    out_path = MATCHER_DIR / f"{name}_pairs_{SUFFIX}.parquet"
    with ParquetChunkWriter(out_path) as writer:
        for i, (start, end) in enumerate(chunks):
            batch = cands.slice(start, end - start)
            v2_batch = v2.slice(start, end - start)
            qa = np.searchsorted(s1_rows, batch["s1_row"].to_numpy())
            qb = batch["pool_row"].to_numpy().astype(np.int64)
            target = batch["target"].to_numpy()
            # row alignment with the V2 file
            same = (
                pc.all(pc.equal(pool["entity_id"].take(pa.array(qb)),
                                v2_batch["candidate_entity_id"])).as_py()
                and pc.all(pc.equal(s1["entity_id"].take(pa.array(qa)),
                                    v2_batch["source1_entity_id"])).as_py()
                and np.array_equal(target, v2_batch["target"].to_numpy())
            )
            if not same:
                raise ValueError(f"{name}: chunk {i} is not row-aligned with the V2 file")
            v2_in = {c: v2_batch[c].to_numpy() for c in V2_INPUTS}
            feats = compute_features_v3a(s1_index, pool_index, qa, qb, v2_in)
            writer.write(pa.table({**{c: pa.array(v) for c, v in feats.items()},
                                   "target": pa.array(target)}))
            checks.update(feats, target)
            log(f"  {name}: chunk {i + 1}/{len(chunks)} ({checks.rows:,} pairs)")
    return out_path, checks


def main():
    t_start = time.time()
    split = load_matcher_split()
    pool, _ = load_pool(COLUMNS)
    t0 = time.time()
    pool_index = FieldIndexV3a(pool, "pool")
    log(f"pool V3a index built in {time.time() - t0:.1f}s")
    summary = {}
    for name in SAMPLES:
        ids = set(split.loc[split["split"] == name, "entity_id"])
        path, checks = build(name, ids, pool, pool_index)
        summary[name] = {"output": str(path), "rows": checks.rows, "aligned_with_v2": True,
                         "problems": sorted(set(checks.problems)),
                         "feature_means_by_target": checks.means()}
    summary["runtime_seconds"] = round(time.time() - t_start, 1)
    with open(MATCHER_DIR / f"summary_{SUFFIX}.json", "w") as f:
        json.dump(summary, f, indent=2)
    for name in SAMPLES:
        print(f"{name}: {summary[name]['rows']:,} rows, aligned with V2, "
              f"problems: {summary[name]['problems']}")
    means = pd.DataFrame({(n, t): {c: v[f"mean_target_{t}"]
                                   for c, v in summary[n]["feature_means_by_target"].items()}
                          for n in SAMPLES for t in (1, 0)})
    print("\nfeature means by target:\n" + means.round(4).to_string())
    log(f"done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
