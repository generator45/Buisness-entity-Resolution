"""Stage 3c: add-on feature files, row-aligned with the V2 datasets.

For each matcher sample, reads the stage-3 candidate pairs in S1-aligned
chunks (the same order stage 3b wrote the V2 file), takes any V2 inputs the
add-on needs from <sample>_pairs_v2.parquet, computes the add-on's features
and writes data/marts/matcher/<sample>_pairs_<addon>.parquet (features +
target). Add-ons (see ADDONS): v3a (pair_features_v3a), v3b
(pair_features_v3b), v3c (pair_features_v3c).

Row alignment with the V2 file is verified on every row: S1 and candidate
IDs recomputed from the candidate pairs must equal the V2 file's IDs, and the
targets must agree. Training joins the files by row position.

Run from code/business_entity_resolution/ after stage 3b:
    python3 src/pipeline/run_stage3c_matcher_features_v3a.py [--addon v3b|v3c]
"""

import argparse
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
    V3A_DTYPES,
    V3A_UNIT_INTERVAL,
    FieldIndexV3a,
    compute_features_v3a,
)
from pair_features_v3b import (  # noqa: E402
    V3B_DTYPES,
    V3B_INPUTS,
    FieldIndexV3b,
    compute_features_v3b,
)
from pair_features_v3c import (  # noqa: E402
    V3C_DTYPES,
    V3C_INPUTS,
    FieldIndexV3c,
    compute_features_v3c,
)
from run_stage3_matcher_data import CANDIDATE_DIR, SAMPLES, load_pool, load_s1_sample, log  # noqa: E402
from run_stage3b_matcher_features_v2 import aligned_chunks  # noqa: E402
from split import load_matcher_split  # noqa: E402

ADDONS = {
    "v3a": {
        "index": FieldIndexV3a, "compute": compute_features_v3a, "v2_inputs": V2_INPUTS,
        "dtypes": V3A_DTYPES,
        "columns": ["entity_id", RAW_NAME, RAW_ADDRESS, NAME_FIELD, ADDRESS_FIELD],
        "ranges": {**{c: (0.0, 1.0) for c in V3A_UNIT_INTERVAL}, "name_rank_in_s1": (1, None)},
    },
    "v3b": {
        "index": FieldIndexV3b, "compute": compute_features_v3b, "v2_inputs": V3B_INPUTS,
        "dtypes": V3B_DTYPES,
        "columns": ["entity_id", NAME_FIELD, ADDRESS_FIELD],
        "ranges": {"address_number_min_edit": (-1, None),
                   "address_number_min_rel_diff": (-1.0, 1.0),
                   "name_s1_idf_sum": (0.0, None), "name_cand_idf_sum": (0.0, None),
                   "name_shared_idf_sum": (0.0, None)},
    },
    "v3c": {
        "index": FieldIndexV3c, "compute": compute_features_v3c, "v2_inputs": V3C_INPUTS,
        "dtypes": V3C_DTYPES,
        "columns": ["entity_id", NAME_FIELD, ADDRESS_FIELD],
        "ranges": {"cand_address_key_idf": (-1.0, 1.0), "s1_address_key_idf": (-1.0, 1.0),
                   "cand_address_distinct_names": (-1.0, None),
                   "s1_address_distinct_names": (-1.0, None)},
    },
}


class Checks:
    def __init__(self, dtypes, ranges):
        self.columns = list(dtypes)
        self.binary = [c for c, t in dtypes.items() if t == np.int8]
        self.ranges = ranges
        self.rows, self.problems = 0, []
        self.sums = {t: np.zeros(len(self.columns)) for t in (0, 1)}
        self.counts = {0: 0, 1: 0}

    def update(self, feats, target):
        self.rows += len(target)
        # column by column: stacking every feature as float64 would cost
        # ~0.6 GB per chunk
        if not all(np.isfinite(feats[c]).all() for c in self.columns):
            self.problems.append("non-finite feature values")
        for c in self.binary:
            if not np.isin(feats[c], (0, 1)).all():
                self.problems.append(f"{c} not binary")
        for c, (lo, hi) in self.ranges.items():
            v = feats[c]
            if (lo is not None and (v < lo - 1e-6).any()) or (hi is not None and (v > hi + 1e-6).any()):
                self.problems.append(f"{c} outside [{lo}, {hi}]")
        for t in (0, 1):
            sel = target == t
            self.sums[t] += [feats[c][sel].sum(dtype=np.float64) for c in self.columns]
            self.counts[t] += int(sel.sum())

    def means(self):
        return {c: {f"mean_target_{t}": self.sums[t][i] / max(self.counts[t], 1) for t in (1, 0)}
                for i, c in enumerate(self.columns)}


def build(name, suffix, spec, ids, pool, pool_index):
    s1, s1_rows = load_s1_sample(ids, spec["columns"])
    s1_index = spec["index"](s1, "s1", pool_index)
    cands = pq.read_table(CANDIDATE_DIR / f"{name}_candidates.parquet")
    v2 = pq.read_table(MATCHER_DIR / f"{name}_pairs_v2.parquet",
                       columns=["source1_entity_id", "candidate_entity_id", "target",
                                *spec["v2_inputs"]])
    if v2.num_rows != cands.num_rows:
        raise ValueError(f"{name}: V2 file has {v2.num_rows} rows, candidates {cands.num_rows}")
    chunks = list(aligned_chunks(cands["s1_row"].to_numpy(), PAIR_CHUNK_SIZE))
    checks = Checks(spec["dtypes"], spec["ranges"])
    out_path = MATCHER_DIR / f"{name}_pairs_{suffix}.parquet"
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
            v2_in = {c: v2_batch[c].to_numpy() for c in spec["v2_inputs"]}
            feats = spec["compute"](s1_index, pool_index, qa, qb, v2_in)
            writer.write(pa.table({**{c: pa.array(v) for c, v in feats.items()},
                                   "target": pa.array(target)}))
            checks.update(feats, target)
            log(f"  {name}: chunk {i + 1}/{len(chunks)} ({checks.rows:,} pairs)")
    return out_path, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addon", choices=sorted(ADDONS), default="v3a")
    suffix = ap.parse_args().addon
    spec = ADDONS[suffix]
    t_start = time.time()
    split = load_matcher_split()
    pool, _ = load_pool(spec["columns"])
    t0 = time.time()
    pool_index = spec["index"](pool, "pool")
    log(f"pool {suffix} index built in {time.time() - t0:.1f}s")
    summary = {}
    for name in SAMPLES:
        ids = set(split.loc[split["split"] == name, "entity_id"])
        path, checks = build(name, suffix, spec, ids, pool, pool_index)
        summary[name] = {"output": str(path), "rows": checks.rows, "aligned_with_v2": True,
                         "problems": sorted(set(checks.problems)),
                         "feature_means_by_target": checks.means()}
    summary["runtime_seconds"] = round(time.time() - t_start, 1)
    with open(MATCHER_DIR / f"summary_{suffix}.json", "w") as f:
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
