"""Stage 6: test-set inference -> output/matching_results.tsv + candidate_pairs.tsv.

Runs the production pipeline on the test split, one step per process (each
step reads the previous step's files, so a step can be re-run on its own and
no process holds more than one step's indexes):

  block  production blocker (blocking_config, TOP_N per S1) of every test S1
         against the test S2+S3 pool -> <work>/candidates.parquet
         (s1_row, pool_row; one row group per blocking chunk, S1-sorted)
  v2     V1 + V2 features -> <work>/pairs_v2.parquet (s1_row, pool_row,
         features); written in S1-aligned chunks of ~PAIR_CHUNK_SIZE pairs,
         one row group each
  v3a, v3b, v3c, v3d
         add-on features, one row group per V2 row group (row-aligned; the
         (s1_row, pool_row) of every row is carried along and checked)
  score  the --model tuned LightGBM (default v5) over the joined feature
         files -> <work>/predictions_<model>.parquet (s1_row, pool_row,
         pred_prob)
  write  post-processing (the model's R3 rule, tuned on matcher_val, see
         stage 5) and the two submission files; candidate_pairs.tsv is
         exactly the scored set
  check  streaming format checks of both files (the official validator is
         run separately on matching_results.tsv; on the candidate file it
         would hold ~250M IDs in Python sets)

S1-side feature lookups are built for S1_PART rows at a time (about half a
matcher sample), so every step's peak memory stays at or below the
training-time level regardless of the test size. All statistics (idf,
address-key df) come from the pool side, as in training.

``--limit N`` runs on the first N test S1 records only (trial run), with its
own work and output directories.

Run from code/business_entity_resolution/:
    python3 src/pipeline/run_stage6_test_inference.py --step block [--limit 50000]
"""

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from blocking import block_to_disk, iter_union_top_n  # noqa: E402
from blocking_config import TOP_N, default_strategies  # noqa: E402
from config import INTERMEDIATE_DIR, MODELS_DIR, OUTPUT_DIR, PAIR_CHUNK_SIZE, STAGING_DIR  # noqa: E402
from pair_features_v2 import FieldIndexV2, compute_features_v2  # noqa: E402
from run_stage3_matcher_data import COLUMNS  # noqa: E402
from run_stage3b_matcher_features_v2 import V2_COLUMNS, aligned_chunks  # noqa: E402
from run_stage3c_matcher_features_v3a import ADDONS  # noqa: E402
from run_stage4_train_matcher_v1 import VERSIONS  # noqa: E402

# Each model's R3 rule from stage 5 on matcher_val (in-sample optimum): each
# S1's top candidate needs p >= t_top, others p >= t_rest and p >= margin *
# top p; a candidate is only kept for the S1 that gives it the highest p
# (every S2/S3 ID matches at most one S1 in train ground truth). Per-segment
# thresholds (R4) gained <= 0.0002 out of fold and are not used.
MODELS = {
    # out-of-fold macro F0.5 0.9670
    "v4": {"path": MODELS_DIR / "matcher_v4_tuned_lgbm.txt",
           "rule": {"t_top": 0.70, "t_rest": 0.10, "margin": 0.7}},
    # v4 + V3d features; out-of-fold macro F0.5 0.9705
    "v5": {"path": MODELS_DIR / "matcher_v5_tuned_lgbm.txt",
           "rule": {"t_top": 0.75, "t_rest": 0.10, "margin": 0.7}},
}
MIN_PROB = 0.02  # as in stage 5; every rule threshold is above it
S1_PART = 120_000  # multiple of the blocking chunk (20,000); about half a matcher sample
BLOCK_CHUNK = 20_000
ID_BATCH = 100_000  # S1 rows per candidate_pairs.tsv write


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class ParquetChunkWriter:
    """Appends tables to one Parquet file, one row group per table.

    The steps below rely on row groups holding complete S1 candidate groups,
    so pyarrow's default row-group splitting (~1M rows) must not apply.
    """

    def __init__(self, path):
        self.path, self._writer = Path(path), None

    def write(self, table: pa.Table):
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema)
        self._writer.write_table(table, row_group_size=max(table.num_rows, 1))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._writer is not None:
            self._writer.close()


class Run:
    def __init__(self, limit, model):
        self.limit, self.model = limit, model
        tag = "test" if limit is None else f"test_trial_{limit}"
        self.work = INTERMEDIATE_DIR / tag
        self.out = OUTPUT_DIR if limit is None else INTERMEDIATE_DIR / tag / "output"
        self.work.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)

    def path(self, name):
        return self.work / name

    def s1(self, columns):
        t = pq.read_table(STAGING_DIR / "test" / "s1.parquet", columns=columns)
        if self.limit is not None:
            t = t.slice(0, self.limit)
        return t.combine_chunks()

    @staticmethod
    def pool(columns):
        s2 = pq.read_table(STAGING_DIR / "test" / "s2.parquet", columns=columns)
        s3 = pq.read_table(STAGING_DIR / "test" / "s3.parquet", columns=columns)
        return pa.concat_tables([s2, s3]).combine_chunks()


# ------------------------------------------------------------------ block

def step_block(run: Run):
    s1 = run.s1(COLUMNS)
    pool = run.pool(COLUMNS)
    log(f"test S1: {s1.num_rows:,} | pool S2+S3: {pool.num_rows:,}")
    strategies = default_strategies()
    tmpdir = run.path("tmp_blocking")
    block_to_disk(strategies, {"test": s1}, pool, tmpdir, chunk_size=BLOCK_CHUNK,
                  log=log, with_df=True)
    caps = [s.max_df for s in strategies]
    del strategies
    gc.collect()
    counts = np.zeros(s1.num_rows, np.int64)
    with ParquetChunkWriter(run.path("candidates.parquet")) as writer:
        for offset, q, c, _ in iter_union_top_n("test", s1, pool.num_rows, caps, tmpdir,
                                                TOP_N, chunk_size=BLOCK_CHUNK):
            if (np.diff(q) < 0).any():
                raise ValueError("candidates not sorted by S1 row")
            counts += np.bincount(q, minlength=s1.num_rows)
            if len(q):
                writer.write(pa.table({"s1_row": pa.array(q.astype(np.int32)),
                                       "pool_row": pa.array(c.astype(np.int32))}))
    shutil.rmtree(tmpdir)
    stats = {
        "s1": int(s1.num_rows), "pool": int(pool.num_rows), "pairs": int(counts.sum()),
        "avg_cands": float(counts.mean()), "median_cands": float(np.median(counts)),
        "p99_cands": float(np.percentile(counts, 99)), "max_cands": int(counts.max()),
        "s1_without_candidates": int((counts == 0).sum()),
    }
    json.dump(stats, open(run.path("blocking_stats.json"), "w"), indent=2)
    log(f"blocking: {stats}")


# --------------------------------------------------------------- features

def s1_parts(n_s1):
    return [(a, min(a + S1_PART, n_s1)) for a in range(0, n_s1, S1_PART)]


def step_v2(run: Run):
    s1_all = run.s1(V2_COLUMNS)
    pool = run.pool(V2_COLUMNS)
    t0 = time.time()
    pool_index = FieldIndexV2(pool, "pool")
    del pool
    gc.collect()
    log(f"pool V2 index built in {time.time() - t0:.1f}s")
    cand = pq.ParquetFile(run.path("candidates.parquet"))
    n_total, done = cand.metadata.num_rows, 0
    rg, n_rg = 0, cand.metadata.num_row_groups
    with ParquetChunkWriter(run.path("pairs_v2.parquet")) as writer:
        for a, b in s1_parts(s1_all.num_rows):
            s1_index = FieldIndexV2(s1_all.slice(a, b - a), "s1", pool_index)
            # candidate row groups are blocking chunks: never straddle a part
            while rg < n_rg:
                batch = cand.read_row_group(rg)
                s1_row = batch["s1_row"].to_numpy()
                if s1_row[0] >= b:
                    break
                if s1_row[0] < a or s1_row[-1] >= b:
                    raise ValueError(f"row group {rg} straddles S1 part [{a}, {b})")
                pool_row = batch["pool_row"].to_numpy()
                for start, end in aligned_chunks(s1_row, PAIR_CHUNK_SIZE):
                    qa = (s1_row[start:end] - a).astype(np.int64)
                    qb = pool_row[start:end].astype(np.int64)
                    feats = compute_features_v2(s1_index, pool_index, qa, qb)
                    writer.write(pa.table({
                        "s1_row": pa.array(s1_row[start:end]),
                        "pool_row": pa.array(pool_row[start:end]),
                        **{c: pa.array(v) for c, v in feats.items()},
                    }))
                    done += end - start
                rg += 1
            log(f"  v2: S1 part [{a:,}, {b:,}) done | {done:,}/{n_total:,} pairs")
            del s1_index
            gc.collect()
    if done != n_total:
        raise ValueError(f"v2 wrote {done} of {n_total} pairs")


def step_addon(run: Run, suffix):
    spec = ADDONS[suffix]
    s1_all = run.s1(spec["columns"])
    pool = run.pool(spec["columns"])
    t0 = time.time()
    pool_index = spec["index"](pool, "pool")
    del pool
    gc.collect()
    log(f"pool {suffix} index built in {time.time() - t0:.1f}s")
    v2 = pq.ParquetFile(run.path("pairs_v2.parquet"))
    cols = ["s1_row", "pool_row", *spec["v2_inputs"]]
    rg, n_rg, done = 0, v2.metadata.num_row_groups, 0
    with ParquetChunkWriter(run.path(f"pairs_{suffix}.parquet")) as writer:
        for a, b in s1_parts(s1_all.num_rows):
            s1_index = spec["index"](s1_all.slice(a, b - a), "s1", pool_index)
            while rg < n_rg:
                batch = v2.read_row_group(rg, columns=cols)
                s1_row = batch["s1_row"].to_numpy()
                if s1_row[0] >= b:
                    break
                if s1_row[0] < a or s1_row[-1] >= b:
                    raise ValueError(f"V2 row group {rg} straddles S1 part [{a}, {b})")
                qa = (s1_row - a).astype(np.int64)
                qb = batch["pool_row"].to_numpy().astype(np.int64)
                v2_in = {c: batch[c].to_numpy() for c in spec["v2_inputs"]}
                feats = spec["compute"](s1_index, pool_index, qa, qb, v2_in)
                writer.write(pa.table({"s1_row": batch["s1_row"], "pool_row": batch["pool_row"],
                                       **{c: pa.array(v) for c, v in feats.items()}}))
                done += batch.num_rows
                rg += 1
            log(f"  {suffix}: S1 part [{a:,}, {b:,}) done | {done:,}/{v2.metadata.num_rows:,} pairs")
            del s1_index
            gc.collect()
    if rg != n_rg:
        raise ValueError(f"{suffix} processed {rg} of {n_rg} V2 row groups")


# ------------------------------------------------------------------ score

def step_score(run: Run):
    sources = VERSIONS[run.model]["sources"]
    files = [(pq.ParquetFile(run.path(f"pairs{suffix}.parquet")), cols) for suffix, cols in sources]
    n_rg = files[0][0].metadata.num_row_groups
    for f, _ in files:
        if f.metadata.num_row_groups != n_rg:
            raise ValueError("feature files have different row groups")
    model = lgb.Booster(model_file=str(MODELS[run.model]["path"]))
    features = [c for _, cols in sources for c in cols]
    if model.feature_name() != features:
        raise ValueError("model features differ from the feature files")
    n_total = files[0][0].metadata.num_rows
    done = 0
    with ParquetChunkWriter(run.path(f"predictions_{run.model}.parquet")) as writer:
        for rg in range(n_rg):
            first = None
            x = None
            col0 = 0
            for f, cols in files:
                t = f.read_row_group(rg, columns=["s1_row", "pool_row", *cols])
                if first is None:
                    first = t
                    x = np.empty((t.num_rows, len(features)), dtype=np.float32)
                elif not (t["s1_row"].equals(first["s1_row"])
                          and t["pool_row"].equals(first["pool_row"])):
                    raise ValueError(f"row group {rg}: feature files not row-aligned")
                for j, c in enumerate(cols):
                    x[:, col0 + j] = t[c].to_numpy()
                col0 += len(cols)
            prob = model.predict(x)
            writer.write(pa.table({"s1_row": first["s1_row"], "pool_row": first["pool_row"],
                                   "pred_prob": pa.array(prob.astype(np.float32))}))
            done += len(prob)
            if rg % 25 == 0 or rg == n_rg - 1:
                log(f"  scored {done:,}/{n_total:,} pairs")


# ------------------------------------------------------------------ write

def select_matches(s1, cand, p, t_top, t_rest, margin):
    """R3 rule over pairs with p >= MIN_PROB; returns a boolean mask."""
    # one S1 per candidate: the S1 giving the candidate its highest p
    order = np.lexsort((-p, cand))
    first = np.r_[True, cand[order][1:] != cand[order][:-1]]
    winner = np.zeros(len(p), bool)
    winner[order[first]] = True
    # each S1's top candidate among its winning pairs
    idx = np.flatnonzero(winner)
    order = idx[np.lexsort((-p[idx], s1[idx]))]
    first = np.r_[True, s1[order][1:] != s1[order][:-1]]
    is_top = np.zeros(len(p), bool)
    is_top[order[first]] = True
    top_p = np.zeros(int(s1.max()) + 1 if len(s1) else 0)
    top_p[s1[order[first]]] = p[order[first]]
    tp = top_p[s1]
    rest = ~is_top & (p >= t_rest) & (tp >= t_top) & (p >= margin * tp)
    return winner & ((is_top & (p >= t_top)) | rest)


def join_lists(s1_ids, n_s1, s1_row, ids):
    """One 'S1<TAB>id,id,...' line per S1 row in [0, n_s1); rows sorted by S1."""
    offsets = np.searchsorted(s1_row, np.arange(n_s1 + 1)).astype(np.int32)
    if isinstance(ids, pa.ChunkedArray):
        ids = ids.combine_chunks()
    if isinstance(s1_ids, pa.ChunkedArray):
        s1_ids = s1_ids.combine_chunks()
    lists = pa.ListArray.from_arrays(pa.array(offsets), ids)
    joined = pc.binary_join(lists, ",")
    return pc.binary_join_element_wise(s1_ids, joined, "\t")


def step_write(run: Run):
    s1_ids = run.s1(["entity_id"])["entity_id"]
    pool_ids = run.pool(["entity_id"])["entity_id"]
    n_s1 = len(s1_ids)

    # --- matches
    rule = MODELS[run.model]["rule"]
    pred = pq.read_table(run.path(f"predictions_{run.model}.parquet"),
                         filters=[("pred_prob", ">=", MIN_PROB)])
    s1 = pred["s1_row"].to_numpy().astype(np.int64)
    cand = pred["pool_row"].to_numpy().astype(np.int64)
    p = pred["pred_prob"].to_numpy().astype(np.float64)
    keep = select_matches(s1, cand, p, **rule)
    order = np.lexsort((-p[keep], s1[keep]))  # by S1, most confident first
    m_s1, m_cand = s1[keep][order], cand[keep][order]
    lines = join_lists(s1_ids, n_s1, m_s1, pool_ids.take(pa.array(m_cand)))
    match_path = run.out / "matching_results.tsv"
    with open(match_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("\n".join(lines.to_pylist()) + "\n")
    n_matched = np.bincount(m_s1, minlength=n_s1)
    stats = {
        "s1": n_s1, "pairs_p_ge_min": int(len(p)), "predicted_pairs": int(keep.sum()),
        "s1_with_matches": int((n_matched > 0).sum()),
        "s1_singletons_predicted": int((n_matched == 0).sum()),
        "avg_matches_per_s1": float(n_matched.mean()),
        "model": run.model, "rule": {**rule, "one_s1_per_candidate": True},
    }
    del pred, s1, cand, p, keep, lines
    gc.collect()
    log(f"matching_results.tsv: {stats}")

    # --- candidates: the scored set, S1 by S1
    cands = pq.read_table(run.path("candidates.parquet"))
    c_s1 = cands["s1_row"].to_numpy()
    c_pool = cands["pool_row"].to_numpy()
    del cands
    cand_path = run.out / "candidate_pairs.tsv"
    with open(cand_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for a in range(0, n_s1, ID_BATCH):
            b = min(a + ID_BATCH, n_s1)
            lo, hi = np.searchsorted(c_s1, [a, b])
            ids = pool_ids.take(pa.array(c_pool[lo:hi]))
            lines = join_lists(s1_ids.slice(a, b - a), b - a, c_s1[lo:hi] - a, ids)
            f.write("\n".join(lines.to_pylist()) + "\n")
    stats["candidate_pairs"] = int(len(c_s1))
    json.dump(stats, open(run.path(f"output_stats_{run.model}.json"), "w"), indent=2)
    log(f"wrote {match_path} and {cand_path} ({len(c_s1):,} candidate pairs)")


# ------------------------------------------------------------------ check

def step_check(run: Run):
    """Streaming checks: header, one row per test S1 in order, S2/S3 IDs only,
    no duplicates in a list, every match among that S1's candidates, and the
    candidate file equal to the scored set."""
    s1_ids = run.s1(["entity_id"])["entity_id"].to_pylist()
    problems = []
    n_cand_ids = n_match_ids = 0
    with open(run.out / "matching_results.tsv", encoding="utf-8") as fm, \
            open(run.out / "candidate_pairs.tsv", encoding="utf-8") as fc:
        if fm.readline() != "source1_entity_id\tmatched_entity_ids\n":
            problems.append("matching header")
        if fc.readline() != "source1_entity_id\tcandidate_entity_ids\n":
            problems.append("candidate header")
        for i, (lm, lc) in enumerate(zip(fm, fc)):
            sm, _, rm = lm.rstrip("\n").partition("\t")
            sc, _, rc = lc.rstrip("\n").partition("\t")
            if i >= len(s1_ids) or not (sm == sc == s1_ids[i]):
                problems.append(f"line {i + 2}: S1 id / order mismatch")
                break
            m = rm.split(",") if rm else []
            c = rc.split(",") if rc else []
            cs = set(c)
            if len(cs) != len(c) or len(set(m)) != len(m):
                problems.append(f"{sm}: duplicate id in list")
            if any(not x.startswith(("S2-", "S3-")) for x in c):
                problems.append(f"{sm}: non S2/S3 candidate id")
            if not set(m) <= cs:
                problems.append(f"{sm}: match outside candidates")
            n_cand_ids += len(c)
            n_match_ids += len(m)
            if len(problems) > 20:
                break
        else:
            if i + 1 != len(s1_ids) or fm.readline() or fc.readline():
                problems.append("row count differs from test S1")
    n_scored = pq.ParquetFile(run.path(f"predictions_{run.model}.parquet")).metadata.num_rows
    if n_cand_ids != n_scored:
        problems.append(f"candidate ids {n_cand_ids} != scored pairs {n_scored}")
    log(f"check: {len(s1_ids):,} S1 rows, {n_cand_ids:,} candidate ids, "
        f"{n_match_ids:,} matched ids | problems: {problems or 'none'}")
    if problems:
        sys.exit(1)


STEPS = {"block": step_block, "v2": step_v2,
         "v3a": lambda r: step_addon(r, "v3a"), "v3b": lambda r: step_addon(r, "v3b"),
         "v3c": lambda r: step_addon(r, "v3c"), "v3d": lambda r: step_addon(r, "v3d"),
         "score": step_score, "write": step_write, "check": step_check}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=list(STEPS), required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N test S1 only (trial)")
    ap.add_argument("--model", choices=sorted(MODELS), default="v5")
    args = ap.parse_args()
    t0 = time.time()
    STEPS[args.step](Run(args.limit, args.model))
    log(f"step {args.step} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
