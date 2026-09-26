"""Stage 2d (eval): per-record top-N candidate ranking on blocking validation.

Runs the production blocking (blocking_config.default_strategies) on the
blocking validation split with key document frequencies, ranks each S1
record's candidates by blocking evidence, and reports recall / volume when
only the top N per record are kept:

- A "agreement": number of strategies that retrieved the candidate, then the
  document frequency of the rarest key it shares (ascending)
- B "rarity": sum over retrieving strategies of 1 - ln(df) / ln(cap + 1)
  (see blocking.iter_union_scored)

Ties are broken by pool row, so results are deterministic.

Run from code/business_entity_resolution/:
    python3 src/pipeline/run_stage2d_topn_eval.py
"""

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from blocking import block_to_disk, iter_union_scored  # noqa: E402
from blocking_config import default_strategies  # noqa: E402
from config import INTERMEDIATE_DIR  # noqa: E402
from run_stage2_blocking_eval import load_inputs  # noqa: E402

TOP_N = [25, 50, 75, 100, 150, 200, 300]
OUT_DIR = INTERMEDIATE_DIR / "blocking"
CHUNK = 20_000


def rank_within_s1(q, *sort_keys):
    """0-based rank of each pair within its S1 group under ``sort_keys``
    (numpy lexsort order: last key is primary, before the group key)."""
    order = np.lexsort((*sort_keys, q))
    q_sorted = q[order]
    starts = np.flatnonzero(np.r_[True, q_sorted[1:] != q_sorted[:-1]])
    sizes = np.diff(np.r_[starts, len(q)])
    rank = np.empty(len(q), np.int64)
    rank[order] = np.arange(len(q)) - np.repeat(starts, sizes)
    return rank


def main():
    t0 = time.time()
    s1, pool, gt_s1, gt_pool = load_inputs()
    n_pool, n_s1 = pool.num_rows, s1.num_rows
    strategies = default_strategies()
    caps = [s.max_df for s in strategies]
    workdir = OUT_DIR / "tmp_topn"
    block_to_disk(strategies, {"val": s1}, pool, workdir, CHUNK, with_df=True)
    del pool, strategies
    gc.collect()

    gt_keys = np.sort(gt_s1.astype(np.int64) * n_pool + gt_pool)
    gt_per_s1 = np.bincount(gt_s1, minlength=n_s1)
    ns = TOP_N + [np.iinfo(np.int64).max]
    kept = {v: np.zeros((len(ns), n_s1), np.int64) for v in "AB"}
    found = {v: np.zeros((len(ns), n_s1), np.int64) for v in "AB"}
    for offset, q, c, bits, n_hits, min_df, rarity in iter_union_scored(
            "val", s1, n_pool, caps, workdir, CHUNK):
        lo, hi = offset * n_pool, (offset + CHUNK) * n_pool
        chunk_gt = gt_keys[np.searchsorted(gt_keys, lo):np.searchsorted(gt_keys, hi)]
        is_true = np.isin(q * n_pool + c, chunk_gt, assume_unique=True)
        ranks = {
            "A": rank_within_s1(q, c, min_df, -n_hits),
            "B": rank_within_s1(q, c, -rarity),
        }
        for v, rank in ranks.items():
            for i, n in enumerate(ns):
                keep = rank < n
                kept[v][i] += np.bincount(q[keep], minlength=n_s1)
                found[v][i] += np.bincount(q[keep & is_true], minlength=n_s1)

    rows = []
    has_gt = gt_per_s1 > 0
    for v, label in (("A", "agreement"), ("B", "rarity")):
        for i, n in enumerate(ns):
            k, f = kept[v][i], found[v][i]
            rows.append({
                "ranking": label, "top_n": "all" if i == len(ns) - 1 else n,
                "recall": f.sum() / len(gt_keys),
                "s1_fully_covered": (f[has_gt] == gt_per_s1[has_gt]).mean(),
                "avg_cands": k.mean(), "median": np.median(k),
                "p95": np.percentile(k, 95), "p99": np.percentile(k, 99),
                "pair_precision": f.sum() / max(k.sum(), 1),
            })
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "topn_eval.tsv", sep="\t", index=False)
    fmt = {"recall": "{:.2%}".format, "s1_fully_covered": "{:.2%}".format,
           "pair_precision": "{:.2%}".format, "avg_cands": "{:.1f}".format}
    print(table.to_string(index=False, formatters=fmt))
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
