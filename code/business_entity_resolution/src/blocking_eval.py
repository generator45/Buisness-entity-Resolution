"""Blocking evaluation: recall, candidate volume, incremental contribution.

Strategies are evaluated cumulatively in list order. A candidate pair (or a
true match) is credited to the *first* strategy in the list that retrieved
it, so each row of the report answers: "what did adding this strategy buy on
top of everything above it, and at what candidate cost?"
"""

import time

import numpy as np
import pandas as pd

from blocking import generate_candidates

# index of the lowest set bit for every uint16 value (0 -> -1, unused)
_LOWEST_BIT = np.array(
    [-1] + [(v & -v).bit_length() - 1 for v in range(1, 1 << 16)], dtype=np.int8
)


def evaluate_blocking(strategies, s1, n_pool, gt_s1_rows, gt_pool_rows, chunk_size=20_000):
    """Run blocking over ``s1`` and score it against ground-truth pairs.

    ``gt_s1_rows`` / ``gt_pool_rows`` are parallel arrays of true
    (s1_row, pool_row) pairs, as row indices into ``s1`` and the pool.
    """
    n_s1, n_strat = s1.num_rows, len(strategies)
    gt_keys = np.sort(gt_s1_rows.astype(np.int64) * n_pool + gt_pool_rows)
    gt_per_s1 = np.bincount(gt_s1_rows, minlength=n_s1)

    counts = np.zeros((n_s1, n_strat), dtype=np.int64)  # new pairs per S1, by first strategy
    found = np.zeros((n_s1, n_strat), dtype=np.int32)  # new true matches per S1
    solo_pairs = np.zeros(n_strat, dtype=np.int64)
    solo_true = np.zeros(n_strat, dtype=np.int64)
    only_pairs = np.zeros(n_strat, dtype=np.int64)  # found by this strategy alone
    only_true = np.zeros(n_strat, dtype=np.int64)
    hit_keys = []

    t0 = time.time()
    for offset, q, c, bits in generate_candidates(strategies, s1, n_pool, chunk_size):
        lo, hi = offset * n_pool, (offset + chunk_size) * n_pool
        chunk_gt = gt_keys[np.searchsorted(gt_keys, lo):np.searchsorted(gt_keys, hi)]
        keys = q * n_pool + c
        is_true = np.isin(keys, chunk_gt, assume_unique=True)
        hit_keys.append(keys[is_true])
        first = _LOWEST_BIT[bits]
        for k in range(n_strat):
            sel = first == k
            counts[:, k] += np.bincount(q[sel], minlength=n_s1)
            found[:, k] += np.bincount(q[sel & is_true], minlength=n_s1)
            has = (bits >> k) & 1 == 1
            solo_pairs[k] += has.sum()
            solo_true[k] += (has & is_true).sum()
            only = bits == (1 << k)
            only_pairs[k] += only.sum()
            only_true[k] += (only & is_true).sum()
    elapsed = time.time() - t0

    n_gt = len(gt_keys)
    cum_counts = np.cumsum(counts, axis=1)
    cum_found = np.cumsum(found, axis=1)
    has_gt = gt_per_s1 > 0
    rows = []
    for k, strat in enumerate(strategies):
        cc = cum_counts[:, k]
        rows.append({
            "strategy": ("" if k == 0 else "+ ") + strat.name,
            "recall": cum_found[:, k].sum() / n_gt,
            "s1_fully_covered": (cum_found[has_gt, k] == gt_per_s1[has_gt]).mean(),
            "avg_cands": cc.mean(),
            "median": np.median(cc),
            "p95": np.percentile(cc, 95),
            "p99": np.percentile(cc, 99),
            "max": cc.max(),
            "zero_cand_s1": (cc == 0).mean(),
            "new_true": int(found[:, k].sum()),
            "new_pairs": int(counts[:, k].sum()),
            "solo_recall": solo_true[k] / n_gt,
            "solo_pairs": int(solo_pairs[k]),
            # leave-one-out: what removing only this strategy would lose
            "loo_recall_loss": only_true[k] / n_gt,
            "loo_pairs_saved": int(only_pairs[k]),
        })
    report = pd.DataFrame(rows)
    hit_keys = np.concatenate(hit_keys) if hit_keys else np.zeros(0, np.int64)
    missed = gt_keys[~np.isin(gt_keys, hit_keys, assume_unique=True)]
    return report, missed // n_pool, missed % n_pool, elapsed


def format_report(report: pd.DataFrame) -> str:
    fmt = report.copy()
    for col in ("recall", "s1_fully_covered", "zero_cand_s1", "solo_recall", "loo_recall_loss"):
        fmt[col] = (fmt[col] * 100).map("{:.2f}%".format)
    fmt["avg_cands"] = fmt["avg_cands"].map("{:.1f}".format)
    for col in ("median", "p95", "p99", "max", "new_true", "new_pairs", "solo_pairs",
                "loo_pairs_saved"):
        fmt[col] = fmt[col].map("{:,.0f}".format)
    return fmt.to_string(index=False)
