"""Deterministic train/validation split of Source 1 entities.

The split is by Source 1 entity: a validation S1 keeps all of its ground-truth
matches, and blocking for it is always evaluated against the *full* train
S2/S3 pool (including the unmatched records), so validation numbers reflect
the real retrieval difficulty rather than a trimmed pool.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import (
    INTERMEDIATE_DIR,
    MATCHER_SAMPLE_FRACTION,
    MATCHER_SEED,
    SEED,
    STAGING_DIR,
)

SPLIT_PATH = INTERMEDIATE_DIR / "train" / "split.parquet"
MATCHER_SPLIT_PATH = INTERMEDIATE_DIR / "train" / "matcher_split.parquet"
VAL_FRACTION = 0.10


def make_split(val_fraction: float = VAL_FRACTION, seed: int = SEED) -> pd.DataFrame:
    ids = pq.read_table(STAGING_DIR / "train" / "s1.parquet", columns=["entity_id"])
    ids = ids["entity_id"].to_numpy(zero_copy_only=False)
    rng = np.random.default_rng(seed)
    is_val = rng.random(len(ids)) < val_fraction
    df = pd.DataFrame({"entity_id": ids, "split": np.where(is_val, "val", "train")})
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), SPLIT_PATH)
    return df


def load_split() -> pd.DataFrame:
    """Load the persisted split, creating it on first use."""
    if not SPLIT_PATH.exists():
        return make_split()
    return pq.read_table(SPLIT_PATH).to_pandas()


def make_matcher_split(
    fraction: float = MATCHER_SAMPLE_FRACTION, seed: int = MATCHER_SEED
) -> pd.DataFrame:
    """Disjoint S1 samples for matcher training and matcher validation.

    Each sample is ``fraction`` of all train S1 entities. Both are drawn from
    the entities *outside* the blocking validation split (whose records were
    used to tune blocking caps), in one seeded permutation: the first slice
    is ``matcher_train``, the next ``matcher_val``, so they cannot overlap.
    Everything else is ``unused``.
    """
    blocking = load_split()
    eligible = blocking.loc[blocking["split"] == "train", "entity_id"].to_numpy()
    n = int(round(fraction * len(blocking)))
    if 2 * n > len(eligible):
        raise ValueError(f"cannot draw 2 x {n} S1 entities from {len(eligible)}")
    order = np.random.default_rng(seed).permutation(len(eligible))
    label = np.full(len(blocking), "unused", dtype=object)
    pos = pd.Index(blocking["entity_id"]).get_indexer(eligible[order[: 2 * n]])
    label[pos[:n]] = "matcher_train"
    label[pos[n:]] = "matcher_val"
    df = pd.DataFrame({"entity_id": blocking["entity_id"], "split": label})
    MATCHER_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), MATCHER_SPLIT_PATH)
    return df


def load_matcher_split() -> pd.DataFrame:
    """Load the persisted matcher split, creating it on first use."""
    if not MATCHER_SPLIT_PATH.exists():
        return make_matcher_split()
    return pq.read_table(MATCHER_SPLIT_PATH).to_pandas()
