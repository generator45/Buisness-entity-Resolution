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

from config import INTERMEDIATE_DIR, SEED, STAGING_DIR

SPLIT_PATH = INTERMEDIATE_DIR / "train" / "split.parquet"
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
