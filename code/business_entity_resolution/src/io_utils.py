"""Chunked TSV I/O, schema contracts, and lightweight data-quality checks.

Every source file is read in fixed-size chunks and written incrementally to
Parquet via a streaming pyarrow writer, so no stage ever needs to hold a
full multi-million-row source file in memory at once.
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import CHUNK_SIZE, GROUND_TRUTH_SCHEMA, SOURCE_SCHEMA


class SchemaError(ValueError):
    """Raised when a raw TSV's header doesn't match the expected contract."""


def _validate_header(actual_cols, expected_cols, path: Path):
    if list(actual_cols) != list(expected_cols):
        raise SchemaError(
            f"{path}: unexpected columns {list(actual_cols)!r}, "
            f"expected {list(expected_cols)!r}. "
            "Reading a .tsv without sep='\\t' silently produces a single "
            "column containing the whole line — check that first."
        )


def iter_source_chunks(path: Path, chunksize: int = CHUNK_SIZE):
    """Yield validated chunks of a *_source{1,2,3}.tsv file."""
    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],
        chunksize=chunksize,
    )
    first = True
    for chunk in reader:
        if first:
            _validate_header(chunk.columns, SOURCE_SCHEMA, path)
            first = False
        yield chunk


def iter_ground_truth_chunks(path: Path, chunksize: int = CHUNK_SIZE):
    """Yield validated chunks of train_ground_truth.tsv."""
    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],
        chunksize=chunksize,
    )
    first = True
    for chunk in reader:
        if first:
            _validate_header(chunk.columns, GROUND_TRUTH_SCHEMA, path)
            first = False
        yield chunk


class DQStats:
    """Accumulates simple data-quality counters across chunks."""

    def __init__(self, id_col: str):
        self.id_col = id_col
        self.rows = 0
        self.null_ids = 0
        self.empty_business_name = 0
        self.empty_business_address = 0

    def update(self, chunk: pd.DataFrame):
        self.rows += len(chunk)
        self.null_ids += int(
            chunk[self.id_col].isna().sum() + (chunk[self.id_col] == "").sum()
        )
        if "business_name" in chunk.columns:
            self.empty_business_name += int((chunk["business_name"] == "").sum())
        if "business_address" in chunk.columns:
            self.empty_business_address += int(
                (chunk["business_address"] == "").sum()
            )

    def report(self, label: str) -> str:
        return (
            f"  {label}: {self.rows:,} rows"
            f" | null/empty id: {self.null_ids:,}"
            f" | empty business_name: {self.empty_business_name:,}"
            f" | empty business_address: {self.empty_business_address:,}"
        )


class ParquetChunkWriter:
    """Streams DataFrame chunks to a single Parquet file incrementally."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = None

    def write(self, df: pd.DataFrame):
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema)
        self._writer.write_table(table)

    def close(self):
        if self._writer is not None:
            self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
