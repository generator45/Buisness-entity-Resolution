"""Stage 1: ingest raw TSVs, normalize name/address, write staging Parquet.

Reads dataset/{train,test}/*_source{1,2,3}.tsv (and train_ground_truth.tsv)
in fixed-size chunks and writes normalized Parquet to
data/staging/{train,test}/{s1,s2,s3}.parquet, plus
data/staging/train/ground_truth.parquet. Pure function of the raw inputs
and normalize.py — safe to delete data/staging/ and re-run at any time.

Run from code/business_entity_resolution/:
    python3 src/pipeline/run_stage1_ingest.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import SOURCE_FILES, SOURCE_NAMES, STAGING_DIR  # noqa: E402
from io_utils import (  # noqa: E402
    DQStats,
    ParquetChunkWriter,
    iter_ground_truth_chunks,
    iter_source_chunks,
)
from normalize import (  # noqa: E402
    extract_postal_code_proxy,
    normalize_business_address,
    normalize_business_name,
    transliterate_business_address,
    transliterate_business_name,
)


def _normalize_chunk(chunk):
    chunk = chunk.copy()
    chunk["business_name_norm"] = chunk["business_name"].map(normalize_business_name)
    chunk["business_address_norm"] = chunk["business_address"].map(
        normalize_business_address
    )
    chunk["business_name_translit"] = chunk["business_name"].map(
        transliterate_business_name
    )
    chunk["business_address_translit"] = chunk["business_address"].map(
        transliterate_business_address
    )
    chunk["postal_proxy"] = chunk["business_address"].map(extract_postal_code_proxy)
    return chunk


def ingest_source(split: str, source: str) -> DQStats:
    raw_path = SOURCE_FILES[split][source]
    out_path = STAGING_DIR / split / f"{source}.parquet"
    stats = DQStats("entity_id")
    t0 = time.time()
    with ParquetChunkWriter(out_path) as writer:
        for chunk in iter_source_chunks(raw_path):
            chunk = _normalize_chunk(chunk)
            stats.update(chunk)
            writer.write(chunk)
    print(
        stats.report(f"{split}/{source}")
        + f" | {time.time() - t0:.1f}s -> {out_path}"
    )
    return stats


def ingest_ground_truth() -> DQStats:
    raw_path = SOURCE_FILES["train"]["ground_truth"]
    out_path = STAGING_DIR / "train" / "ground_truth.parquet"
    stats = DQStats("source1_entity_id")
    t0 = time.time()
    with ParquetChunkWriter(out_path) as writer:
        for chunk in iter_ground_truth_chunks(raw_path):
            stats.update(chunk)
            writer.write(chunk)
    print(
        stats.report("train/ground_truth")
        + f" | {time.time() - t0:.1f}s -> {out_path}"
    )
    return stats


def main():
    print("Stage 1: ingestion + normalization")
    for split in ("train", "test"):
        for source in SOURCE_NAMES:
            ingest_source(split, source)
    ingest_ground_truth()
    print("Done.")


if __name__ == "__main__":
    main()
