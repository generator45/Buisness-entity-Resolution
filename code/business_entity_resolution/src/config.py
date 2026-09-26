"""Paths, seeds, and shared constants for the entity resolution pipeline."""

from pathlib import Path

# code/business_entity_resolution/src/config.py -> project root is 3 levels up
PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "6ab10eb3b23ba_student_resource"
    / "student_resource"
    / "dataset"
)
RAW_TRAIN_DIR = RAW_DATA_DIR / "train"
RAW_TEST_DIR = RAW_DATA_DIR / "test"

DATA_DIR = PROJECT_ROOT / "data"
STAGING_DIR = DATA_DIR / "staging"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
MARTS_DIR = DATA_DIR / "marts"

OUTPUT_DIR = PROJECT_ROOT / "output"

SEED = 42

# Sources for which raw files exist per split.
SOURCE_NAMES = ("s1", "s2", "s3")

SOURCE_FILES = {
    "train": {
        "s1": RAW_TRAIN_DIR / "train_source1.tsv",
        "s2": RAW_TRAIN_DIR / "train_source2.tsv",
        "s3": RAW_TRAIN_DIR / "train_source3.tsv",
        "ground_truth": RAW_TRAIN_DIR / "train_ground_truth.tsv",
    },
    "test": {
        "s1": RAW_TEST_DIR / "test_source1.tsv",
        "s2": RAW_TEST_DIR / "test_source2.tsv",
        "s3": RAW_TEST_DIR / "test_source3.tsv",
    },
}

SOURCE_SCHEMA = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_SCHEMA = ["source1_entity_id", "matched_entity_ids"]

CHUNK_SIZE = 250_000

# --- matcher training-data preparation (stage 3) ---
# Each matcher sample is this fraction of *all* train S1 entities, drawn
# (disjointly) from those outside the blocking validation split, which was
# used to tune blocking caps.
MATCHER_SAMPLE_FRACTION = 0.10
MATCHER_SEED = SEED + 1
MATCHER_DIR = MARTS_DIR / "matcher"
# pairs processed per feature chunk
PAIR_CHUNK_SIZE = 2_000_000
# strong_name_and_address_overlap = name_jaccard >= X and address_jaccard >= Y
STRONG_NAME_JACCARD = 0.5
STRONG_ADDRESS_JACCARD = 0.5
