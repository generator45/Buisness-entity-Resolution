"""Stage 2c: learn the transliteration dictionary (translit_dict.py).

Uses only matcher_train S1 entities: their ground-truth pairs whose candidate
name contains non-Latin characters. Writes
data/intermediate/translit_dict.json with the mapping and a small report.

Run from code/business_entity_resolution/:
    python3 src/pipeline/run_stage2c_learn_translit_dict.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from config import INTERMEDIATE_DIR, STAGING_DIR  # noqa: E402
from split import load_matcher_split  # noqa: E402
from translit_dict import MIN_COUNT, MIN_SHARE, learn  # noqa: E402

OUT_PATH = INTERMEDIATE_DIR / "translit_dict.json"
NON_LATIN_RE = r"[^\x{0000}-\x{024F}]"


def main():
    split = load_matcher_split()
    train_ids = set(split.loc[split["split"] == "matcher_train", "entity_id"])
    gt = pq.read_table(STAGING_DIR / "train" / "ground_truth.parquet").to_pandas()
    gt = gt[gt["source1_entity_id"].isin(train_ids)]
    gt = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")
    gt = gt[gt["m"].notna() & (gt["m"] != "")]

    cols = ["entity_id", "business_name", "business_name_translit"]
    s1 = pq.read_table(STAGING_DIR / "train" / "s1.parquet", columns=cols)
    pool = pa.concat_tables(
        pq.read_table(STAGING_DIR / "train" / f"{s}.parquet", columns=cols) for s in ("s2", "s3"))
    cand = pool.take(pc.index_in(pa.array(gt["m"]), value_set=pool["entity_id"]))
    src = s1.take(pc.index_in(pa.array(gt["source1_entity_id"]), value_set=s1["entity_id"]))
    non_latin = pc.match_substring_regex(cand["business_name"], NON_LATIN_RE).to_numpy(
        zero_copy_only=False)
    s1_names = src["business_name_translit"].to_numpy(zero_copy_only=False)[non_latin]
    cand_names = cand["business_name_translit"].to_numpy(zero_copy_only=False)[non_latin]
    mapping = learn(zip(s1_names, cand_names))

    report = {"train_true_pairs": len(gt), "non_latin_true_pairs": int(non_latin.sum()),
              "mappings": len(mapping), "min_count": MIN_COUNT, "min_share": MIN_SHARE}
    frequent = {}
    for c in cand_names:
        for tok in c.split():
            if tok in mapping:
                frequent[tok] = frequent.get(tok, 0) + 1
    report["most_used"] = {k: mapping[k] for k, _ in
                           sorted(frequent.items(), key=lambda kv: -kv[1])[:40]}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"report": report, "mapping": mapping}, f, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
