"""The production blocking configuration, shared by evaluation and pipelines.

Chosen on the blocking validation split (see run_stage2_blocking_eval.py,
set ``blocking_v2_candidates`` for the per-strategy measurements). Every key
is scoped by country; each cap is the max document frequency of a key in the
S2/S3 pool.

v2 (after the French / dotted-abbreviation / ordinal normalization fixes):
- name word and word-pair keys use the accent-folded (transliterated) field,
  so "Comité" / "Comite" share keys (costs nothing on US / India)
- the core-name key maps tokens through the learned transliteration
  dictionary before the phonetic skeleton (stage 2c; +0.67 pts unique recall
  at 10.6 cands/S1) and replaces the plain phonetic core key, which it
  subsumes
- rarest-word address pairs for addresses without usable house numbers
  (+0.64 pts unique recall at 24 cands/S1)
Measured and dropped: word keys through the dictionary (no gain) and rarest
name x address word keys (0.21 pts for 16 cands/S1, ~2,200 pairs per unique
true match).

Per-record cap (run_stage2d_topn_eval.py): each S1 record keeps its TOP_N
candidates ranked by summed key rarity (blocking.iter_union_scored). On
blocking validation: 98.43% recall at 220 cands/S1 uncapped, 97.75% at 144
with TOP_N = 200 (p99 962 -> 200).
"""

from blocking import (
    AddressNumberKeys,
    AddressRarePairKeys,
    CoreNameKey,
    KeyBlock,
    NamePairKeys,
    SpacelessNameKey,
    TokenKeys,
)
from config import TRANSLIT_DICT_PATH
from normalize import phonetic_skeleton
from translit_dict import MappedPhonetic, TranslitMap

NORM, TRANSLIT = "business_name_norm", "business_name_translit"
ADDRESS = "business_address_translit"

# Columns the strategies read (besides entity_id).
BLOCKING_COLUMNS = [NORM, TRANSLIT, ADDRESS, "country"]

# token cap: 200 was picked from a 25..1000 sweep, then lowered to 100 once
# leave-one-out showed it the least efficient strategy (0.61 pts unique
# recall for ~30 cands/S1 at cap 200)
TOKEN_MAX_DF = 100
CORE_MAX_DF = 1000
ADDR_MAX_DF = 100
# pair cap from a 10..1000 sweep: 100 keeps ~half the gain of 1000 at ~10%
# of its candidate pairs
PAIR_MAX_DF = 100
SPACELESS_MAX_DF = 1000
RARE_PAIR_MAX_DF = 100
TOP_N = 200


def load_translit_map() -> TranslitMap:
    if not TRANSLIT_DICT_PATH.exists():
        raise FileNotFoundError(
            f"{TRANSLIT_DICT_PATH} missing: run src/pipeline/run_stage2c_learn_translit_dict.py")
    return TranslitMap.load(TRANSLIT_DICT_PATH)


def default_strategies(
    token_max_df=TOKEN_MAX_DF,
    core_max_df=CORE_MAX_DF,
    addr_max_df=ADDR_MAX_DF,
    pair_max_df=PAIR_MAX_DF,
    spaceless_max_df=SPACELESS_MAX_DF,
    rare_pair_max_df=RARE_PAIR_MAX_DF,
):
    """Fresh (unfitted) strategy list in production order."""
    core_transform = MappedPhonetic(load_translit_map(), phonetic_skeleton)
    return [
        KeyBlock(f"name_token_translit|country[{token_max_df}]", TokenKeys(TRANSLIT),
                 token_max_df),
        KeyBlock(
            f"core_name_dict_phonetic|country[{core_max_df}]",
            CoreNameKey(TRANSLIT, transform=core_transform),
            core_max_df,
        ),
        KeyBlock(
            f"addr_number_x_word_norm|country[{addr_max_df}]",
            AddressNumberKeys(ADDRESS, normalize_numbers=True),
            addr_max_df,
        ),
        KeyBlock(f"name_pair_translit|country[{pair_max_df}]", NamePairKeys(TRANSLIT),
                 pair_max_df),
        KeyBlock(
            f"spaceless_name|country[{spaceless_max_df}]",
            SpacelessNameKey(TRANSLIT),
            spaceless_max_df,
        ),
        KeyBlock(f"addr_rare_pairs|country[{rare_pair_max_df}]", AddressRarePairKeys(ADDRESS),
                 rare_pair_max_df),
    ]
