"""The production blocking configuration, shared by evaluation and pipelines.

Chosen on the blocking validation split (see run_stage2_blocking_eval.py):
95.37% recall at ~186 candidates per S1 record. Every key is scoped by
country; each cap is the max document frequency of a key in the S2/S3 pool.
"""

from blocking import (
    AddressNumberKeys,
    CoreNameKey,
    KeyBlock,
    NamePairKeys,
    SpacelessNameKey,
    TokenKeys,
)
from normalize import phonetic_skeleton

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


def default_strategies(
    token_max_df=TOKEN_MAX_DF,
    core_max_df=CORE_MAX_DF,
    addr_max_df=ADDR_MAX_DF,
    pair_max_df=PAIR_MAX_DF,
    spaceless_max_df=SPACELESS_MAX_DF,
):
    """Fresh (unfitted) strategy list in production order."""
    return [
        KeyBlock(f"name_token|country[{token_max_df}]", TokenKeys(NORM), token_max_df),
        KeyBlock(
            f"core_name_phonetic|country[{core_max_df}]",
            CoreNameKey(TRANSLIT, transform=phonetic_skeleton),
            core_max_df,
        ),
        KeyBlock(
            f"addr_number_x_word_norm|country[{addr_max_df}]",
            AddressNumberKeys(ADDRESS, normalize_numbers=True),
            addr_max_df,
        ),
        KeyBlock(f"name_pair|country[{pair_max_df}]", NamePairKeys(NORM), pair_max_df),
        KeyBlock(
            f"spaceless_name|country[{spaceless_max_df}]",
            SpacelessNameKey(TRANSLIT),
            spaceless_max_df,
        ),
    ]
