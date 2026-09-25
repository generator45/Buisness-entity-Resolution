"""Blocking strategy configurations shared by evaluation and production runs.

Each set is an ordered list of KeyBlock strategies; order only affects how
the evaluation attributes "new" matches, not which candidates are produced.
"""

import functools

from blocking import (
    AddressNumberKeys,
    CoreNameKey,
    ExactKey,
    KeyBlock,
    NamePairKeys,
    SpacelessNameKey,
    TokenKeys,
)
from normalize import consonant_skeleton, phonetic_skeleton

# Default frequency caps, each chosen from a validation sweep.
MAX_DF = 1000
TOKEN_MAX_DF = 200
ADDR_MAX_DF = 100
PAIR_MAX_DF = 100

# Staging columns the key functions read.
KEY_COLUMNS = [
    "entity_id",
    "country",
    "business_name_norm",
    "business_name_translit",
    "business_address_translit",
]

NORM, TRANSLIT = "business_name_norm", "business_name_translit"


def strategy_sets(max_df=MAX_DF, token_max_df=TOKEN_MAX_DF, addr_max_df=ADDR_MAX_DF,
                  pair_max_df=PAIR_MAX_DF):
    """Named, ordered strategy lists. Order sets incremental attribution."""
    return {
        # the v1 baseline: global (not country-scoped) keys
        "baseline_global": [
            KeyBlock("exact_name", ExactKey(NORM, by_country=False)),
            KeyBlock(f"name_token[{max_df}]", TokenKeys(NORM, by_country=False), max_df),
        ],
        # every key scoped by country. Pruned by leave-one-out on validation:
        # - name_token over translit / skeleton fields: +0.03 pts recall for
        #   ~53 cands/S1 and ~3.5 GB RAM
        # - exact_name, core_name, core_name_translit: each lost <= 16 true
        #   matches when removed; subsumed by the skeleton core key
        "full": [
            # token cap from a 25..1000 sweep: 200 keeps 93.6% recall (vs
            # 94.1% at 1000) at 189 cands/S1 (vs 291)
            KeyBlock(
                f"name_token|country[{token_max_df}]", TokenKeys(NORM), token_max_df
            ),
            KeyBlock(
                f"core_name_phonetic|country[{max_df}]",
                CoreNameKey(TRANSLIT, transform=phonetic_skeleton),
                max_df,
            ),
            KeyBlock(
                f"addr_number_x_word_norm|country[{addr_max_df}]",
                AddressNumberKeys("business_address_translit", normalize_numbers=True),
                addr_max_df,
            ),
            # pair cap from a 10..1000 sweep: 100 keeps ~half the gain of
            # 1000 at ~10% of its candidate pairs
            KeyBlock(f"name_pair|country[{pair_max_df}]", NamePairKeys(NORM), pair_max_df),
            KeyBlock(f"spaceless_name|country[{max_df}]", SpacelessNameKey(TRANSLIT), max_df),
        ],
        # before/after for the cheap fixes: each new variant sits right after
        # the version it replaces, so its new_true is exactly what the fix adds
        "cheap_fixes": [
            KeyBlock(
                f"name_token|country[{token_max_df}]", TokenKeys(NORM), token_max_df
            ),
            KeyBlock(
                f"core_name_skeleton|country[{max_df}]",
                CoreNameKey(TRANSLIT, transform=consonant_skeleton),
                max_df,
            ),
            KeyBlock(
                f"core_name_skeleton+indic|country[{max_df}]",
                CoreNameKey(
                    TRANSLIT, transform=functools.partial(consonant_skeleton, indic_folds=True)
                ),
                max_df,
            ),
            KeyBlock(
                f"core_name_skeleton+indic+digits|country[{max_df}]",
                CoreNameKey(TRANSLIT, transform=phonetic_skeleton),
                max_df,
            ),
            KeyBlock(
                f"addr_number_x_word|country[{addr_max_df}]",
                AddressNumberKeys("business_address_translit"),
                addr_max_df,
            ),
            KeyBlock(
                f"addr_number_x_word_norm|country[{addr_max_df}]",
                AddressNumberKeys("business_address_translit", normalize_numbers=True),
                addr_max_df,
            ),
            KeyBlock(f"name_pair|country[{pair_max_df}]", NamePairKeys(NORM), pair_max_df),
            KeyBlock(f"spaceless_name|country[{max_df}]", SpacelessNameKey(TRANSLIT), max_df),
        ],
    }
