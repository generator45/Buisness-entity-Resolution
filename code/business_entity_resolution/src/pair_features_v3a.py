"""V3a add-on features, aimed at the error modes found on Indian records.

Computed on top of the V2 datasets (some inputs are read from the V2 columns)
and stored as a separate, row-aligned add-on file:

- name-based S1 context: ``name_best_sim`` = max(name_token_set_ratio,
  name_spaceless_ratio, name_phonetic_jaccard), and the candidate's rank and
  gap on it within its S1's full candidate list. V2's only rank used
  name x address similarity, which is 0 whenever an address is missing.
- name differences: idf (scaled to [0, 1] by log N) of the rarest meaningful
  phonetic-skeleton name token found on one side but not the other, per side.
  V2 measured only what names share, so "Great Logistics" vs "Great
  Foundation" at the same shared address looked like a match.
- script mismatch: exactly one side's name (address) contains non-Latin
  characters, telling the model when token features are less reliable. Uses
  the raw fields; country-agnostic.
- robust house numbers: shared normalized numbers / the smaller number set,
  and their Jaccard. Injected numbers on one side ("DOOR NO 75", "BLOCK
  G-950") break V2's all-or-nothing number conflict flag but not these.
"""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from normalize import normalize_house_number, phonetic_skeleton
from pair_features import ADDRESS_FIELD, NAME_FIELD, TokenSets, _safe_div
from pair_features_v2 import PHONETIC_STOP_TOKENS

RAW_NAME, RAW_ADDRESS = "business_name", "business_address"
NON_LATIN_RE = r"[^\x{0000}-\x{024F}]"

V3A_DTYPES = {
    "name_best_sim": np.float32,
    "name_rank_in_s1": np.int32,
    "name_gap_to_s1_best": np.float32,
    "name_unmatched_max_idf_s1": np.float32,
    "name_unmatched_max_idf_cand": np.float32,
    "name_script_mismatch": np.int8,
    "address_script_mismatch": np.int8,
    "address_number_containment": np.float32,
    "address_number_jaccard": np.float32,
}
V3A_COLUMNS = list(V3A_DTYPES)
V3A_BINARY = [c for c, t in V3A_DTYPES.items() if t == np.int8]
V3A_UNIT_INTERVAL = [c for c, t in V3A_DTYPES.items() if t == np.float32]
# V2 columns this module reads as inputs
V2_INPUTS = ["name_token_set_ratio", "name_spaceless_ratio", "name_phonetic_jaccard",
             "address_shared_numeric_token_count"]


def _non_latin(table: pa.Table, field: str) -> np.ndarray:
    col = table[field]
    col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    return pc.match_substring_regex(col, NON_LATIN_RE).to_numpy(zero_copy_only=False)


class FieldIndexV3a:
    """Per-side structures for the V3a features.

    Phonetic name tokens need CSR tokens *and* a lookup on both sides, since
    unmatched tokens are searched in both directions; idf comes from the pool.
    """

    def __init__(self, table: pa.Table, side: str, pool_index=None):
        s1 = side == "s1"
        idf = {"idf_from": pool_index.name_phonetic} if s1 else {"compute_idf": True}
        self.name_phonetic = TokenSets(
            table, NAME_FIELD, transform=phonetic_skeleton, stop=PHONETIC_STOP_TOKENS,
            with_tokens=True, with_lookup=True, **idf,
        )
        self.addr_numbers = TokenSets(
            table, ADDRESS_FIELD, transform=normalize_house_number, with_tokens=False,
        )
        self.name_non_latin = _non_latin(table, RAW_NAME)
        self.addr_non_latin = _non_latin(table, RAW_ADDRESS)


def _unmatched_max_idf(a: TokenSets, b: TokenSets, qa, qb, max_idf: float):
    """Per pair: idf of the rarest meaningful token of ``a`` missing from ``b``."""
    pair, tokens, meaningful, idf = a.expand(qa, with_idf=True)
    miss = meaningful & ~b.contains(qb[pair], tokens)
    best = np.zeros(len(qa))
    np.maximum.at(best, pair[miss], idf[miss])
    return best / max_idf


def rank_and_gap(qa: np.ndarray, score: np.ndarray):
    """Rank (1 + number of strictly higher scores) and gap to the best score
    within each S1 group; ``qa`` must hold complete groups in sorted order."""
    n = len(qa)
    starts = np.flatnonzero(np.r_[True, qa[1:] != qa[:-1]])
    group = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, n]))
    best = np.maximum.reduceat(score, starts)
    order = np.lexsort((-score, qa))
    s_sorted, g_sorted = score[order], group[order]
    new_value = np.r_[True, (g_sorted[1:] != g_sorted[:-1]) | (s_sorted[1:] != s_sorted[:-1])]
    first_of_value = np.maximum.accumulate(np.where(new_value, np.arange(n), 0))
    rank = np.empty(n, np.int64)
    rank[order] = first_of_value - starts[g_sorted] + 1
    return rank, best[group] - score


def compute_features_v3a(s1: FieldIndexV3a, pool: FieldIndexV3a, qa, qb, v2: dict):
    """V3a columns for pairs (S1 local row qa[i], pool row qb[i]).

    ``v2`` holds the V2_INPUTS columns for the same pairs; ``qa`` must hold
    complete S1 groups in sorted order.
    """
    best_sim = np.maximum.reduce([
        v2["name_token_set_ratio"], v2["name_spaceless_ratio"], v2["name_phonetic_jaccard"],
    ]).astype(np.float64)
    rank, gap = rank_and_gap(qa, best_sim)
    max_idf = pool.name_phonetic.max_idf

    shared = v2["address_shared_numeric_token_count"].astype(np.float64)
    na, nb = s1.addr_numbers.count[qa], pool.addr_numbers.count[qb]

    feats = {
        "name_best_sim": best_sim,
        "name_rank_in_s1": rank,
        "name_gap_to_s1_best": gap,
        "name_unmatched_max_idf_s1": _unmatched_max_idf(
            s1.name_phonetic, pool.name_phonetic, qa, qb, max_idf),
        "name_unmatched_max_idf_cand": _unmatched_max_idf(
            pool.name_phonetic, s1.name_phonetic, qb, qa, max_idf),
        "name_script_mismatch": s1.name_non_latin[qa] != pool.name_non_latin[qb],
        "address_script_mismatch": s1.addr_non_latin[qa] != pool.addr_non_latin[qb],
        "address_number_containment": _safe_div(shared, np.minimum(na, nb)),
        "address_number_jaccard": _safe_div(shared, na + nb - shared),
    }
    return {c: np.asarray(v).astype(V3A_DTYPES[c]) for c, v in feats.items()}
