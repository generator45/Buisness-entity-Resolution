"""V2 pairwise features: the V1 set plus features aimed at V1's error modes.

Added on top of pair_features.FEATURE_COLUMNS (same text fields and token
definitions; see that module's docstring):

- missingness: name / address empty on either side (V1 read a missing
  address as "different address")
- containment: shared tokens / size of the smaller set (partial addresses,
  extra words)
- name_meaningful_jaccard: Jaccard over meaningful tokens only, so shared
  legal forms ("LLC") no longer count as name agreement
- idf-weighted overlap: Jaccard weighted by token rarity in the train pool,
  and the rarest shared token's idf (scaled to [0, 1] by log N)
- name_phonetic_jaccard: meaningful-token Jaccard over phonetic skeletons
  (cross-script transliteration drift, typos)
- fuzzy strings (rapidfuzz): Jaro-Winkler and token-set ratio of names,
  token-set ratio of addresses, and a similarity / containment check on
  the spaceless core name (domain and run-together names: "Apex & Sons"
  vs "apexsons.com")
- house numbers / postal code: both present, conflict (both present, none
  shared), postal match / conflict
- S1 context (computed by the caller over each S1's full candidate list):
  candidate count, rank and gap of name_address_similarity_product

All values are 0 when an input is empty; fuzzy scores are forced to 0 when
either string is empty (rapidfuzz would score "" vs "" as identical).
"""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from rapidfuzz import fuzz
from rapidfuzz.distance import Indel, JaroWinkler
from rapidfuzz.process import cpdist

from normalize import phonetic_skeleton
from pair_features import (
    ADDRESS_FIELD,
    FEATURE_DTYPES,
    NAME_FIELD,
    NAME_STOP_TOKENS,
    FieldIndex,
    StringStats,
    TokenSets,
    _overlap,
    _safe_div,
    compute_features,
)

# spaceless core names shorter than this are too generic for containment
MIN_SPACELESS_CONTAINS = 4

CONTEXT_DTYPES = {
    "s1_candidate_count": np.int32,
    "sim_rank_in_s1": np.int32,
    "sim_gap_to_s1_best": np.float32,
}
V2_NEW_DTYPES = {
    "name_missing_either": np.int8,
    "address_missing_either": np.int8,
    "name_meaningful_jaccard": np.float32,
    "name_token_containment": np.float32,
    "address_token_containment": np.float32,
    "name_idf_jaccard": np.float32,
    "name_max_shared_idf": np.float32,
    "address_idf_jaccard": np.float32,
    "address_max_shared_idf": np.float32,
    "name_phonetic_jaccard": np.float32,
    "name_jaro_winkler": np.float32,
    "name_token_set_ratio": np.float32,
    "name_spaceless_ratio": np.float32,
    "name_spaceless_contains": np.int8,
    "address_token_set_ratio": np.float32,
    "address_numbers_both_present": np.int8,
    "address_number_conflict": np.int8,
    "postal_match": np.int8,
    "postal_conflict": np.int8,
    **CONTEXT_DTYPES,
}
V2_FEATURE_DTYPES = {**FEATURE_DTYPES, **V2_NEW_DTYPES}
V2_FEATURE_COLUMNS = list(V2_FEATURE_DTYPES)
V2_BINARY_FEATURES = [c for c, t in V2_FEATURE_DTYPES.items() if t == np.int8]
V2_UNIT_INTERVAL_FEATURES = [c for c, t in V2_FEATURE_DTYPES.items() if t == np.float32]
PHONETIC_STOP_TOKENS = frozenset(phonetic_skeleton(t) for t in NAME_STOP_TOKENS)


def spaceless_core(table: pa.Table, field: str = NAME_FIELD) -> pa.Array:
    """Name with legal-form / web tokens dropped and spaces removed."""
    col = table[field]
    col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    lists = pc.split_pattern(col, " ")
    flat = pc.list_flatten(lists)
    parents = pc.list_parent_indices(lists).to_numpy()
    stop = pa.array(sorted(NAME_STOP_TOKENS))
    keep = pc.and_(pc.invert(pc.is_in(flat, value_set=stop)), pc.not_equal(flat, ""))
    counts = np.bincount(parents[keep.to_numpy(zero_copy_only=False)], minlength=table.num_rows)
    offsets = pa.array(np.r_[0, np.cumsum(counts)].astype(np.int32))
    return pc.binary_join(pa.ListArray.from_arrays(offsets, flat.filter(keep)), "")


class FieldIndexV2(FieldIndex):
    """FieldIndex plus the structures the V2 features need."""

    def __init__(self, table: pa.Table, side: str, pool_index=None):
        s1 = side == "s1"
        super().__init__(table, side, with_idf=True, idf_from=pool_index if s1 else None)
        self.name_phonetic = TokenSets(
            table, NAME_FIELD, transform=phonetic_skeleton, stop=PHONETIC_STOP_TOKENS,
            with_tokens=s1, with_lookup=not s1,
        )
        self.postal = StringStats(table, "postal_proxy")
        self.name_text = _combined(table[NAME_FIELD])
        self.addr_text = _combined(table[ADDRESS_FIELD])
        self.spaceless = spaceless_core(table)


def _combined(col):
    return col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col


def _idf_overlap(a: TokenSets, b: TokenSets, qa, qb):
    """Shared meaningful-token idf per pair: (sum, max)."""
    pair, tokens, meaningful, idf = a.expand(qa, with_idf=True)
    hit = b.contains(qb[pair], tokens) & meaningful
    n = len(qa)
    total = np.bincount(pair[hit], weights=idf[hit], minlength=n)
    best = np.zeros(n)
    np.maximum.at(best, pair[hit], idf[hit])
    return total, best


def _fuzzy(scorer, a: pa.Array, b: pa.Array, qa, qb, scale=1.0):
    """Element-wise rapidfuzz score, 0 where either string is empty."""
    left, right = a.take(pa.array(qa)), b.take(pa.array(qb))
    scores = cpdist(left.to_pylist(), right.to_pylist(), scorer=scorer, workers=-1)
    both = (pc.utf8_length(left).to_numpy() > 0) & (pc.utf8_length(right).to_numpy() > 0)
    return np.where(both, np.asarray(scores, np.float64) / scale, 0.0)


def _contains(a: pa.Array, b: pa.Array, qa, qb):
    left = a.take(pa.array(qa)).to_pylist()
    right = b.take(pa.array(qb)).to_pylist()
    return np.fromiter(
        (
            min(len(x), len(y)) >= MIN_SPACELESS_CONTAINS and (x in y or y in x)
            for x, y in zip(left, right)
        ),
        dtype=bool, count=len(left),
    )


def context_features(qa: np.ndarray, sim: np.ndarray):
    """Per-S1 context over a chunk that holds complete S1 candidate groups.

    ``qa`` must be sorted (pairs grouped by S1). Rank is 1 + the number of
    the S1's candidates with a strictly higher score (ties share a rank).
    """
    n = len(qa)
    starts = np.flatnonzero(np.r_[True, qa[1:] != qa[:-1]])
    sizes = np.diff(np.r_[starts, n])
    group = np.repeat(np.arange(len(starts)), sizes)
    best = np.maximum.reduceat(sim, starts)
    order = np.lexsort((-sim, qa))
    s_sorted, g_sorted = sim[order], group[order]
    new_value = np.r_[True, (g_sorted[1:] != g_sorted[:-1]) | (s_sorted[1:] != s_sorted[:-1])]
    first_of_value = np.maximum.accumulate(np.where(new_value, np.arange(n), 0))
    rank_sorted = first_of_value - starts[g_sorted] + 1
    rank = np.empty(n, np.int64)
    rank[order] = rank_sorted
    feats = {
        "s1_candidate_count": np.repeat(sizes, sizes),
        "sim_rank_in_s1": rank,
        "sim_gap_to_s1_best": best[group] - sim,
    }
    return {c: v.astype(CONTEXT_DTYPES[c]) for c, v in feats.items()}


def compute_features_v2(s1: FieldIndexV2, pool: FieldIndexV2, qa: np.ndarray, qb: np.ndarray):
    """V1 + V2 feature columns for pairs (S1 local row qa[i], pool row qb[i]).

    ``qa`` must hold complete S1 groups in sorted order (for the context
    features).
    """
    feats = compute_features(s1, pool, qa, qb)

    name_n = (s1.name_tokens.count[qa], pool.name_tokens.count[qb])
    addr_n = (s1.addr_tokens.count[qa], pool.addr_tokens.count[qb])
    name_m = (s1.name_tokens.meaningful_count[qa], pool.name_tokens.meaningful_count[qb])
    name_shared = feats["name_token_overlap_count"].astype(np.float64)
    addr_shared = feats["address_token_overlap_count"].astype(np.float64)
    name_m_shared = feats["name_shared_meaningful_token_count"].astype(np.float64)

    name_idf_sum, name_idf_max = _idf_overlap(s1.name_tokens, pool.name_tokens, qa, qb)
    addr_idf_sum, addr_idf_max = _idf_overlap(s1.addr_tokens, pool.addr_tokens, qa, qb)
    name_idf_union = s1.name_tokens.idf_sum[qa] + pool.name_tokens.idf_sum[qb] - name_idf_sum
    addr_idf_union = s1.addr_tokens.idf_sum[qa] + pool.addr_tokens.idf_sum[qb] - addr_idf_sum

    _, phon_shared = _overlap(s1.name_phonetic, pool.name_phonetic, qa, qb)
    phon_union = (s1.name_phonetic.meaningful_count[qa]
                  + pool.name_phonetic.meaningful_count[qb] - phon_shared)

    num_a = s1.addr_numbers.count[qa] > 0
    num_b = pool.addr_numbers.count[qb] > 0
    both_numbers = num_a & num_b
    postal_both = s1.postal.nonempty[qa] & pool.postal.nonempty[qb]
    postal_equal = s1.postal.hash[qa] == pool.postal.hash[qb]

    new = {
        "name_missing_either": ~(s1.name.nonempty[qa] & pool.name.nonempty[qb]),
        "address_missing_either": ~(s1.addr.nonempty[qa] & pool.addr.nonempty[qb]),
        "name_meaningful_jaccard": _safe_div(name_m_shared, name_m[0] + name_m[1] - name_m_shared),
        "name_token_containment": _safe_div(name_shared, np.minimum(*name_n)),
        "address_token_containment": _safe_div(addr_shared, np.minimum(*addr_n)),
        "name_idf_jaccard": _safe_div(name_idf_sum, name_idf_union),
        "name_max_shared_idf": name_idf_max / pool.name_tokens.max_idf,
        "address_idf_jaccard": _safe_div(addr_idf_sum, addr_idf_union),
        "address_max_shared_idf": addr_idf_max / pool.addr_tokens.max_idf,
        "name_phonetic_jaccard": _safe_div(phon_shared, phon_union),
        "name_jaro_winkler": _fuzzy(JaroWinkler.normalized_similarity,
                                    s1.name_text, pool.name_text, qa, qb),
        "name_token_set_ratio": _fuzzy(fuzz.token_set_ratio,
                                       s1.name_text, pool.name_text, qa, qb, scale=100.0),
        "name_spaceless_ratio": _fuzzy(Indel.normalized_similarity,
                                       s1.spaceless, pool.spaceless, qa, qb),
        "name_spaceless_contains": _contains(s1.spaceless, pool.spaceless, qa, qb),
        "address_token_set_ratio": _fuzzy(fuzz.token_set_ratio,
                                          s1.addr_text, pool.addr_text, qa, qb, scale=100.0),
        "address_numbers_both_present": both_numbers,
        "address_number_conflict": both_numbers & (feats["address_shared_numeric_token_count"] == 0),
        "postal_match": postal_both & postal_equal,
        "postal_conflict": postal_both & ~postal_equal,
    }
    feats.update({c: np.asarray(v).astype(V2_NEW_DTYPES[c]) for c, v in new.items()})
    feats.update(context_features(qa, feats["name_address_similarity_product"]))
    return {c: feats[c] for c in V2_FEATURE_COLUMNS}
