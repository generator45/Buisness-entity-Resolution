"""Low-compute pairwise features for (S1 record, candidate record) pairs.

Each text field is turned into per-record sets of distinct hashed tokens,
stored as CSR arrays (the same layout as the blocking index). The candidate
(pool) side additionally gets a sorted lookup of (record, token) keys, so the
token overlap of a whole chunk of pairs is a vectorized membership test:
expand every S1 token of every pair, check it against the candidate's set,
and count hits per pair. No per-pair Python loops.

Definitions (applied identically to names and addresses):

- text: the transliterated normalized field (Unicode-folded, accents
  removed, legal-suffix / address-abbreviation variants canonicalized), so
  scripts and accents compare consistently.
- token: whitespace token of that text; sets are of *distinct* tokens.
- numeric token: token containing a digit, compared in canonical form
  (``normalize_house_number``: "05204" == "5204", "2609c" == "2609").
- meaningful token: not in the field's stop list, and longer than one
  character unless it contains a digit.
- exact match: normalized strings equal *and* non-empty (two empty names
  are not evidence of a match).
- ratios / Jaccard: 0.0 when the denominator is 0.
"""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from blocking import _hash_strings, _mix, tokenize
from config import STRONG_ADDRESS_JACCARD, STRONG_NAME_JACCARD
from normalize import (
    ADDRESS_STOP_TOKENS,
    LEGAL_FORM_TOKENS,
    WEB_TOKENS,
    nfkc_casefold,
    normalize_house_number,
)

NAME_FIELD = "business_name_translit"
ADDRESS_FIELD = "business_address_translit"
NAME_STOP_TOKENS = LEGAL_FORM_TOKENS | WEB_TOKENS

FEATURE_DTYPES = {
    "name_exact_match": np.int8,
    "name_token_overlap_count": np.int16,
    "name_token_jaccard": np.float32,
    "name_length_ratio": np.float32,
    "name_shared_numeric_token_count": np.int16,
    "name_shared_meaningful_token_count": np.int16,
    "address_exact_match": np.int8,
    "address_token_overlap_count": np.int16,
    "address_token_jaccard": np.float32,
    "address_length_ratio": np.float32,
    "address_shared_numeric_token_count": np.int16,
    "address_shared_meaningful_token_count": np.int16,
    "country_exact_match": np.int8,
    "name_address_similarity_product": np.float32,
    "name_and_address_exact_match": np.int8,
    "strong_name_and_address_overlap": np.int8,
}
FEATURE_COLUMNS = list(FEATURE_DTYPES)
BINARY_FEATURES = [c for c, t in FEATURE_DTYPES.items() if t == np.int8]
UNIT_INTERVAL_FEATURES = [c for c, t in FEATURE_DTYPES.items() if t == np.float32]


def _is_meaningful(token: str, stop) -> bool:
    has_digit = any(ch.isdigit() for ch in token)
    return token not in stop and (len(token) > 1 or has_digit)


class TokenSets:
    """Distinct hashed tokens per record for one field, in CSR form.

    ``with_tokens`` keeps the CSR token arrays (needed on the side that is
    expanded, i.e. S1); ``with_lookup`` builds the sorted (record, token)
    key array for membership tests (needed on the candidate side).
    """

    def __init__(self, table: pa.Table, field: str, transform=None, stop=frozenset(),
                 with_tokens=True, with_lookup=False):
        rows, codes, vocab = tokenize(table, field, transform)
        tok_hash = _hash_strings(vocab)
        meaningful = np.array([_is_meaningful(t, stop) for t in vocab], dtype=bool)
        # distinct per record; a transform can map two raw tokens to one value
        key = _mix(rows.astype(np.uint64), tok_hash[codes])
        uniq, first = np.unique(key, return_index=True)
        if with_lookup:
            self.lookup = uniq  # np.unique output is already sorted
        del uniq, key
        first.sort()  # keep rows ascending for the CSR layout
        rows, codes = rows[first], codes[first]
        self.count = np.bincount(rows, minlength=table.num_rows).astype(np.int32)
        if with_tokens:
            self.offsets = np.r_[0, np.cumsum(self.count)]
            self.tokens = tok_hash[codes]
            self.meaningful = meaningful[codes]

    def expand(self, rows: np.ndarray):
        """(pair index, token, meaningful flag) for every token of ``rows``."""
        starts = self.offsets[rows]
        lens = self.count[rows]
        total = int(lens.sum())
        pair = np.repeat(np.arange(len(rows)), lens)
        run_starts = np.repeat(np.cumsum(lens) - lens, lens)
        pos = np.repeat(starts, lens) + (np.arange(total) - run_starts)
        return pair, self.tokens[pos], self.meaningful[pos]

    def contains(self, rows: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        keys = _mix(rows.astype(np.uint64), tokens)
        pos = np.searchsorted(self.lookup, keys)
        pos[pos == len(self.lookup)] = 0
        return self.lookup[pos] == keys if len(self.lookup) else np.zeros(len(keys), bool)


class StringStats:
    """Per-record hash, non-emptiness and character length of a string field."""

    def __init__(self, table: pa.Table, field: str, normalize=None):
        col = table[field]
        col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
        enc = pc.dictionary_encode(col)
        vocab = enc.dictionary.to_numpy(zero_copy_only=False)
        if normalize is not None:
            vocab = np.array([normalize(v) for v in vocab], dtype=object)
        idx = enc.indices.to_numpy()
        self.hash = _hash_strings(vocab)[idx]
        self.nonempty = (vocab != "")[idx]
        self.length = pc.utf8_length(col).to_numpy(zero_copy_only=False).astype(np.int32)


class FieldIndex:
    """Everything the features need about one side (S1 sample or pool)."""

    def __init__(self, table: pa.Table, side: str):
        s1 = side == "s1"
        kw = dict(with_tokens=s1, with_lookup=not s1)
        self.name_tokens = TokenSets(table, NAME_FIELD, stop=NAME_STOP_TOKENS, **kw)
        self.name_numbers = TokenSets(table, NAME_FIELD, transform=normalize_house_number, **kw)
        self.addr_tokens = TokenSets(table, ADDRESS_FIELD, stop=ADDRESS_STOP_TOKENS, **kw)
        self.addr_numbers = TokenSets(table, ADDRESS_FIELD, transform=normalize_house_number, **kw)
        self.name = StringStats(table, NAME_FIELD)
        self.addr = StringStats(table, ADDRESS_FIELD)
        self.country = StringStats(table, "country", normalize=lambda v: nfkc_casefold(v).strip())


def _overlap(a: TokenSets, b: TokenSets, qa: np.ndarray, qb: np.ndarray):
    """Shared distinct tokens per pair, overall and meaningful-only."""
    pair, tokens, meaningful = a.expand(qa)
    hit = b.contains(qb[pair], tokens)
    n = len(qa)
    shared = np.bincount(pair[hit], minlength=n)
    shared_meaningful = np.bincount(pair[hit & meaningful], minlength=n)
    return shared, shared_meaningful


def _safe_div(num, den):
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, 0.0)


def _exact(a: StringStats, b: StringStats, qa, qb):
    return a.nonempty[qa] & b.nonempty[qb] & (a.hash[qa] == b.hash[qb])


def _length_ratio(a: StringStats, b: StringStats, qa, qb):
    la, lb = a.length[qa], b.length[qb]
    return _safe_div(np.minimum(la, lb), np.maximum(la, lb))


def compute_features(s1: FieldIndex, pool: FieldIndex, qa: np.ndarray, qb: np.ndarray,
                     strong_name=STRONG_NAME_JACCARD, strong_address=STRONG_ADDRESS_JACCARD):
    """Feature columns for pairs (S1 local row qa[i], pool row qb[i])."""
    name_shared, name_meaningful = _overlap(s1.name_tokens, pool.name_tokens, qa, qb)
    name_numeric, _ = _overlap(s1.name_numbers, pool.name_numbers, qa, qb)
    addr_shared, addr_meaningful = _overlap(s1.addr_tokens, pool.addr_tokens, qa, qb)
    addr_numeric, _ = _overlap(s1.addr_numbers, pool.addr_numbers, qa, qb)

    name_union = s1.name_tokens.count[qa] + pool.name_tokens.count[qb] - name_shared
    addr_union = s1.addr_tokens.count[qa] + pool.addr_tokens.count[qb] - addr_shared
    name_jaccard = _safe_div(name_shared, name_union)
    addr_jaccard = _safe_div(addr_shared, addr_union)
    name_exact = _exact(s1.name, pool.name, qa, qb)
    addr_exact = _exact(s1.addr, pool.addr, qa, qb)

    feats = {
        "name_exact_match": name_exact,
        "name_token_overlap_count": name_shared,
        "name_token_jaccard": name_jaccard,
        "name_length_ratio": _length_ratio(s1.name, pool.name, qa, qb),
        "name_shared_numeric_token_count": name_numeric,
        "name_shared_meaningful_token_count": name_meaningful,
        "address_exact_match": addr_exact,
        "address_token_overlap_count": addr_shared,
        "address_token_jaccard": addr_jaccard,
        "address_length_ratio": _length_ratio(s1.addr, pool.addr, qa, qb),
        "address_shared_numeric_token_count": addr_numeric,
        "address_shared_meaningful_token_count": addr_meaningful,
        "country_exact_match": _exact(s1.country, pool.country, qa, qb),
        "name_address_similarity_product": name_jaccard * addr_jaccard,
        "name_and_address_exact_match": name_exact & addr_exact,
        "strong_name_and_address_overlap": (name_jaccard >= strong_name)
        & (addr_jaccard >= strong_address),
    }
    return {c: np.asarray(v).astype(FEATURE_DTYPES[c]) for c, v in feats.items()}
