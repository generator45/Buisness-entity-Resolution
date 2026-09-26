"""V3b add-on features: house-number closeness and name distinctiveness.

Aimed at V3a's missed matches:

- house-number closeness. The data perturbs house numbers (5915 vs 5917,
  40 vs 42, 4122 vs 412), so V2/V3a's shared-number features read many true
  matches as "different number". Each address keeps its first
  ``MAX_NUMBERS`` distinct normalized numbers (normalize_house_number);
  every cross pair of numbers is compared and per S1-candidate pair we keep
  - address_number_min_edit: smallest digit edit distance (0 = shared)
  - address_number_min_rel_diff: smallest |a - b| / max(a, b)
  - address_number_prefix_or_suffix: some pair where the shorter number
    (>= 2 digits) is a proper prefix or suffix of the longer one
  Pairs where either address has no number get -1 for the first two.
- name distinctiveness. A strong name match with no usable address is only
  convincing when the name is rare:
  - name_s1_idf_sum / name_cand_idf_sum: total idf of each side's meaningful
    name tokens (idf from the train pool)
  - name_shared_idf_sum: total idf of the shared meaningful tokens
  - name_spaceless_exact: spaceless core names equal and non-empty
"""

import numpy as np
import pandas as pd
import pyarrow as pa
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist

from blocking import tokenize
from normalize import normalize_house_number
from pair_features import ADDRESS_FIELD, NAME_FIELD, NAME_STOP_TOKENS, StringStats, TokenSets
from pair_features_v2 import _idf_overlap, spaceless_core

MAX_NUMBERS = 5
MAX_EXACT_DIGITS = 18  # int64-exact; longer numbers skip numeric comparisons
_POW10 = 10 ** np.arange(MAX_EXACT_DIGITS + 1, dtype=np.int64)

V3B_DTYPES = {
    "address_number_min_edit": np.int16,
    "address_number_min_rel_diff": np.float32,
    "address_number_prefix_or_suffix": np.int8,
    "name_s1_idf_sum": np.float32,
    "name_cand_idf_sum": np.float32,
    "name_shared_idf_sum": np.float32,
    "name_spaceless_exact": np.int8,
}
V3B_COLUMNS = list(V3B_DTYPES)
V3B_BINARY = [c for c, t in V3B_DTYPES.items() if t == np.int8]
V3B_INPUTS = []  # needs no V2 columns


class NumberLists:
    """First MAX_NUMBERS distinct normalized numbers of each address, CSR."""

    def __init__(self, table: pa.Table):
        rows, codes, vocab = tokenize(table, ADDRESS_FIELD, transform=normalize_house_number)
        strings, inverse = np.unique(vocab.astype(str), return_inverse=True)
        df = pd.DataFrame({"r": rows, "c": inverse[codes]}).drop_duplicates()
        df = df[df.groupby("r").cumcount() < MAX_NUMBERS]
        self.count = np.bincount(df["r"].to_numpy(), minlength=table.num_rows)
        self.offsets = np.r_[0, np.cumsum(self.count)]
        self.codes = df["c"].to_numpy()
        self.strings = strings.astype(object)
        self.length = np.array([len(s) for s in strings], dtype=np.int64)
        self.value = np.array(
            # ordinals ("23o") and very long numbers skip numeric comparisons
            [int(s) if s.isdigit() and len(s) <= MAX_EXACT_DIGITS else -1 for s in strings],
            dtype=np.int64,
        )


class FieldIndexV3b:
    def __init__(self, table: pa.Table, side: str, pool_index=None):
        s1 = side == "s1"
        idf = {"idf_from": pool_index.name_tokens} if s1 else {"compute_idf": True}
        self.name_tokens = TokenSets(table, NAME_FIELD, stop=NAME_STOP_TOKENS,
                                     with_tokens=s1, with_lookup=not s1, **idf)
        self.numbers = NumberLists(table)
        self.spaceless = StringStats(pa.table({"s": spaceless_core(table)}), "s")


def number_closeness(a: NumberLists, b: NumberLists, qa, qb):
    """(min edit distance, min relative difference, any prefix/suffix) per pair."""
    n = len(qa)
    la, lb = a.count[qa], b.count[qb]
    k = la * lb
    total = int(k.sum())
    min_edit = np.full(n, -1, np.int64)
    min_rel = np.full(n, -1.0)
    presuf = np.zeros(n, bool)
    if total == 0:
        return min_edit, min_rel, presuf
    within = np.arange(total) - np.repeat(np.cumsum(k) - k, k)
    lb_rep = np.repeat(lb, k)
    ca = a.codes[np.repeat(a.offsets[qa], k) + within // lb_rep]
    cb = b.codes[np.repeat(b.offsets[qb], k) + within % lb_rep]

    edit = np.asarray(cpdist(a.strings[ca].tolist(), b.strings[cb].tolist(),
                             scorer=Levenshtein.distance, workers=-1), dtype=np.int64)
    va, vb = a.value[ca], b.value[cb]
    exact = (va >= 0) & (vb >= 0)
    rel = np.where(exact, np.abs(va - vb) / np.maximum(np.maximum(va, vb), 1), 1.0)
    rel = np.where(edit == 0, 0.0, rel)

    lA, lB = a.length[ca], b.length[cb]
    longer, shorter = np.where(lA >= lB, va, vb), np.where(lA >= lB, vb, va)
    dl, short_len = np.abs(lA - lB), np.minimum(lA, lB)
    ok = exact & (dl > 0) & (short_len >= 2)
    dl_c, sl_c = np.where(ok, dl, 0), np.where(ok, short_len, 0)
    combo_presuf = ok & (
        (longer // _POW10[dl_c] == shorter) | (longer % _POW10[sl_c] == shorter)
    )

    has = k > 0
    starts = (np.cumsum(k) - k)[has]
    min_edit[has] = np.minimum.reduceat(edit, starts)
    min_rel[has] = np.minimum.reduceat(rel, starts)
    presuf[has] = np.maximum.reduceat(combo_presuf.astype(np.int8), starts).astype(bool)
    return min_edit, min_rel, presuf


def compute_features_v3b(s1: FieldIndexV3b, pool: FieldIndexV3b, qa, qb, v2=None):
    min_edit, min_rel, presuf = number_closeness(s1.numbers, pool.numbers, qa, qb)
    shared_idf, _ = _idf_overlap(s1.name_tokens, pool.name_tokens, qa, qb)
    feats = {
        "address_number_min_edit": min_edit,
        "address_number_min_rel_diff": min_rel,
        "address_number_prefix_or_suffix": presuf,
        "name_s1_idf_sum": s1.name_tokens.idf_sum[qa],
        "name_cand_idf_sum": pool.name_tokens.idf_sum[qb],
        "name_shared_idf_sum": shared_idf,
        "name_spaceless_exact": (s1.spaceless.nonempty[qa] & pool.spaceless.nonempty[qb]
                                 & (s1.spaceless.hash[qa] == pool.spaceless.hash[qb])),
    }
    return {c: np.asarray(v).astype(V3B_DTYPES[c]) for c, v in feats.items()}
