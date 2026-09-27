"""V3d add-on features: evidence for pairs the address can't decide.

Half of the missed matches on matcher_val (and a third of the false
positives) have an empty address on one side, so the model decides them on
the name alone. Two kinds of evidence help there:

Name distinctiveness (counted on the S2/S3 pool, country-scoped; the name
key is the order-independent hash of the phonetic-skeleton name tokens with
legal-form tokens dropped, as the V3c core name key):
- cand_name_pool_df: log(1 + pool records sharing the candidate's name key)
- s1_name_pool_df: the same for the S1 name looked up in the pool (0 unseen)
- name_key_equal: S1 and candidate name keys are identical
(-1 for the df features when the name has no meaningful token)

Agreement with the S1's other candidates. A "strong" candidate has both
name_token_set_ratio and address_token_set_ratio >= STRONG (V2 inputs); the
"reference" of a pair is the S1's highest-similarity candidate other than
itself (by name_address_similarity_product):
- group_strong_count: number of strong candidates of the S1
- same_name_as_strong: another strong candidate of the S1 has this
  candidate's name key
- name_sim_to_reference: Indel similarity of the spaceless core names of the
  candidate and its reference (-1 if either name is empty or the S1 has a
  single candidate)
- address_sim_to_reference: token-set ratio of their addresses (-1 if either
  address is empty or the S1 has a single candidate)

The group features need chunks that hold complete S1 groups, sorted by S1.
"""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from rapidfuzz import fuzz
from rapidfuzz.distance import Indel
from rapidfuzz.process import cpdist

from blocking import _hash_strings
from normalize import phonetic_skeleton
from pair_features import ADDRESS_FIELD, NAME_FIELD
from pair_features_v2 import PHONETIC_STOP_TOKENS, spaceless_core
from pair_features_v3c import set_keys

V3D_DTYPES = {
    "cand_name_pool_df": np.float32,
    "s1_name_pool_df": np.float32,
    "name_key_equal": np.int8,
    "group_strong_count": np.float32,
    "same_name_as_strong": np.int8,
    "name_sim_to_reference": np.float32,
    "address_sim_to_reference": np.float32,
}
V3D_COLUMNS = list(V3D_DTYPES)
V3D_INPUTS = ["name_address_similarity_product", "name_token_set_ratio",
              "address_token_set_ratio"]
STRONG = 0.8


def _name_keys(table: pa.Table):
    """Country-scoped name keys (uint64) and whether the name has any token."""
    keys, nonempty = set_keys(table, NAME_FIELD, phonetic_skeleton, PHONETIC_STOP_TOKENS)
    country = table["country"].to_numpy()
    codes, inverse = np.unique(country, return_inverse=True)
    with np.errstate(over="ignore"):
        keys = keys ^ _hash_strings(codes)[inverse]
    return keys, nonempty


class FieldIndexV3d:
    def __init__(self, table: pa.Table, side: str, pool_index=None):
        self.name_key, self.name_nonempty = _name_keys(table)
        if side == "pool":
            self.vocab, self.df = np.unique(self.name_key[self.name_nonempty],
                                            return_counts=True)
            self.row_df = self.lookup(self.name_key, self.name_nonempty)
            # candidate-to-candidate comparisons need the pool text
            self.spaceless = spaceless_core(table)
            addr = table[ADDRESS_FIELD]
            self.addr_text = addr.combine_chunks() if isinstance(addr, pa.ChunkedArray) else addr
        else:
            self.row_df = pool_index.lookup(self.name_key, self.name_nonempty)

    def lookup(self, keys, nonempty):
        pos = np.searchsorted(self.vocab, keys)
        pos[pos == len(self.vocab)] = 0
        found = nonempty & (self.vocab[pos] == keys)
        df = np.where(found, self.df[pos], 0)
        return np.where(nonempty, np.log1p(df), -1.0)


def _reference(qa, sim):
    """Row index of each pair's reference candidate (-1 if none)."""
    n = len(qa)
    order = np.lexsort((-sim, qa))
    starts = np.flatnonzero(np.r_[True, qa[order][1:] != qa[order][:-1]])
    sizes = np.diff(np.r_[starts, n])
    top = np.repeat(order[starts], sizes)  # per sorted position: its group's top
    second = np.full(len(starts), -1)
    has2 = sizes > 1
    second[has2] = order[starts[has2] + 1]
    second = np.repeat(second, sizes)
    ref_sorted = np.where(order == top, second, top)
    ref = np.empty(n, np.int64)
    ref[order] = ref_sorted
    return ref


def _pair_scores(scorer, text: pa.Array, a, b, scale=1.0):
    """Scorer over pool rows a[i] vs b[i]; -1 where b < 0 or a text is empty."""
    ok = b >= 0
    left, right = text.take(pa.array(a[ok])), text.take(pa.array(b[ok]))
    out = np.full(len(a), -1.0)
    scores = np.asarray(cpdist(left.to_pylist(), right.to_pylist(), scorer=scorer, workers=-1),
                        np.float64) / scale
    both = (pc.utf8_length(left).to_numpy() > 0) & (pc.utf8_length(right).to_numpy() > 0)
    out[ok] = np.where(both, scores, -1.0)
    return out


def compute_features_v3d(s1: FieldIndexV3d, pool: FieldIndexV3d, qa, qb, v2: dict):
    if (np.diff(qa) < 0).any():
        raise ValueError("pairs must be sorted by S1")
    starts = np.flatnonzero(np.r_[True, qa[1:] != qa[:-1]])
    sizes = np.diff(np.r_[starts, len(qa)])
    group = np.repeat(np.arange(len(starts)), sizes)
    strong = ((v2["name_token_set_ratio"] >= STRONG)
              & (v2["address_token_set_ratio"] >= STRONG))
    strong_count = np.bincount(group, weights=strong, minlength=len(starts))

    # strong candidates of the same S1 sharing this candidate's name key
    ckey = pool.name_key[qb]
    cnonempty = pool.name_nonempty[qb]
    _, inv = np.unique(np.stack([group.astype(np.uint64), ckey]), axis=1, return_inverse=True)
    inv = inv.ravel()
    same_strong = np.bincount(inv, weights=strong)[inv] - strong

    ref = _reference(qa, v2["name_address_similarity_product"])
    ref_b = np.where(ref >= 0, qb[np.maximum(ref, 0)], -1)
    feats = {
        "cand_name_pool_df": pool.row_df[qb],
        "s1_name_pool_df": s1.row_df[qa],
        "name_key_equal": s1.name_nonempty[qa] & cnonempty & (s1.name_key[qa] == ckey),
        "group_strong_count": strong_count[group],
        "same_name_as_strong": cnonempty & (same_strong > 0),
        "name_sim_to_reference": _pair_scores(Indel.normalized_similarity,
                                              pool.spaceless, qb, ref_b),
        "address_sim_to_reference": _pair_scores(fuzz.token_set_ratio,
                                                 pool.addr_text, qb, ref_b, scale=100.0),
    }
    return {c: np.asarray(v).astype(V3D_DTYPES[c]) for c, v in feats.items()}
