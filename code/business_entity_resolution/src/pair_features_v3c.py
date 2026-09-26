"""V3c add-on features: address distinctiveness.

Many false positives share a registered office address that hosts several
different businesses, and many missed matches share an address but have an
unrelated brand name. Whether an address match is evidence depends on how
distinctive the address is. Counted on the train S2/S3 pool:

- core address key: order-independent hash of an address's distinct
  meaningful tokens (numbers in normalize_house_number form, stop tokens and
  single letters dropped), so formatting and reordering don't matter
- core name key: the same over phonetic-skeleton name tokens, so variants
  of one business name count once

Features (-1 when the relevant address is empty):
- cand_address_key_idf: log(N / df) / log(N) of the candidate's core
  address in the pool (1 = unique, low = shared by many records)
- cand_address_distinct_names: log(1 + number of distinct core names at the
  candidate's core address in the pool)
- s1_address_key_idf / s1_address_distinct_names: the same for the S1
  address looked up in the pool (unseen address: idf 1, names 0)
- address_key_equal: S1 and candidate core address keys are identical
"""

import numpy as np
import pandas as pd
import pyarrow as pa

from blocking import _hash_strings, tokenize
from normalize import normalize_house_number, phonetic_skeleton
from pair_features import ADDRESS_FIELD, NAME_FIELD, _is_meaningful
from normalize import ADDRESS_STOP_TOKENS
from pair_features_v2 import PHONETIC_STOP_TOKENS

V3C_DTYPES = {
    "cand_address_key_idf": np.float32,
    "cand_address_distinct_names": np.float32,
    "s1_address_key_idf": np.float32,
    "s1_address_distinct_names": np.float32,
    "address_key_equal": np.int8,
}
V3C_COLUMNS = list(V3C_DTYPES)
V3C_INPUTS = []


def _address_token(t: str) -> str:
    return normalize_house_number(t) if any(ch.isdigit() for ch in t) else t


def set_keys(table: pa.Table, field: str, transform, stop) -> tuple:
    """Order-independent key of each record's distinct meaningful tokens.

    Returns (keys uint64, nonempty bool) per row; rows with no meaningful
    token get key 0 and nonempty False.
    """
    rows, codes, vocab = tokenize(table, field, transform)
    keep_vocab = np.array([_is_meaningful(t, stop) for t in vocab], dtype=bool)
    keep = keep_vocab[codes]
    h = _hash_strings(vocab)
    df = pd.DataFrame({"r": rows[keep], "h": h[codes[keep]]}).drop_duplicates()
    keys = np.zeros(table.num_rows, dtype=np.uint64)
    if len(df):
        r = df["r"].to_numpy()
        order = np.argsort(r, kind="stable")
        r, hv = r[order], df["h"].to_numpy()[order]
        starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
        with np.errstate(over="ignore"):
            keys[r[starts]] = np.add.reduceat(hv, starts)
    nonempty = np.zeros(table.num_rows, dtype=bool)
    nonempty[df["r"].to_numpy()] = True
    return keys, nonempty


class FieldIndexV3c:
    def __init__(self, table: pa.Table, side: str, pool_index=None):
        self.addr_key, self.addr_nonempty = set_keys(
            table, ADDRESS_FIELD, _address_token, ADDRESS_STOP_TOKENS)
        if side == "pool":
            name_key, _ = set_keys(table, NAME_FIELD, phonetic_skeleton, PHONETIC_STOP_TOKENS)
            self.n = table.num_rows
            addr = self.addr_key[self.addr_nonempty]
            self.vocab, self.df = np.unique(addr, return_counts=True)
            pairs = np.unique(np.stack([addr, name_key[self.addr_nonempty]]), axis=1)
            names_vocab, names = np.unique(pairs[0], return_counts=True)
            assert np.array_equal(names_vocab, self.vocab)
            self.distinct_names = names
            self.row_idf, self.row_names = self.lookup(self.addr_key, self.addr_nonempty)
        else:
            self.row_idf, self.row_names = pool_index.lookup(self.addr_key, self.addr_nonempty)

    def lookup(self, keys, nonempty):
        """(idf, log1p distinct names) of address keys in this (pool) index."""
        pos = np.searchsorted(self.vocab, keys)
        pos[pos == len(self.vocab)] = 0
        found = nonempty & (self.vocab[pos] == keys)
        df = np.where(found, self.df[pos], 1)
        names = np.where(found, self.distinct_names[pos], 0)
        idf = np.log(self.n / df) / np.log(self.n)
        return (np.where(nonempty, idf, -1.0),
                np.where(nonempty, np.log1p(names), -1.0))


def compute_features_v3c(s1: FieldIndexV3c, pool: FieldIndexV3c, qa, qb, v2=None):
    feats = {
        "cand_address_key_idf": pool.row_idf[qb],
        "cand_address_distinct_names": pool.row_names[qb],
        "s1_address_key_idf": s1.row_idf[qa],
        "s1_address_distinct_names": s1.row_names[qa],
        "address_key_equal": (s1.addr_nonempty[qa] & pool.addr_nonempty[qb]
                              & (s1.addr_key[qa] == pool.addr_key[qb])),
    }
    return {c: np.asarray(v).astype(V3C_DTYPES[c]) for c, v in feats.items()}
