"""Blocking / candidate generation as inverted-index retrieval.

A strategy is a *key function* plus an inverted index over the S2/S3 pool:

    key_fn(table) -> (rows, keys)   # 0..n blocking keys per record
    strategy.fit(pool)              # index pool keys -> pool rows
    strategy.query(s1) -> (q, c)    # parallel int arrays: S1 row, pool row

Keys are 64-bit hashes (strings are hashed once per distinct value), so
composite keys like (country, token) or (country, house number, street word)
cost the same as a plain token. Rows are positional indices into the
``pool`` / ``s1`` pyarrow Tables. No strategy ever scans the pool per query:
the index is CSR arrays (postings sorted by key, plus offsets) and a query is
a vectorized binary search followed by a posting-list expansion. Strategies
are independent, so adding one is just appending it to the list handed to
``generate_candidates``.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from normalize import LEGAL_FORM_TOKENS, WEB_TOKENS, normalize_house_number

_MIX = np.uint64(0x9E3779B97F4A7C15)


def _hash_strings(values) -> np.ndarray:
    return pd.util.hash_array(np.asarray(values, dtype=object), categorize=False)


def _mix(a, b) -> np.ndarray:
    """Order-dependent combination of two uint64 hash arrays."""
    with np.errstate(over="ignore"):
        return (np.asarray(a, dtype=np.uint64) * _MIX) ^ np.asarray(b, dtype=np.uint64)


def _column(table: pa.Table, field: str) -> pa.Array:
    col = table[field]
    return col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col


def _row_hashes(table: pa.Table, field: str):
    """Per-row hash of a string column, plus a mask of non-empty rows."""
    enc = pc.dictionary_encode(_column(table, field))
    vocab = enc.dictionary.to_numpy(zero_copy_only=False)
    idx = enc.indices.to_numpy()
    return _hash_strings(vocab)[idx], (vocab != "")[idx]


def _dedupe(rows, keys):
    df = pd.DataFrame({"r": rows, "k": keys}).drop_duplicates()
    return df["r"].to_numpy(), df["k"].to_numpy()


def tokenize(table: pa.Table, field: str, transform=None):
    """Whitespace tokens of an already-normalized field.

    Returns ``(rows, codes, vocab)``: parallel arrays of row index and vocab
    code, with ``vocab`` the (optionally transformed) distinct tokens. Empty
    tokens are dropped. Transforms run once per distinct token, not per row.
    """
    lists = pc.split_pattern(_column(table, field), " ")
    rows = pc.list_parent_indices(lists).to_numpy().astype(np.int64)
    enc = pc.dictionary_encode(pc.list_flatten(lists))
    codes = enc.indices.to_numpy().astype(np.int64)
    vocab = enc.dictionary.to_numpy(zero_copy_only=False)
    if transform is not None:
        vocab = np.array([transform(t) for t in vocab], dtype=object)
    keep = (vocab != "")[codes]
    return rows[keep], codes[keep], vocab


class KeyFn:
    """Base for blocking-key functions; optionally scoped by country."""

    def __init__(self, by_country: bool = True):
        self.by_country = by_country

    def _scope(self, table, rows, keys):
        if self.by_country:
            country, _ = _row_hashes(table, "country")
            keys = _mix(country[rows], keys)
        return rows, keys


class ExactKey(KeyFn):
    """One key per record: the whole normalized field."""

    def __init__(self, field, by_country=True):
        super().__init__(by_country)
        self.field = field

    def __call__(self, table):
        h, nonempty = _row_hashes(table, self.field)
        rows = np.flatnonzero(nonempty)
        return self._scope(table, rows, h[rows])


class TokenKeys(KeyFn):
    """One key per distinct token of the field."""

    def __init__(self, field, transform=None, by_country=True):
        super().__init__(by_country)
        self.field, self.transform = field, transform

    def __call__(self, table):
        rows, codes, vocab = tokenize(table, self.field, self.transform)
        rows, keys = _dedupe(rows, _hash_strings(vocab)[codes])
        return self._scope(table, rows, keys)


class CoreNameKey(KeyFn):
    """Order-independent key over the name's distinct non-legal-form tokens.

    "Secure Brothers Limited Private" and "Secure Brothers Pvt Ltd" both
    reduce to {brothers, secure}; "Best Cafe Inc." and "Best Cafe" to
    {best, cafe}. The key is the (wrapping) sum of token hashes, so token
    order and duplicates don't matter. Records whose name is nothing but
    legal-form tokens get no key.
    """

    def __init__(self, field, transform=None, by_country=True):
        super().__init__(by_country)
        self.field, self.transform = field, transform
        stop = LEGAL_FORM_TOKENS
        self.stop = {transform(t) for t in stop} if transform else set(stop)

    def __call__(self, table):
        rows, codes, vocab = tokenize(table, self.field, self.transform)
        keep = ~np.isin(vocab, list(self.stop))[codes]
        rows, th = _dedupe(rows[keep], _hash_strings(vocab)[codes[keep]])
        order = np.argsort(rows, kind="stable")
        rows, th = rows[order], th[order]
        if not len(rows):
            return rows, th
        starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
        with np.errstate(over="ignore"):
            keys = np.add.reduceat(th, starts)
        return self._scope(table, rows[starts], keys)


class NamePairKeys(KeyFn):
    """One key per unordered pair of distinct non-legal-form name tokens.

    Targets names built only from common words: "green", "frontier" and
    "healthcare" are each capped out as single tokens, but green|frontier
    is rare. Pairs are unordered, so word-order transpositions still match,
    and one extra or missing word only removes some of the pairs.
    """

    def __init__(self, field, transform=None, by_country=True):
        super().__init__(by_country)
        self.field, self.transform = field, transform
        stop = LEGAL_FORM_TOKENS
        self.stop = {transform(t) for t in stop} if transform else set(stop)

    def __call__(self, table):
        rows, codes, vocab = tokenize(table, self.field, self.transform)
        keep = ~np.isin(vocab, list(self.stop))[codes]
        rows, th = _dedupe(rows[keep], _hash_strings(vocab)[codes[keep]])
        tok = pd.DataFrame({"r": rows, "h": th})
        pairs = tok.merge(tok, on="r", suffixes=("_a", "_b"))
        # hash order picks each unordered pair exactly once
        pairs = pairs[pairs["h_a"] < pairs["h_b"]]
        keys = _mix(pairs["h_a"].to_numpy(), pairs["h_b"].to_numpy())
        return self._scope(table, pairs["r"].to_numpy(), keys)


class AddressNumberKeys(KeyFn):
    """(house/unit number, address word) pairs from the normalized address.

    Every numeric token (up to ``max_numbers`` per record) is paired with
    every alphabetic token, e.g. "205 dougan ave blytheville ar" ->
    205|dougan, 205|ave, 205|blytheville, 205|ar. Component reordering and
    missing components don't break the specific pairs (205|dougan) that
    identify the premises, and the per-key df cap drops generic ones
    (205|ave). Independent of the business name, so it can reach records
    whose name is unrelated or garbled.

    With ``normalize_numbers``, numeric tokens are canonicalized
    ("05204" -> "5204", "2609c" -> "2609", "23rd" -> "23"), and adjacent
    numeric tokens are also joined, since punctuation cleanup splits
    "14-08" into "14 08" while the other source may write "1408".
    """

    def __init__(self, field, max_numbers=3, normalize_numbers=False, by_country=True):
        super().__init__(by_country)
        self.field, self.max_numbers = field, max_numbers
        self.normalize_numbers = normalize_numbers

    def __call__(self, table):
        rows, codes, vocab = tokenize(table, self.field)
        is_num = np.array([any(ch.isdigit() for ch in t) for t in vocab], dtype=bool)
        is_alpha = np.array([t.isalpha() for t in vocab], dtype=bool)
        num_str = (
            np.array([normalize_house_number(t) for t in vocab], dtype=object)
            if self.normalize_numbers else vocab
        )
        h_num, h_alpha = _hash_strings(num_str), _hash_strings(vocab)

        sel = is_num[codes]
        num = pd.DataFrame({"r": rows[sel], "h": h_num[codes[sel]]}).drop_duplicates()
        num = num[num.groupby("r").cumcount() < self.max_numbers]
        if self.normalize_numbers:
            # tokens are in original order, so neighbours are adjacent entries
            adj = np.flatnonzero(
                (rows[1:] == rows[:-1]) & is_num[codes[1:]] & is_num[codes[:-1]]
            )
            joined = [
                normalize_house_number(_leading_digits(vocab[a]) + _leading_digits(vocab[b]))
                for a, b in zip(codes[adj], codes[adj + 1])
            ]
            if joined:
                num = pd.concat(
                    [num, pd.DataFrame({"r": rows[adj], "h": _hash_strings(joined)})]
                ).drop_duplicates()

        sel = is_alpha[codes]
        alpha = pd.DataFrame({"r": rows[sel], "h": h_alpha[codes[sel]]}).drop_duplicates()
        pairs = num.merge(alpha, on="r", suffixes=("_n", "_a"))
        keys = _mix(pairs["h_n"].to_numpy(), pairs["h_a"].to_numpy())
        return self._scope(table, pairs["r"].to_numpy(), keys)


class SpacelessNameKey(KeyFn):
    """The name with legal-form / web tokens dropped and spaces removed.

    Matches names written as one word or as a website against their spaced
    form: "May, Evans & Zito, LLC" and "Mayevanszito.Com" both become
    "mayevanszito"; "Children's Project LLC" and "#CHILDRENSPROJECT" both
    "childrensproject". Token order is kept. Keys shorter than ``min_len``
    characters are skipped as too generic.
    """

    def __init__(self, field, min_len=4, by_country=True):
        super().__init__(by_country)
        self.field, self.min_len = field, min_len
        self.stop = pa.array(sorted(LEGAL_FORM_TOKENS | WEB_TOKENS))

    def __call__(self, table):
        lists = pc.split_pattern(_column(table, self.field), " ")
        flat = pc.list_flatten(lists)
        parents = pc.list_parent_indices(lists).to_numpy()
        keep = pc.and_(pc.invert(pc.is_in(flat, value_set=self.stop)), pc.not_equal(flat, ""))
        keep_np = keep.to_numpy(zero_copy_only=False)
        counts = np.bincount(parents[keep_np], minlength=table.num_rows)
        offsets = pa.array(np.r_[0, np.cumsum(counts)].astype(np.int32))
        joined = pc.binary_join(pa.ListArray.from_arrays(offsets, flat.filter(keep)), "")
        ok = pc.greater_equal(pc.utf8_length(joined), self.min_len)
        rows = np.flatnonzero(ok.to_numpy(zero_copy_only=False))
        enc = pc.dictionary_encode(joined.take(pa.array(rows)))
        keys = _hash_strings(enc.dictionary.to_numpy(zero_copy_only=False))[
            enc.indices.to_numpy()
        ]
        return self._scope(table, rows, keys)


def _leading_digits(token: str) -> str:
    """First digit run of a token, leading zeros kept (for joining parts)."""
    start = next((i for i, ch in enumerate(token) if ch.isdigit()), len(token))
    end = start
    while end < len(token) and token[end].isdigit():
        end += 1
    return token[start:end]


class InvertedIndex:
    """uint64 key -> pool rows, stored as CSR (postings sorted by key).

    Keys whose posting list is longer than ``max_df`` are dropped at build
    time: they would be skipped at query time anyway, and they hold most of
    the postings (frequent tokens), so dropping them bounds memory.
    """

    def __init__(self, keys: np.ndarray, rows: np.ndarray, max_df=None):
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        rows = rows[order].astype(np.int32)
        del order
        starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]]) if len(keys) else np.zeros(0, np.int64)
        df = np.diff(np.r_[starts, len(keys)])
        vocab = keys[starts]
        del keys
        self.n_keys_total, self.n_postings_total = len(vocab), len(rows)
        if max_df is not None:
            keep = df <= max_df
            rows = rows[np.repeat(keep, df)]
            vocab, df = vocab[keep], df[keep]
        self.postings = rows
        self.vocab = vocab
        self.df = df
        self.offsets = np.r_[0, np.cumsum(df)]

    def lookup(self, keys: np.ndarray) -> np.ndarray:
        """Vocab code per key, -1 when the key is not in the index."""
        pos = np.searchsorted(self.vocab, keys)
        pos[pos == len(self.vocab)] = 0
        return np.where(self.vocab[pos] == keys, pos, -1) if len(self.vocab) else np.full(len(keys), -1)

    def expand(self, query_rows: np.ndarray, codes: np.ndarray):
        """Expand (query_row, code) hits into (query_row, pool_row) pairs."""
        starts = self.offsets[codes]
        lens = self.offsets[codes + 1] - starts
        total = int(lens.sum())
        q = np.repeat(query_rows, lens)
        # position of each output element inside the flat postings array
        run_starts = np.repeat(np.cumsum(lens) - lens, lens)
        pos = np.repeat(starts, lens) + (np.arange(total) - run_starts)
        return q, self.postings[pos]


class KeyBlock:
    """Candidates sharing at least one blocking key with the S1 record.

    Keys whose document frequency in the pool exceeds ``max_df`` are treated
    as stop keys and not used for retrieval: at this scale a single such key
    (``ltd`` alone is in ~2M pool records) would otherwise pull millions of
    candidates per S1 record.
    """

    def __init__(self, name, key_fn, max_df=None):
        self.name, self.key_fn, self.max_df = name, key_fn, max_df

    def fit(self, pool: pa.Table, chunk_size: int = 1_000_000):
        # keys are built per pool slice to bound transient memory
        all_rows, all_keys = [], []
        for offset in range(0, pool.num_rows, chunk_size):
            rows, keys = self.key_fn(pool.slice(offset, chunk_size))
            all_rows.append(rows.astype(np.int32) + offset)
            all_keys.append(keys)
        self.index = InvertedIndex(
            np.concatenate(all_keys), np.concatenate(all_rows), self.max_df
        )
        return self

    def query(self, s1: pa.Table):
        rows, keys = self.key_fn(s1)
        codes = self.index.lookup(keys)
        ok = codes >= 0
        # the index was pruned at the max_df it was built with; re-checking
        # here lets a lower cap be swept without rebuilding the index
        if self.max_df is not None:
            ok[ok] = self.index.df[codes[ok]] <= self.max_df
        return self.index.expand(rows[ok], codes[ok])


def generate_candidates(strategies, s1: pa.Table, n_pool: int, chunk_size: int = 20_000):
    """Union + dedupe of all strategies' candidates, chunked over S1.

    Yields, per chunk, ``(offset, s1_rows, pool_rows, bits)`` with unique
    (s1_row, pool_row) pairs; bit ``k`` of ``bits`` is set when strategy
    ``k`` retrieved that pair. ``s1_rows`` are global row indices into s1.
    """
    if len(strategies) > 16:
        raise ValueError("bitmask is uint16: at most 16 strategies")
    for offset in range(0, s1.num_rows, chunk_size):
        chunk = s1.slice(offset, chunk_size)
        keys, bits = [], []
        for k, strat in enumerate(strategies):
            q, c = strat.query(chunk)
            keys.append((q.astype(np.int64) + offset) * n_pool + c)
            bits.append(np.full(len(q), 1 << k, dtype=np.uint16))
        keys = np.concatenate(keys)
        bits = np.concatenate(bits)
        order = np.argsort(keys, kind="stable")
        keys, bits = keys[order], bits[order]
        if len(keys):
            starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
            bits = np.bitwise_or.reduceat(bits, starts)
            keys = keys[starts]
        yield offset, keys // n_pool, keys % n_pool, bits
