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

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from normalize import ADDRESS_STOP_TOKENS, LEGAL_FORM_TOKENS, WEB_TOKENS, normalize_house_number

_MIX = np.uint64(0x9E3779B97F4A7C15)
N_INDEX_PARTITIONS = 8  # 2^3: KeyBlock.fit partitions keys by their top 3 bits


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


class TokenRarity:
    """Pool document frequency of a field's meaningful tokens.

    Meaningful: not in ``stop``, at least two characters, and alphabetic when
    ``alpha_only``. Counted per 1M-row pool slice so memory stays bounded;
    tokens never seen in the pool count as the rarest (df 0).
    """

    def __init__(self, field, stop=frozenset(), alpha_only=False, transform=None):
        self.field, self.stop, self.alpha_only = field, stop, alpha_only
        self.transform = transform

    def tokens(self, table):
        """Distinct (row, token hash) pairs of meaningful tokens."""
        rows, codes, vocab = tokenize(table, self.field, self.transform)
        ok = np.array([
            len(t) > 1 and t not in self.stop and (t.isalpha() or not self.alpha_only)
            for t in vocab], dtype=bool)
        keep = ok[codes]
        return _dedupe(rows[keep], _hash_strings(vocab)[codes[keep]])

    def fit(self, pool, chunk_size=1_000_000):
        parts = []
        for offset in range(0, pool.num_rows, chunk_size):
            _, h = self.tokens(pool.slice(offset, chunk_size))
            u, c = np.unique(h, return_counts=True)
            parts.append(pd.DataFrame({"h": u, "c": c}))
        df = pd.concat(parts).groupby("h", sort=True)["c"].sum()
        self.vocab, self.df = df.index.to_numpy(dtype=np.uint64), df.to_numpy()
        return self

    def rarest(self, table, k):
        """The ``k`` rarest meaningful tokens of each row: (rows, hashes)."""
        rows, h = self.tokens(table)
        pos = np.searchsorted(self.vocab, h)
        pos[pos == len(self.vocab)] = 0
        df = np.where(self.vocab[pos] == h, self.df[pos], 0)
        order = np.lexsort((h, df, rows))
        rows, h = rows[order], h[order]
        keep = _rank_in_group(rows) < k
        return rows[keep], h[keep]


def _rank_in_group(rows):
    """0-based position of each element within its run of equal ``rows``."""
    if not len(rows):
        return rows
    starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
    sizes = np.diff(np.r_[starts, len(rows)])
    return np.arange(len(rows)) - np.repeat(starts, sizes)


class AddressRarePairKeys(KeyFn):
    """Unordered pairs among each address's ``k`` rarest alphabetic words.

    For addresses without usable house numbers ("Patil Galli, Jalalpur,
    Raibag"), where AddressNumberKeys finds nothing: patil|galli,
    galli|jalalpur, ... Rarity is pool document frequency, so generic words
    (city, state, street type) are rarely chosen.
    """

    def __init__(self, field, k=3, by_country=True):
        super().__init__(by_country)
        self.k = k
        self.rarity = TokenRarity(field, stop=ADDRESS_STOP_TOKENS, alpha_only=True)

    def prepare(self, pool):
        self.rarity.fit(pool)

    def __call__(self, table):
        rows, h = self.rarity.rarest(table, self.k)
        tok = pd.DataFrame({"r": rows, "h": h})
        pairs = tok.merge(tok, on="r", suffixes=("_a", "_b"))
        pairs = pairs[pairs["h_a"] < pairs["h_b"]]
        keys = _mix(pairs["h_a"].to_numpy(), pairs["h_b"].to_numpy())
        return self._scope(table, pairs["r"].to_numpy(), keys)


class NameAddressRareKeys(KeyFn):
    """Each of the ``k_name`` rarest name words x ``k_addr`` rarest address words.

    For records whose house number is present on one side only and whose
    names differ slightly ("Red Bistro" / "Red Bistro Center" on Nueces St,
    Crystal City): red|crystal, bistro|nueces, ...
    """

    def __init__(self, name_field, address_field, k_name=2, k_addr=2, by_country=True):
        super().__init__(by_country)
        self.k_name, self.k_addr = k_name, k_addr
        self.name_rarity = TokenRarity(name_field, stop=LEGAL_FORM_TOKENS | WEB_TOKENS)
        self.addr_rarity = TokenRarity(address_field, stop=ADDRESS_STOP_TOKENS, alpha_only=True)

    def prepare(self, pool):
        self.name_rarity.fit(pool)
        self.addr_rarity.fit(pool)

    def __call__(self, table):
        nr, nh = self.name_rarity.rarest(table, self.k_name)
        ar, ah = self.addr_rarity.rarest(table, self.k_addr)
        pairs = pd.DataFrame({"r": nr, "n": nh}).merge(pd.DataFrame({"r": ar, "a": ah}), on="r")
        keys = _mix(pairs["n"].to_numpy(), pairs["a"].to_numpy())
        return self._scope(table, pairs["r"].to_numpy(), keys)


def _leading_digits(token: str) -> str:
    """First digit run of a token, leading zeros kept (for joining parts)."""
    start = next((i for i, ch in enumerate(token) if ch.isdigit()), len(token))
    end = start
    while end < len(token) and token[end].isdigit():
        end += 1
    return token[start:end]


def _build_csr(keys, rows, max_df):
    """Sort (key, row) pairs by key and group: (vocab, df, postings, n_keys, n_postings)."""
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    rows = rows[order].astype(np.int32)
    del order
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]]) if len(keys) else np.zeros(0, np.int64)
    df = np.diff(np.r_[starts, len(keys)])
    vocab = keys[starts]
    del keys
    n_keys, n_postings = len(vocab), len(rows)
    if max_df is not None:
        keep = df <= max_df
        rows = rows[np.repeat(keep, df)]
        vocab, df = vocab[keep], df[keep]
    return vocab, df, rows, n_keys, n_postings


class InvertedIndex:
    """uint64 key -> pool rows, stored as CSR (postings sorted by key).

    Keys whose posting list is longer than ``max_df`` are dropped at build
    time: they would be skipped at query time anyway, and they hold most of
    the postings (frequent tokens), so dropping them bounds memory.
    """

    def __init__(self, keys: np.ndarray, rows: np.ndarray, max_df=None):
        vocab, df, postings, n_keys, n_postings = _build_csr(keys, rows, max_df)
        self._set(vocab, df, postings, n_keys, n_postings)

    def _set(self, vocab, df, postings, n_keys_total, n_postings_total):
        self.n_keys_total, self.n_postings_total = n_keys_total, n_postings_total
        self.postings = postings
        self.vocab = vocab
        self.df = df
        self.offsets = np.r_[0, np.cumsum(df)]

    @classmethod
    def from_partitions(cls, parts, max_df=None):
        """Build from (keys, rows) partitions split by the key's top bits.

        Partition p holds keys in [p * 2^61, (p + 1) * 2^61), so building each
        partition separately and concatenating gives exactly the index a
        single build would, with ~1/8 of the sorting memory.
        """
        built = []
        for keys, rows in parts:
            built.append(_build_csr(keys, rows, max_df))
        index = cls.__new__(cls)
        index._set(
            np.concatenate([b[0] for b in built]) if built else np.zeros(0, np.uint64),
            np.concatenate([b[1] for b in built]) if built else np.zeros(0, np.int64),
            np.concatenate([b[2] for b in built]) if built else np.zeros(0, np.int32),
            sum(b[3] for b in built), sum(b[4] for b in built),
        )
        return index

    def lookup(self, keys: np.ndarray) -> np.ndarray:
        """Vocab code per key, -1 when the key is not in the index."""
        pos = np.searchsorted(self.vocab, keys)
        pos[pos == len(self.vocab)] = 0
        return np.where(self.vocab[pos] == keys, pos, -1) if len(self.vocab) else np.full(len(keys), -1)

    def expand(self, query_rows: np.ndarray, codes: np.ndarray, with_df: bool = False):
        """Expand (query_row, code) hits into (query_row, pool_row) pairs.

        With ``with_df``, also returns the document frequency of the key that
        produced each pair.
        """
        starts = self.offsets[codes]
        lens = self.offsets[codes + 1] - starts
        total = int(lens.sum())
        q = np.repeat(query_rows, lens)
        # position of each output element inside the flat postings array
        run_starts = np.repeat(np.cumsum(lens) - lens, lens)
        pos = np.repeat(starts, lens) + (np.arange(total) - run_starts)
        if with_df:
            return q, self.postings[pos], np.repeat(self.df[codes], lens)
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
        # key functions that need pool statistics (e.g. token rarity) get them first
        if hasattr(self.key_fn, "prepare"):
            self.key_fn.prepare(pool)
        # keys are built per pool slice and split into 8 partitions by their top
        # bits, each sorted separately, to bound transient memory
        parts = [([], []) for _ in range(N_INDEX_PARTITIONS)]
        for offset in range(0, pool.num_rows, chunk_size):
            rows, keys = self.key_fn(pool.slice(offset, chunk_size))
            rows = rows.astype(np.int32) + offset
            part = (np.asarray(keys, dtype=np.uint64) >> np.uint64(61)).astype(np.int8)
            for p in range(N_INDEX_PARTITIONS):
                sel = part == p
                parts[p][0].append(keys[sel])
                parts[p][1].append(rows[sel])
            del rows, keys, part

        def partitions():
            for p in range(N_INDEX_PARTITIONS):
                keys = np.concatenate(parts[p][0]) if parts[p][0] else np.zeros(0, np.uint64)
                rows = np.concatenate(parts[p][1]) if parts[p][1] else np.zeros(0, np.int32)
                parts[p] = None
                yield keys, rows

        self.index = InvertedIndex.from_partitions(partitions(), self.max_df)
        return self

    def query(self, s1: pa.Table, with_df: bool = False):
        rows, keys = self.key_fn(s1)
        codes = self.index.lookup(keys)
        ok = codes >= 0
        # the index was pruned at the max_df it was built with; re-checking
        # here lets a lower cap be swept without rebuilding the index
        if self.max_df is not None:
            ok[ok] = self.index.df[codes[ok]] <= self.max_df
        return self.index.expand(rows[ok], codes[ok], with_df=with_df)


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


def block_to_disk(strategies, s1_tables: dict, pool: pa.Table, workdir,
                  chunk_size: int = 20_000, log=print, with_df: bool = False):
    """Low-memory blocking: one strategy index in memory at a time.

    Each strategy is fitted on the pool, queried for every S1 table in
    ``chunk_size`` slices, and its (s1_row * n_pool + pool_row) keys are
    saved per (table, chunk, strategy) under ``workdir``; the index is then
    released before the next strategy is built. ``iter_union`` merges the
    files back into the same stream ``generate_candidates`` yields.
    """
    if len(strategies) > 16:
        raise ValueError("bitmask is uint16: at most 16 strategies")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    n_pool = pool.num_rows
    for k, strat in enumerate(strategies):
        t0 = time.time()
        strat.fit(pool)
        idx = strat.index
        n_pairs = 0
        for name, s1 in s1_tables.items():
            for g, offset in enumerate(range(0, s1.num_rows, chunk_size)):
                out = strat.query(s1.slice(offset, chunk_size), with_df=with_df)
                q, c = out[0], out[1]
                np.save(workdir / f"{name}_{g}_{k}.npy",
                        (q.astype(np.int64) + offset) * n_pool + c)
                if with_df:  # document frequency of the key behind each pair
                    np.save(workdir / f"{name}_{g}_{k}_df.npy", out[2].astype(np.int32))
                n_pairs += len(q)
        log(f"  {strat.name}: {time.time() - t0:.1f}s | kept {len(idx.vocab):,}/"
            f"{idx.n_keys_total:,} keys, {len(idx.postings):,}/{idx.n_postings_total:,} "
            f"postings | {n_pairs:,} raw pairs")
        strat.index = idx = None
        if hasattr(strat.key_fn, "prepare"):  # drop pool statistics too
            for attr in ("rarity", "name_rarity", "addr_rarity"):
                if hasattr(strat.key_fn, attr):
                    getattr(strat.key_fn, attr).__dict__.pop("vocab", None)
                    getattr(strat.key_fn, attr).__dict__.pop("df", None)
        gc.collect()


def iter_union(name, s1: pa.Table, n_pool: int, n_strategies: int, workdir,
               chunk_size: int = 20_000, delete: bool = True):
    """Merge ``block_to_disk`` output for one S1 table, chunk by chunk.

    Yields ``(offset, s1_rows, pool_rows, bits)`` exactly like
    ``generate_candidates``.
    """
    workdir = Path(workdir)
    for g, offset in enumerate(range(0, s1.num_rows, chunk_size)):
        keys, bits = [], []
        for k in range(n_strategies):
            path = workdir / f"{name}_{g}_{k}.npy"
            arr = np.load(path)
            keys.append(arr)
            bits.append(np.full(len(arr), 1 << k, dtype=np.uint16))
            if delete:
                path.unlink()
        keys = np.concatenate(keys)
        bits = np.concatenate(bits)
        order = np.argsort(keys, kind="stable")
        keys, bits = keys[order], bits[order]
        if len(keys):
            starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
            bits = np.bitwise_or.reduceat(bits, starts)
            keys = keys[starts]
        yield offset, keys // n_pool, keys % n_pool, bits


def iter_union_scored(name, s1: pa.Table, n_pool: int, caps, workdir,
                      chunk_size: int = 20_000, delete: bool = True):
    """Like ``iter_union`` (needs ``block_to_disk(..., with_df=True)``), plus
    per-pair blocking evidence for ranking candidates.

    Yields ``(offset, s1_rows, pool_rows, bits, n_hits, min_df, rarity)``:
    ``n_hits`` = number of strategies that retrieved the pair, ``min_df`` =
    document frequency of the rarest key it shares (any strategy), and
    ``rarity`` = sum over retrieving strategies k of
    1 - ln(df_k) / ln(cap_k + 1), with df_k the rarest key of strategy k it
    shares (1 for a unique key, ~0 at the cap).
    """
    workdir = Path(workdir)
    for g, offset in enumerate(range(0, s1.num_rows, chunk_size)):
        keys, bits, dfs, rar = [], [], [], []
        for k, cap in enumerate(caps):
            kpath = workdir / f"{name}_{g}_{k}.npy"
            dpath = workdir / f"{name}_{g}_{k}_df.npy"
            key, df = np.load(kpath), np.load(dpath).astype(np.int64)
            if delete:
                kpath.unlink()
                dpath.unlink()
            # one entry per pair per strategy: its rarest shared key
            order = np.lexsort((df, key))
            key, df = key[order], df[order]
            first = np.r_[True, key[1:] != key[:-1]] if len(key) else np.zeros(0, bool)
            key, df = key[first], df[first]
            keys.append(key)
            dfs.append(df)
            bits.append(np.full(len(key), 1 << k, dtype=np.uint16))
            rar.append(1.0 - np.log(np.maximum(df, 1)) / np.log(cap + 1))
        keys, bits = np.concatenate(keys), np.concatenate(bits)
        dfs, rar = np.concatenate(dfs), np.concatenate(rar)
        order = np.argsort(keys, kind="stable")
        keys, bits, dfs, rar = keys[order], bits[order], dfs[order], rar[order]
        if len(keys):
            starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
            n_hits = np.diff(np.r_[starts, len(keys)])
            bits = np.bitwise_or.reduceat(bits, starts)
            dfs = np.minimum.reduceat(dfs, starts)
            rar = np.add.reduceat(rar, starts)
            keys = keys[starts]
        else:
            n_hits = np.zeros(0, np.int64)
        yield offset, keys // n_pool, keys % n_pool, bits, n_hits, dfs, rar


def rank_within_groups(q, *sort_keys):
    """0-based rank of each element within its run of equal ``q`` under
    ``sort_keys`` (numpy lexsort order: the last key is the primary one)."""
    order = np.lexsort((*sort_keys, q))
    q_sorted = q[order]
    starts = (np.flatnonzero(np.r_[True, q_sorted[1:] != q_sorted[:-1]]) if len(q)
              else np.zeros(0, np.int64))
    sizes = np.diff(np.r_[starts, len(q)])
    rank = np.empty(len(q), np.int64)
    rank[order] = np.arange(len(q)) - np.repeat(starts, sizes)
    return rank


def iter_union_top_n(name, s1: pa.Table, n_pool: int, caps, workdir, top_n: int,
                     chunk_size: int = 20_000, delete: bool = True):
    """``iter_union_scored`` keeping each S1 record's ``top_n`` candidates by
    rarity (ties broken by pool row). Yields ``(offset, s1_rows, pool_rows,
    bits)`` like ``iter_union``."""
    for offset, q, c, bits, _, _, rarity in iter_union_scored(
            name, s1, n_pool, caps, workdir, chunk_size, delete):
        keep = rank_within_groups(q, c, -rarity) < top_n
        yield offset, q[keep], c[keep], bits[keep]
