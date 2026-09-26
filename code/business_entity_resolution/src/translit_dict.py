"""Transliteration dictionary learned from training ground truth.

unidecode turns Indic-script names into Latin tokens that rarely equal the
English spelling ("टेक्नोलॉजी" -> "ttekneaalji" vs "technology"), so
token-level keys and features miss these matches. The ground truth contains
many (Latin S1 name, non-Latin candidate name) true pairs; aligning their
meaningful tokens position by position when both names have the same number
of tokens yields a mapping transliterated-token -> English token.

Only matcher_train S1 entities are used to learn it (never the blocking
validation or matcher_val samples). A mapping is kept when it was seen at
least MIN_COUNT times and accounts for at least MIN_SHARE of all alignments of
that transliterated token (identity alignments included, so tokens that
already match are left alone).
"""

import json
from collections import Counter, defaultdict

from normalize import LEGAL_FORM_TOKENS, WEB_TOKENS

MIN_COUNT = 3
MIN_SHARE = 0.6
STOP = LEGAL_FORM_TOKENS | WEB_TOKENS


def meaningful_tokens(name: str) -> list:
    return [t for t in name.split() if len(t) > 1 and t not in STOP]


def learn(pairs) -> dict:
    """``pairs``: iterable of (s1_translit_name, candidate_translit_name)."""
    counts = defaultdict(Counter)
    for s1_name, cand_name in pairs:
        a, b = meaningful_tokens(s1_name), meaningful_tokens(cand_name)
        if not a or len(a) != len(b):
            continue
        for s1_tok, cand_tok in zip(a, b):
            counts[cand_tok][s1_tok] += 1
    mapping = {}
    for cand_tok, targets in counts.items():
        target, n = targets.most_common(1)[0]
        if target != cand_tok and n >= MIN_COUNT and n / sum(targets.values()) >= MIN_SHARE:
            mapping[cand_tok] = target
    return mapping


class TranslitMap:
    """Token transform: transliterated token -> learned English token."""

    def __init__(self, mapping: dict):
        self.mapping = mapping

    def __call__(self, token: str) -> str:
        return self.mapping.get(token, token)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f)["mapping"])


class MappedPhonetic:
    """Learned mapping, then the phonetic skeleton (for core-name keys)."""

    def __init__(self, translit_map: TranslitMap, skeleton):
        self.translit_map, self.skeleton = translit_map, skeleton

    def __call__(self, token: str) -> str:
        return self.skeleton(self.translit_map(token))
