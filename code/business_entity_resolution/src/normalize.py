"""Text normalization for business names and addresses.

Deliberately country-agnostic: nothing here branches on the value of the
`country` field. Dictionaries are noise-pattern-driven (legal-suffix and
address-abbreviation synonyms called out in the problem statement), not
country-driven, so they apply uniformly to US, India, and the unseen
France test rows alike.
"""

import re
import unicodedata

from unidecode import unidecode

# Legal-suffix variants collapsed to a single canonical token. Keys and
# canonical values are pre-lowercased tokens (compared after tokenizing on
# whitespace), so this runs after punctuation cleanup.
LEGAL_SUFFIX_MAP = {
    "corporation": "corp",
    "corp": "corp",
    "incorporated": "inc",
    "inc": "inc",
    "limited": "ltd",
    "ltd": "ltd",
    "llc": "llc",
    "l.l.c": "llc",
    "llp": "llp",
    "l.l.p": "llp",
    "pvt": "pvt",
    "private": "pvt",
    "plc": "plc",
    "co": "co",
    "company": "co",
    "and co": "co",
}

# Address-component abbreviation variants collapsed to a canonical token.
ADDRESS_ABBREV_MAP = {
    "road": "rd",
    "rd": "rd",
    "street": "st",
    "st": "st",
    "avenue": "ave",
    "ave": "ave",
    "boulevard": "blvd",
    "blvd": "blvd",
    "drive": "dr",
    "dr": "dr",
    "lane": "ln",
    "ln": "ln",
    "court": "ct",
    "ct": "ct",
    "place": "pl",
    "pl": "pl",
    "highway": "hwy",
    "hwy": "hwy",
    "apartment": "apt",
    "apt": "apt",
    "suite": "ste",
    "ste": "ste",
    "floor": "fl",
    "fl": "fl",
    "building": "bldg",
    "bldg": "bldg",
}

# Tokens that carry no identity: canonical legal forms (post LEGAL_SUFFIX_MAP),
# their French equivalents, and the most frequent unidecode spellings of
# "Private Limited" / "LLP" produced from Indic scripts (e.g. "प्राइवेट
# लिमिटेड" -> "praaivett limittedd", "प्रा. लि." -> "praa li"). Dropped when
# building the order-independent "core name" key.
LEGAL_FORM_TOKENS = frozenset({
    "corp", "inc", "ltd", "llc", "llp", "pvt", "plc", "co", "lp", "the", "and",
    "of", "limited", "private", "sa", "sas", "sasu", "sarl", "sci", "eurl", "snc",
    "limittedd", "limittett", "limirrrrdd", "praaivett", "praiveett",
    "piraiveett", "praaibhett", "praivrrrr", "praa", "li", "elelpii",
})

# Domain / URL tokens left behind when a name is a website
# ("wilfordhancock.com" -> "wilfordhancock com"). "in" is deliberately not
# here: it is also an ordinary word inside names ("Made in Texas").
WEB_TOKENS = frozenset({"www", "com", "net", "org", "http", "https"})

_WHITESPACE_RE = re.compile(r"\s+")
_REPEAT_RE = re.compile(r"(.)\1+")
_VOWELS_RE = re.compile(r"[aeiouy]")
_DIGIT_RUN_RE = re.compile(r"\d+")

# Consonant clusters that transliteration spells inconsistently (aspirates,
# sibilants, w/v, z/j, x/ks), plus the anusvara n/m before labials
# ("inpeks" vs "impex"). Applied after c/q -> k, so "ch" -> "kh" -> "k".
_INDIC_FOLDS = [
    (re.compile(r"sh"), "s"), (re.compile(r"th"), "t"), (re.compile(r"dh"), "d"),
    (re.compile(r"bh"), "b"), (re.compile(r"kh"), "k"), (re.compile(r"gh"), "g"),
    (re.compile(r"w"), "v"), (re.compile(r"z"), "j"), (re.compile(r"x"), "ks"),
    (re.compile(r"n(?=[bp])"), "m"),
]
# Digits used as look-alike letters inside words ("6roovy", "Digita1", "5cience").
_DIGIT_LOOKALIKES = str.maketrans("01345678", "oleasgtb")
_POSTAL_PROXY_RE = re.compile(r"\b\d{4,6}\b")


def _as_text(value) -> str:
    """Coerce a possibly-missing field (None/NaN/float) to a plain string."""
    if value is None:
        return ""
    if isinstance(value, float):
        # pandas represents missing strings as NaN, which is a float.
        return ""
    return str(value)


def nfkc_casefold(text: str) -> str:
    """Unicode-normalize and casefold.

    NFKC + casefold (not just .lower()) so that non-Latin scripts (e.g. the
    Devanagari business names observed in Source 2) and full/half-width or
    compatibility-decomposed characters compare consistently, without
    assuming a Latin alphabet.
    """
    return unicodedata.normalize("NFKC", text).casefold()


def clean_punctuation(text: str) -> str:
    """Normalize '&', strip punctuation, collapse whitespace.

    Keeps a character if it's alphanumeric (Unicode letters/digits, any
    script) or a combining mark (Unicode category ``M*``) — the latter is
    essential for scripts like Devanagari, where vowel signs and virama are
    combining marks, not "letters". Python's ``\\w`` regex class excludes
    combining marks, so a naive ``[^\\w\\s]`` strip silently mangles those
    scripts; this loop avoids that trap.
    """
    text = text.replace("&", " and ")
    kept = [
        ch if (ch.isspace() or ch.isalnum() or unicodedata.category(ch).startswith("M"))
        else " "
        for ch in text
    ]
    text = "".join(kept)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _collapse_tokens(text: str, synonym_map: dict) -> str:
    tokens = text.split(" ")
    collapsed = [synonym_map.get(tok, tok) for tok in tokens]
    return " ".join(t for t in collapsed if t)


def normalize_business_name(raw_name) -> str:
    """Full normalization pipeline for a business_name field."""
    text = _as_text(raw_name)
    text = nfkc_casefold(text)
    text = clean_punctuation(text)
    text = _collapse_tokens(text, LEGAL_SUFFIX_MAP)
    return text


def normalize_business_address(raw_address) -> str:
    """Full normalization pipeline for a business_address field."""
    text = _as_text(raw_address)
    text = nfkc_casefold(text)
    text = clean_punctuation(text)
    text = _collapse_tokens(text, ADDRESS_ABBREV_MAP)
    return text


def transliterate_business_name(raw_name) -> str:
    """Script-agnostic name variant for cross-script matching.

    Source 1 (the reference set) is entirely Latin-script, but Source 2/3
    sometimes carry the *same* business transliterated into another script
    (observed directly in the ground truth: Devanagari names phonetically
    matching Latin-script Source 1 names, e.g. "Red Ventures Private
    Limited" <-> "रेड वेंचर्स प्राइवेट लिमिटेड"). Native character-level
    similarity between those two strings is ~0 even on a true match, so
    this folds any script to its closest ASCII phonetic approximation via
    ``unidecode`` (offline, rule-based — no external lookup) before running
    the same cleanup pipeline as ``normalize_business_name``. On
    already-Latin text this is a no-op past casefolding, so it's applied
    uniformly to every row regardless of source or country rather than
    conditionally — which also folds accented Latin characters (e.g. French
    "é" -> "e") for free.
    """
    text = _as_text(raw_name)
    text = unidecode(text)
    text = nfkc_casefold(text)
    text = clean_punctuation(text)
    text = _collapse_tokens(text, LEGAL_SUFFIX_MAP)
    return text


def transliterate_business_address(raw_address) -> str:
    """Script-agnostic address variant, mirroring transliterate_business_name."""
    text = _as_text(raw_address)
    text = unidecode(text)
    text = nfkc_casefold(text)
    text = clean_punctuation(text)
    text = _collapse_tokens(text, ADDRESS_ABBREV_MAP)
    return text


def consonant_skeleton(token: str, indic_folds: bool = False,
                       digit_lookalikes: bool = False) -> str:
    """Coarse phonetic key for one already-transliterated token.

    unidecode of Indic scripts produces doubled consonants and shifted vowels
    ("ddevlprs", "ttredding", "investtmentt"), so exact token equality with
    the Latin-script spelling rarely holds. Collapsing letter runs, folding
    c/q -> k and ph -> f, and dropping non-leading vowels maps both sides to
    the same key ("developers" and "ddevlprs" -> "dvlprs").

    ``indic_folds`` also folds inconsistently transliterated consonant
    clusters (sh/s, th/t, bh/b, w/v, ...), so "sree", "shri" and "shrii" all
    map to "sr". ``digit_lookalikes`` first maps digits inside alphanumeric
    tokens to the letters they imitate ("6roovy" -> "groovy"). Pure-number
    tokens, and any other token containing digits, are returned unchanged.
    """
    if digit_lookalikes and any(ch.isalpha() for ch in token):
        token = token.translate(_DIGIT_LOOKALIKES)
    if any(ch.isdigit() for ch in token):
        return token
    t = token.replace("ph", "f").replace("c", "k").replace("q", "k")
    if indic_folds:
        for pattern, repl in _INDIC_FOLDS:
            t = pattern.sub(repl, t)
    t = _REPEAT_RE.sub(r"\1", t)
    if len(t) > 1:
        t = t[0] + _VOWELS_RE.sub("", t[1:])
    return _REPEAT_RE.sub(r"\1", t)


def phonetic_skeleton(token: str) -> str:
    """consonant_skeleton with Indic cluster folds and digit look-alikes on."""
    return consonant_skeleton(token, indic_folds=True, digit_lookalikes=True)


def normalize_house_number(token: str) -> str:
    """Canonical form of a numeric address token, or "" if it has no digits.

    Keeps the first digit run and strips leading zeros, so "05204" -> "5204",
    "002610" -> "2610", "2609c" -> "2609", "23rd" / "23nd" -> "23".
    """
    m = _DIGIT_RUN_RE.search(token)
    if m is None:
        return ""
    return m.group().lstrip("0") or "0"


def extract_postal_code_proxy(raw_address) -> str:
    """Heuristic postal-code-like token: the last standalone 4-6 digit run.

    Not a real parser — just a cheap regex proxy (covers US 5-digit ZIP and
    Indian 6-digit PIN reasonably well; French postal codes are also
    5-digit, so this degrades gracefully to the same heuristic there
    without any country-specific branching). Returns "" when no such token
    is found (e.g. missing/partial addresses).
    """
    text = _as_text(raw_address)
    matches = _POSTAL_PROXY_RE.findall(text)
    return matches[-1] if matches else ""
