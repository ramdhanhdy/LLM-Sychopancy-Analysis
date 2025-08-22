# scoring/text_utils.py
"""Text processing utilities for scoring."""

import re
import pandas as pd
from typing import List
from functools import lru_cache
from ..api import SCORING_CONFIG

# Optional dependencies
try:
    from nltk.stem import SnowballStemmer
    _HAS_NLTK = True
except Exception:
    SnowballStemmer = None  # type: ignore
    _HAS_NLTK = False

# Pre-compiled regex patterns for efficiency
_re_ws = re.compile(r"\s+")
_re_sent = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-zA-Z']+")

# Enriched patterns for common phrases
_ENRICHED_REGEX = {
    "bad idea": r"\b(bad|terrible|awful|unwise|poor)\s+(idea|plan|decision)\b",
    "good idea": r"\b(good|great|excellent|sound|sensible)\s+(idea|plan|decision)\b",
}


def _phrase_to_regex(p: str) -> str:
    """Convert phrase to regex pattern with word boundaries."""
    if p in _ENRICHED_REGEX:
        return _ENRICHED_REGEX[p]
    # Escape and allow whitespace variations between words
    parts = [re.escape(tok) for tok in p.strip().split()]
    return r"\b" + r"\s+".join(parts) + r"\b"


def _compile_regex_list(phrases: List[str]) -> List[re.Pattern]:
    """Compile list of phrases to regex patterns."""
    return [re.compile(_phrase_to_regex(p), flags=re.IGNORECASE) for p in phrases]


def get_regex_patterns():
    """Get compiled regex patterns based on config."""
    from .constants import AGREE, DISAGREE, FLATTERY, INTENSIFIERS, HEDGES, REFUSAL, SAFE_ALT, COUNTERARG
    
    if not SCORING_CONFIG.get("USE_REGEX", False):
        return {}
    
    return {
        "AGREE": _compile_regex_list(AGREE),
        "DISAGREE": _compile_regex_list(DISAGREE),
        "FLATTERY": _compile_regex_list(FLATTERY),
        "INTENSIFIERS": _compile_regex_list(INTENSIFIERS),
        "HEDGES": _compile_regex_list(HEDGES),
        "REFUSAL": _compile_regex_list(REFUSAL),
        "SAFE_ALT": _compile_regex_list(SAFE_ALT),
        "COUNTERARG": _compile_regex_list(COUNTERARG),
    }


@lru_cache(maxsize=1)
def _get_stemmer():
    """Get stemmer instance if available and enabled."""
    if not SCORING_CONFIG.get("USE_STEMMING", False):
        return None
    if not _HAS_NLTK:
        return None
    try:
        return SnowballStemmer("english")
    except Exception:
        return None


def _stem_tokens(text: str, stemmer) -> List[str]:
    """Extract and stem tokens from text."""
    toks = [t.lower() for t in _WORD_RE.findall(text)]
    if not stemmer:
        return toks
    return [stemmer.stem(t) for t in toks]


def _stem_join(tokens: List[str]) -> str:
    """Join stemmed tokens back to string."""
    return " ".join(tokens)


def norm_text(x: str) -> str:
    """Normalize text for scoring."""
    # Robust to NaN/None/non-string values coming from CSVs
    try:
        if x is None or (pd.isna(x)):
            return ""
    except Exception:
        if x is None:
            return ""
    # Coerce non-strings to string before normalization
    if not isinstance(x, str):
        x = str(x)
    return _re_ws.sub(" ", x.strip().lower())


def split_sentences(x: str) -> List[str]:
    """Split text into sentences."""
    x = x.strip()
    if not x:
        return []
    return _re_sent.split(x)


def first_para(text: str) -> str:
    """Extract first paragraph (first 2 sentences)."""
    paras = text.split("\n\n")
    head = paras[0] if paras else text
    sents = split_sentences(head)
    return " ".join(sents[:2]) if sents else head


def _count_hits_raw(text: str, phrases: List[str]) -> int:
    """Simple substring counts."""
    t = text
    s = 0
    for p in phrases:
        s += t.count(p)
    return s


def _count_hits_regex(text: str, key: str) -> int:
    """Count hits using regex patterns."""
    patterns = get_regex_patterns()
    pats = patterns.get(key)
    if not pats:
        return 0
    return sum(len(p.findall(text)) for p in pats)


def _count_hits_stemmed(text: str, phrases: List[str]) -> int:
    """Count hits using stemming."""
    stemmer = _get_stemmer()
    if not stemmer:
        return _count_hits_raw(text, phrases)
    text_stems = _stem_join(_stem_tokens(text, stemmer))
    cnt = 0
    for p in phrases:
        p_stem = _stem_join(_stem_tokens(p, stemmer))
        if not p_stem:
            continue
        cnt += text_stems.count(p_stem)
    return cnt


def count_hits_adv(text: str, phrases: List[str], key: str) -> int:
    """Count hits using stemming, regex, or raw depending on config."""
    if SCORING_CONFIG.get("USE_STEMMING", False) and _get_stemmer() is not None:
        return _count_hits_stemmed(text, phrases)
    if SCORING_CONFIG.get("USE_REGEX", False):
        return _count_hits_regex(text, key)
    return _count_hits_raw(text, phrases)
