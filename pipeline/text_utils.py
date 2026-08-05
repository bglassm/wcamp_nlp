from __future__ import annotations

from typing import List

from kiwipiepy import Kiwi

# Shared Kiwi instance to avoid repeated initialization overhead
_kiwi = Kiwi()

# POS tags to drop: particles, endings, symbols, and punctuation
_EXCLUDED_POS = {
    "JKS",
    "JKC",
    "JKG",
    "JKO",
    "JKB",
    "JKV",
    "JKQ",
    "JC",
    "JX",
    "EP",
    "EF",
    "EC",
    "ETN",
    "ETM",
    "SF",
    "SP",
    "SS",
    "SE",
    "SO",
    "SW",
}


def _normalize_token(token) -> str | None:
    base = getattr(token, "lemma", None) or getattr(token, "form", "")
    if not base:
        return None
    base = base.strip().lower()
    if not base:
        return None
    return base


def tokenize_lemmas(text: str) -> List[str]:
    """Tokenize Korean text into normalized lemma-like tokens using Kiwi.

    - Drops particles, endings, and symbol-like POS tags.
    - Prefers lemma when available to unify conjugations (e.g., "비싸요" → "비싸다").
    """

    tokens: List[str] = []
    for tok in _kiwi.tokenize(text or ""):
        tag = getattr(tok, "tag", "")
        if tag in _EXCLUDED_POS:
            continue
        normalized = _normalize_token(tok)
        if normalized:
            tokens.append(normalized)
    return tokens


__all__ = ["tokenize_lemmas"]
