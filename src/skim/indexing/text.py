"""Bilingual tokenisation helpers (Chinese + English) for retrieval."""

from __future__ import annotations

import re
from functools import lru_cache

_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9_+\-\.]*")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_HAS_CJK = re.compile(r"[\u4e00-\u9fff]")

#: Very common words that carry no retrieval signal.
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "is", "are", "was", "were", "and", "or",
    "for", "on", "at", "by", "with", "that", "this", "it", "as", "be", "from",
    "what", "which", "how", "why", "when", "where", "does", "do", "did", "can",
    "的", "了", "是", "在", "和", "与", "我", "你", "他", "这", "那", "个", "就",
    "什么", "怎么", "为什么", "哪", "哪一", "如何", "老师", "讲", "说", "这个",
}


@lru_cache(maxsize=1)
def _jieba():
    try:
        # jieba ships regexes with invalid escape sequences, which CPython reports
        # as SyntaxWarning at compile time. They are harmless but noisy in a CLI.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import jieba

        jieba.setLogLevel(60)  # silence the "building prefix dict" banner
        return jieba
    except ImportError:  # pragma: no cover - jieba is a hard dependency
        return None


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Tokenise mixed Chinese/English text into lowercase terms."""
    if not text:
        return []

    tokens: list[str] = []
    for match in _LATIN.finditer(text):
        tokens.append(match.group(0).lower())

    if _HAS_CJK.search(text):
        jieba = _jieba()
        if jieba is not None:
            tokens.extend(
                token
                for token in jieba.lcut(text)
                if _HAS_CJK.search(token) and len(token) > 1
            )
        else:  # pragma: no cover
            tokens.extend(_CJK.findall(text))

    if keep_stopwords:
        return tokens
    return [token for token in tokens if token and token not in _STOPWORDS]


def keywords(text: str, *, max_terms: int = 12, min_length: int = 2) -> list[str]:
    """Ordered, de-duplicated salient terms (acronyms first, then by frequency)."""
    tokens = tokenize(text)
    if not tokens:
        return []

    frequency: dict[str, int] = {}
    for token in tokens:
        frequency[token] = frequency.get(token, 0) + 1

    # Latin/acronym terms are usually the most discriminative in lecture Q&A.
    acronyms = [t for t in frequency if _LATIN.fullmatch(t) and len(t) >= 2]
    others = [t for t in frequency if t not in acronyms and len(t) >= min_length]

    acronyms.sort(key=lambda t: (-frequency[t], t))
    others.sort(key=lambda t: (-frequency[t], t))
    ordered = acronyms + others

    seen: set[str] = set()
    result: list[str] = []
    for token in ordered:
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= max_terms:
            break
    return result


def contains_term(text: str, term: str) -> bool:
    """Case-insensitive containment that works for CJK too."""
    return term.lower() in text.lower()