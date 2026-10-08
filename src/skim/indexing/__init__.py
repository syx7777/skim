"""Retrieval / localisation layer."""

from .chapters import Chapter, build_chapters, rank_chapters
from .coarse import CoarseResult, coarse_localize, question_needs_vision, transcript_density
from .text import contains_term, keywords, tokenize
from .transcript_index import CandidateWindow, SearchHit, TranscriptIndex

__all__ = [
    "Chapter",
    "build_chapters",
    "rank_chapters",
    "CoarseResult",
    "coarse_localize",
    "question_needs_vision",
    "transcript_density",
    "contains_term",
    "keywords",
    "tokenize",
    "CandidateWindow",
    "SearchHit",
    "TranscriptIndex",
]