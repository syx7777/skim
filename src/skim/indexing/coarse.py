"""Coarse localisation: the cheap, text-only first pass.

Combines two cheap signals -- BM25 hits over the transcript and topical chapter
matching -- into a small set of candidate windows. No frame is decoded and no
VLM token is spent here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..media.transcript import format_timestamp
from .chapters import Chapter, rank_chapters
from .transcript_index import CandidateWindow, TranscriptIndex
from .text import keywords

#: Question phrasings that imply the answer lives in the *visual* channel.
_VISUAL_CUES = {
    "ppt", "slide", "slides", "figure", "figures", "diagram", "chart", "plot",
    "formula", "equation", "derivation", "derive", "board", "blackboard",
    "code", "snippet", "graph", "table", "screenshot", "draw", "drew",
    "图", "图片", "公式", "推导", "板书", "代码", "表格", "画", "图里", "图上",
    "幻灯片", "课件", "演示", "曲线", "坐标系", "图示",
}


@dataclass
class CoarseResult:
    """Output of the cheap first pass."""

    windows: list[CandidateWindow] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)
    matched_chapters: list[tuple[Chapter, float]] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    needs_vision: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "needs_vision": self.needs_vision,
            "terms": self.terms,
            "windows": [window.as_dict() for window in self.windows],
            "matched_chapters": [
                {"title": chapter.title, "range": f"{chapter.start:.0f}-{chapter.end:.0f}s", "score": round(score, 3)}
                for chapter, score in self.matched_chapters
            ],
            "notes": self.notes,
        }


def question_needs_vision(question: str) -> bool:
    """Heuristic: does the question reference visual material explicitly?"""
    lowered = question.lower()
    return any(cue in lowered for cue in _VISUAL_CUES)


def transcript_density(transcript, start: float, end: float) -> float:
    """Characters of speech per second inside a window."""
    span = max(1e-6, end - start)
    text = transcript.text_between(start, end)
    return len(re.sub(r"\s+", "", text)) / span


def coarse_localize(
    question: str,
    index: TranscriptIndex,
    chapters: list[Chapter] | None = None,
    *,
    top_k: int = 8,
    max_windows: int = 4,
    pad: float = 12.0,
) -> CoarseResult:
    """Produce candidate windows for a question using transcript + chapters only."""
    chapters = chapters or []
    result = CoarseResult(chapters=chapters, terms=keywords(question, max_terms=12))

    windows = index.coarse_windows(question, top_k=top_k, pad=pad, max_windows=max_windows)
    result.windows.extend(windows)

    result.matched_chapters = rank_chapters(question, chapters, top_k=3)
    for chapter, score in result.matched_chapters:
        if score < 0.2:
            continue
        if any(_overlaps(window, chapter.start, chapter.end) > 0.5 for window in result.windows):
            continue
        result.windows.append(
            CandidateWindow(
                start=chapter.start,
                end=chapter.end,
                score=score,
                relevance=round(min(1.0, score), 4),
                reason=f"chapter match: {chapter.title}",
                snippets=[chapter.summary] if chapter.summary else [],
                terms=chapter.keywords[:6],
            )
        )

    result.windows.sort(key=lambda window: -window.relevance)
    result.windows = result.windows[: max_windows + 2]
    result.windows.sort(key=lambda window: window.start)

    result.needs_vision = question_needs_vision(question)
    if result.needs_vision:
        result.notes.append("question references visual material (slides/formulas/code)")
    if not result.windows:
        result.notes.append("no transcript hits; agent should scan visually")

    # Flag windows where the teacher likely stopped talking to show something:
    # low speech density is a cheap hint that the visual channel carries meaning.
    for window in result.windows:
        density = transcript_density(index.transcript, window.start, window.end)
        if density < 4.0 and (window.end - window.start) > 20:
            result.notes.append(
                f"sparse speech in {format_timestamp(window.start)}"
                f"-{format_timestamp(window.end)} ({density:.1f} chars/s) -> likely visual-only content"
            )
    return result


def _overlaps(window: CandidateWindow, start: float, end: float) -> float:
    """Fraction of ``window`` that lies inside ``[start, end]``."""
    inter = max(0.0, min(window.end, end) - max(window.start, start))
    span = max(1e-6, window.end - window.start)
    return inter / span