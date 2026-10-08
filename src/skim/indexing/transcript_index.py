"""Segment-level transcript retrieval (BM25 + exact-term boosting).

This is the *cheap* half of the agent: everything here runs on text only, so it
can narrow a two-hour lecture down to a handful of candidate windows before any
frame is decoded or any VLM token is spent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..logging_utils import get_logger
from ..media.transcript import Transcript, TranscriptSegment, format_timestamp
from .text import keywords, tokenize

logger = get_logger("index")

try:  # rank_bm25 is small and pure-python
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None  # type: ignore[assignment]


@dataclass
class SearchHit:
    segment: TranscriptSegment
    score: float
    matched_terms: list[str] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.segment.start

    @property
    def end(self) -> float:
        return self.segment.end

    def as_dict(self) -> dict:
        return {
            "start": round(self.segment.start, 2),
            "end": round(self.segment.end, 2),
            "time": format_timestamp(self.segment.start),
            "text": self.segment.text,
            "score": round(self.score, 4),
            "matched_terms": self.matched_terms,
        }


@dataclass
class CandidateWindow:
    """A coarse, cheap-to-produce guess at where the answer lives."""

    start: float
    end: float
    score: float
    relevance: float  # normalised to [0, 1]
    reason: str
    snippets: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "range": f"{format_timestamp(self.start)} - {format_timestamp(self.end)}",
            "score": round(self.score, 4),
            "relevance": round(self.relevance, 3),
            "reason": self.reason,
            "terms": self.terms,
            "snippets": self.snippets[:3],
        }

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"<{format_timestamp(self.start)}-{format_timestamp(self.end)} rel={self.relevance:.2f}>"


class TranscriptIndex:
    """BM25 index over transcript segments with exact-term boosting."""

    def __init__(self, transcript: Transcript):
        self.transcript = transcript
        self.segments: list[TranscriptSegment] = [
            seg for seg in transcript.segments if seg.text.strip()
        ]
        self._tokenized = [tokenize(seg.text) for seg in self.segments]
        self._bm25 = None
        if BM25Okapi is not None and any(self._tokenized):
            try:
                self._bm25 = BM25Okapi(self._tokenized)
            except Exception as exc:  # pragma: no cover - degenerate corpora
                logger.warning("BM25 unavailable (%s); falling back to term overlap", exc)
                self._bm25 = None

    # ------------------------------------------------------------------ search
    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        start: float | None = None,
        end: float | None = None,
        min_score: float = 0.0,
    ) -> list[SearchHit]:
        """Return the best-matching segments, optionally restricted to a window."""
        if not self.segments:
            return []

        query_terms = keywords(query, max_terms=16) or tokenize(query)
        if not query_terms:
            return []

        candidates = [
            index
            for index, segment in enumerate(self.segments)
            if (start is None or segment.end > start) and (end is None or segment.start < end)
        ]
        if not candidates:
            return []

        bm25_scores: dict[int, float] = {}
        if self._bm25 is not None:
            raw = self._bm25.get_scores(query_terms)
            bm25_scores = {index: float(raw[index]) for index in candidates}

        max_bm25 = max(bm25_scores.values(), default=0.0) or 1.0

        hits: list[SearchHit] = []
        for index in candidates:
            text_lower = self.segments[index].text.lower()
            matched = [term for term in query_terms if term in text_lower]
            if not matched:
                continue

            exact_boost = sum(1.0 for term in matched if len(term) >= 2) / len(query_terms)
            coverage = len(matched) / len(query_terms)
            normalised_bm25 = max(0.0, bm25_scores.get(index, 0.0)) / max_bm25
            score = 0.55 * normalised_bm25 + 0.25 * coverage + 0.20 * exact_boost

            if score >= min_score:
                hits.append(SearchHit(segment=self.segments[index], score=score, matched_terms=matched))

        hits.sort(key=lambda hit: (-hit.score, hit.start))
        return hits[:top_k]

    def locate_term(self, term: str, *, limit: int = 20) -> list[TranscriptSegment]:
        """Every segment mentioning ``term`` (case-insensitive substring)."""
        needle = term.lower().strip()
        if not needle:
            return []
        return [seg for seg in self.segments if needle in seg.text.lower()][:limit]

    # --------------------------------------------------------------- coarse
    def coarse_windows(
        self,
        question: str,
        *,
        top_k: int = 6,
        merge_gap: float = 35.0,
        pad: float = 12.0,
        max_windows: int = 4,
        max_span: float = 150.0,
    ) -> list[CandidateWindow]:
        """Turn query hits into a few padded, merged candidate windows.

        ``merge_gap`` glues hits that are close together (they usually belong to
        the same explanation) and ``max_span`` stops that from degenerating into
        one window covering the entire lecture.
        """
        hits = self.search(question, top_k=top_k)
        if not hits:
            return []

        hits.sort(key=lambda hit: hit.start)
        groups: list[list[SearchHit]] = [[hits[0]]]
        for hit in hits[1:]:
            group = groups[-1]
            close_enough = hit.start - group[-1].end <= merge_gap
            within_span = (hit.end - group[0].start) <= max_span
            if close_enough and within_span:
                group.append(hit)
            else:
                groups.append([hit])

        duration = self.transcript.duration
        windows: list[CandidateWindow] = []
        for group in groups:
            score = sum(hit.score for hit in group)
            begin = max(0.0, min(hit.start for hit in group) - pad)
            finish = max(hit.end for hit in group) + pad
            if duration:
                finish = min(finish, duration)
            terms: list[str] = []
            for hit in group:
                for term in hit.matched_terms:
                    if term not in terms:
                        terms.append(term)
            windows.append(
                CandidateWindow(
                    start=begin,
                    end=finish,
                    score=score,
                    relevance=0.0,  # filled in below
                    reason=f"{len(group)} transcript hit(s) matching {', '.join(terms[:5])}",
                    snippets=[hit.segment.text for hit in group[:3]],
                    terms=terms,
                )
            )

        windows.sort(key=lambda window: -window.score)
        windows = windows[:max_windows]

        top = windows[0].score or 1.0
        for window in windows:
            # sqrt keeps secondary windows meaningful instead of collapsing to ~0
            window.relevance = round(min(1.0, (window.score / top) ** 0.5), 4)

        # Present them back in chronological order for human readability.
        windows.sort(key=lambda window: window.start)
        return windows