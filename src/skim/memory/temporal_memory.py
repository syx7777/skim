"""Temporal Memory: a compact record of what the agent has already looked at.

Two problems this solves:

1. **Redundant work.** Without it the agent happily re-inspects 40-43 min after a
   few turns and burns another set of VLM calls. Every observation is stored
   with its time interval so :meth:`TemporalMemory.new_ratio` can tell the agent
   how much of a proposed window is genuinely unexplored.
2. **Lost context across steps.** Knowledge points, visual transcripts and
   evidence survive independently of the chat history, so the prompt can be
   rebuilt cheaply on every step instead of growing without bound.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Iterable

from ..logging_utils import get_logger
from ..media.transcript import format_timestamp

logger = get_logger("memory")


@dataclass
class Observation:
    """One recorded look at the video."""

    start: float
    end: float
    method: str = "frames"  # frames | transcript | chapters | scan
    timestamps: list[float] = field(default_factory=list)
    frames: int = 0
    knowledge_points: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    visual_text: str = ""
    relevance: float = 0.0
    answers_question: bool = False
    confidence: float = 0.0
    notes: str = ""
    step: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def label(self) -> str:
        return f"{format_timestamp(self.start)}-{format_timestamp(self.end)}"

    def as_dict(self) -> dict:
        return asdict(self)


def _merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((min(a, b), max(a, b)) for a, b in intervals if b > a)
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + 1e-6:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


class TemporalMemory:
    """Append-only log of observations plus derived coverage queries."""

    def __init__(self, duration: float = 0.0):
        self.duration = float(duration)
        self.observations: list[Observation] = []
        self.open_questions: list[str] = []
        self.frame_timestamps: list[float] = []

    # ----------------------------------------------------------------- writes
    def add(self, observation: Observation) -> Observation:
        self.observations.append(observation)
        self.frame_timestamps.extend(observation.timestamps)
        if observation.notes:
            logger.debug("observation %s: %s", observation.label, observation.notes)
        return observation

    def set_open_questions(self, questions: Iterable[str]) -> None:
        self.open_questions = [q.strip() for q in questions if q and q.strip()]

    def add_open_question(self, question: str) -> None:
        question = (question or "").strip()
        if question and question not in self.open_questions:
            self.open_questions.append(question)

    def resolve_open_question(self, question: str) -> None:
        self.open_questions = [q for q in self.open_questions if q != question]

    # ---------------------------------------------------------------- queries
    #: Methods that actually spent vision budget on a window.
    VISUAL_METHODS = frozenset({"frames", "scan"})

    def _intervals(self, *, visual_only: bool) -> list[tuple[float, float]]:
        return _merge_intervals(
            (o.start, o.end)
            for o in self.observations
            if not visual_only or o.method in self.VISUAL_METHODS
        )

    @property
    def checked_intervals(self) -> list[tuple[float, float]]:
        """Visually inspected intervals (the expensive work we must not repeat)."""
        return self._intervals(visual_only=True)

    @property
    def all_intervals(self) -> list[tuple[float, float]]:
        """Every interval touched, transcript reads included."""
        return self._intervals(visual_only=False)

    def new_ratio(self, start: float, end: float, *, visual_only: bool = True) -> float:
        """Fraction of ``[start, end]`` that has *not* been observed yet."""
        if end <= start:
            return 0.0
        span = end - start
        covered = 0.0
        for left, right in self._intervals(visual_only=visual_only):
            covered += max(0.0, min(right, end) - max(left, start))
        return max(0.0, min(1.0, 1.0 - covered / span))

    def is_covered(self, start: float, end: float, threshold: float = 0.6) -> bool:
        """True when most of the window has already been *looked at*.

        Only vision observations count: reading the transcript for a window does
        not mean the slides in it have been seen.
        """
        return self.new_ratio(start, end, visual_only=True) < (1.0 - threshold)

    def coverage_ratio(self) -> float:
        """Share of the whole video that has been visually inspected."""
        if self.duration <= 0:
            return 0.0
        covered = sum(right - left for left, right in self.checked_intervals)
        return max(0.0, min(1.0, covered / self.duration))

    def frames_viewed(self) -> int:
        return len(self.frame_timestamps)

    def unique_frames(self) -> int:
        return len({round(ts, 1) for ts in self.frame_timestamps})

    def vlm_looks(self) -> int:
        return sum(1 for o in self.observations if o.method in {"frames", "scan"})

    def best_confidence(self) -> float:
        return max((o.confidence for o in self.observations), default=0.0)

    def answered_by(self) -> list[Observation]:
        return [o for o in self.observations if o.answers_question]

    def evidence_count(self) -> int:
        return sum(len(o.evidence) for o in self.observations)

    def latest(self) -> Observation | None:
        return self.observations[-1] if self.observations else None

    def relevant_windows(self, min_relevance: float = 0.5) -> list[Observation]:
        return [o for o in self.observations if o.relevance >= min_relevance]

    # --------------------------------------------------------------- prompts
    def digest(self, *, max_items: int = 12) -> str:
        """Compact, prompt-ready summary of everything observed so far."""
        if not self.observations:
            return "(nothing inspected yet)"

        lines: list[str] = []
        for observation in self.observations[-max_items:]:
            parts = [f"- {observation.label} via {observation.method} (rel={observation.relevance:.2f})"]
            if observation.knowledge_points:
                parts.append("  topics: " + "; ".join(observation.knowledge_points[:4]))
            if observation.visual_text:
                parts.append("  on-screen: " + observation.visual_text[:220].replace("\n", " "))
            for evidence in observation.evidence[:3]:
                parts.append(f"  evidence: {evidence[:220]}")
            if observation.answers_question:
                parts.append("  -> judged sufficient to answer")
            lines.extend(parts)
        return "\n".join(lines)

    def coverage_digest(self) -> str:
        intervals = self.checked_intervals
        if not intervals:
            return "(no intervals inspected)"
        return ", ".join(
            f"{format_timestamp(left)}-{format_timestamp(right)}" for left, right in intervals
        )

    def open_questions_digest(self) -> str:
        if not self.open_questions:
            return "(none recorded)"
        return "\n".join(f"- {question}" for question in self.open_questions[:8])

    def stats(self) -> dict:
        return {
            "observations": len(self.observations),
            "vlm_looks": self.vlm_looks(),
            "frames_viewed": self.frames_viewed(),
            "unique_frames": self.unique_frames(),
            "coverage_ratio": round(self.coverage_ratio(), 4),
            "checked_intervals": len(self.checked_intervals),
            "evidence_items": self.evidence_count(),
            "best_confidence": round(self.best_confidence(), 3),
            "open_questions": len(self.open_questions),
        }

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "duration": self.duration,
            "observations": [o.as_dict() for o in self.observations],
            "open_questions": self.open_questions,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TemporalMemory":
        memory = cls(duration=float(payload.get("duration", 0.0)))
        for item in payload.get("observations", []):
            item = dict(item)
            item.pop("created_at", None)
            memory.observations.append(Observation(**item))
        memory.open_questions = list(payload.get("open_questions", []))
        memory.frame_timestamps = [
            ts for observation in memory.observations for ts in observation.timestamps
        ]
        return memory