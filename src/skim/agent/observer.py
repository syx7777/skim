"""The VLM observer: turns a candidate time window into structured evidence.

This is the *expensive* half of the system. Everything upstream (transcript
search, chapters, coarse localisation) exists to make sure these calls land on
windows that actually matter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..llm.client import LLMClient
from ..logging_utils import get_logger
from ..media.frames import FrameStore, dense_factor, plan_timestamps
from ..media.transcript import format_timestamp
from ..memory.temporal_memory import Observation

logger = get_logger("observer")

SYSTEM_PROMPT = """You are the visual observer of a lecture-video question answering agent.
You are shown a few frames sampled from one time window of a course video.
Read the frames carefully: slides, formulas, handwritten notes, code and diagrams
all carry information that the spoken transcript may miss.

Rules:
- Transcribe on-screen text VERBATIM (including formulas and code) in "on_screen_text".
- Only report what is actually visible. Never invent slide content.
- Judge relevance to the user's question, not general interestingness.
- Answer in the same language as the user's question; keep technical terms,
  acronyms and symbols exactly as they appear.
- If the frames are irrelevant, say so with a low relevance score and empty evidence.
- Respond with JSON only."""

USER_PROMPT = """User question:
{question}

Now inspecting window {start} - {end} ({span:.0f}s), frames sampled at:
{frame_list}

Context already gathered (may be incomplete):
{context}

Return JSON with exactly these keys:
{{
  "relevance": 0.0,               // how relevant these frames are to the question, 0-1
  "answers_question": false,      // true only if the frames already contain enough to answer
  "confidence": 0.0,              // your confidence in that judgement, 0-1
  "summary": "",                  // one or two sentences describing what is shown
  "on_screen_text": "",           // verbatim text/formulas/code visible in the frames
  "knowledge_points": [],         // short topical labels for what is taught here
  "evidence": [                   // concrete supporting items, may be empty
    {{"timestamp": 0.0, "quote": ""}}
  ],
  "still_missing": "",            // what is still needed to fully answer, if anything
  "suggest_next": null            // or {{"start": 0.0, "end": 0.0, "reason": ""}}
}}
Use the real timestamps (seconds) of the frames you are citing in "evidence"."""  # noqa: E501


@dataclass
class InspectionOutcome:
    """Result of one VLM look, before it becomes a memory Observation."""

    observation: Observation
    payload: dict[str, Any]
    frames: int

    @property
    def suggest_next(self) -> dict[str, Any] | None:
        suggestion = self.payload.get("suggest_next")
        return suggestion if isinstance(suggestion, dict) else None


class VisualObserver:
    """Owns frame sampling + the VLM call for a single window."""

    def __init__(
        self,
        client: LLMClient,
        frame_store: FrameStore,
        *,
        max_frames_per_call: int = 4,
        jpeg_quality: int = 80,
    ):
        self.client = client
        self.store = frame_store
        self.max_frames_per_call = max_frames_per_call
        self.jpeg_quality = jpeg_quality

    # ------------------------------------------------------------------ public
    def frame_budget(self, relevance: float, *, adaptive: bool = True) -> int:
        """How many frames to spend on a window.

        Adaptive mode scales the budget with the estimated relevance of the
        window; the non-adaptive ablation always uses the same small number,
        which is what makes it a fair "fixed density" comparison.
        """
        if not adaptive:
            return min(2, self.max_frames_per_call)
        return min(dense_factor(relevance, base=2, maximum=self.max_frames_per_call),
                   self.max_frames_per_call)

    def inspect(
        self,
        question: str,
        start: float,
        end: float,
        *,
        count: int | None = None,
        relevance_hint: float = 0.5,
        focus: str | None = None,
        context: str = "(nothing inspected yet)",
        adaptive: bool = True,
        step: int = 0,
    ) -> InspectionOutcome:
        """Sample frames in ``[start, end]`` and ask the VLM about them."""
        budget = count or self.frame_budget(relevance_hint, adaptive=adaptive)
        timestamps = plan_timestamps(
            start, end, budget, max_count=self.max_frames_per_call, duration=self.store.duration
        )
        sampled = self.store.get_many(timestamps)
        if not sampled:
            logger.warning("no frames decoded for %.1f-%.1fs", start, end)
            return InspectionOutcome(
                observation=Observation(
                    start=start,
                    end=end,
                    method="frames",
                    relevance=0.0,
                    confidence=0.0,
                    notes="frame extraction failed",
                    step=step,
                ),
                payload={},
                frames=0,
            )

        attachments = [frame.to_attachment(self.jpeg_quality) for frame in sampled]
        frame_list = ", ".join(
            f"{format_timestamp(frame.timestamp)} (t={frame.timestamp:.1f}s)" for frame in sampled
        )
        prompt = USER_PROMPT.format(
            question=question,
            start=format_timestamp(start),
            end=format_timestamp(end),
            span=max(0.0, end - start),
            frame_list=frame_list,
            context=context,
        )
        if focus:
            prompt += f"\n\nFocus especially on: {focus}"

        try:
            payload = self.client.vision_json(
                prompt, attachments, system=SYSTEM_PROMPT, max_tokens=4096, reasoning="off"
            )
        except Exception as exc:
            logger.warning("VLM inspection failed for %.1f-%.1fs: %s", start, end, exc)
            payload = {"relevance": 0.0, "confidence": 0.0, "summary": f"VLM error: {exc}"}

        observation = self._to_observation(payload, start, end, [f.timestamp for f in sampled], step)
        return InspectionOutcome(observation=observation, payload=payload, frames=len(sampled))

    def scan(
        self,
        question: str,
        timestamps: list[float],
        *,
        window: float = 30.0,
        step: int = 0,
    ) -> list[InspectionOutcome]:
        """Low-frequency visual scan used when the transcript gives no signal."""
        outcomes: list[InspectionOutcome] = []
        for timestamp in timestamps:
            outcome = self.inspect(
                question,
                timestamp,
                min(timestamp + window, self.store.duration or timestamp + window),
                count=1,
                relevance_hint=0.5,
                context="(coarse visual scan of the whole video)",
                adaptive=False,
                step=step,
            )
            outcome.observation.method = "scan"
            outcomes.append(outcome)
        return outcomes

    # ----------------------------------------------------------------- private
    @staticmethod
    def _to_observation(
        payload: dict[str, Any],
        start: float,
        end: float,
        timestamps: list[float],
        step: int,
    ) -> Observation:
        def as_float(key: str, default: float = 0.0) -> float:
            try:
                return max(0.0, min(1.0, float(payload.get(key, default))))
            except (TypeError, ValueError):
                return default

        evidence: list[str] = []
        for item in payload.get("evidence") or []:
            if isinstance(item, dict):
                quote = str(item.get("quote", "")).strip()
                if not quote:
                    continue
                try:
                    stamp = float(item.get("timestamp", start))
                except (TypeError, ValueError):
                    stamp = start
                evidence.append(f"[{format_timestamp(stamp)}] {quote}")
            elif isinstance(item, str) and item.strip():
                evidence.append(item.strip())

        knowledge = [str(k).strip() for k in (payload.get("knowledge_points") or []) if str(k).strip()]

        return Observation(
            start=start,
            end=end,
            method="frames",
            timestamps=list(timestamps),
            frames=len(timestamps),
            knowledge_points=knowledge[:8],
            evidence=evidence[:6],
            visual_text=str(payload.get("on_screen_text", "") or "").strip()[:1200],
            relevance=as_float("relevance"),
            answers_question=bool(payload.get("answers_question", False)),
            confidence=as_float("confidence"),
            notes=str(payload.get("summary") or payload.get("still_missing") or "").strip()[:600],
            step=step,
        )