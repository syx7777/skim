"""The SkimAgent loop.

The control flow mirrors the design in the project brief:

1. **Coarse localisation** -- transcript BM25 + chapters, no frames, no VLM.
2. **Agentic loop** -- the model picks the next action with native tool calling.
   Cheap tools first; ``inspect_video`` is the only one that spends VLM budget
   and it refuses windows Temporal Memory already covers.
3. **Adaptive observation** -- frame density scales with the relevance of the
   window instead of with pixel-level scene change.
4. **Uncertainty-based stopping** -- when the gathered evidence looks sufficient
   the agent is told so and asked to finish, but the decision stays with the
   model; the loop also stops on stagnation or budget exhaustion. A final
   synthesis step guarantees an answer exists even on a hard budget.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..indexing.chapters import Chapter, build_chapters
from ..indexing.coarse import CoarseResult, coarse_localize
from ..indexing.transcript_index import TranscriptIndex
from ..llm.client import LLMClient, UsageTracker
from ..logging_utils import get_logger
from ..media.frames import FrameStore
from ..media.ingest import VideoAsset
from ..media.transcript import format_timestamp
from ..memory.temporal_memory import Observation, TemporalMemory
from .observer import VisualObserver
from .tools import TOOL_SCHEMAS, ToolBox

logger = get_logger("agent")


@dataclass
class AgentConfig:
    """Knobs, including the three ablation switches."""

    max_steps: int = 8
    max_frames_per_call: int = 4
    min_confidence: float = 0.7
    patience: int = 2
    # --- ablation switches ---
    adaptive_observation: bool = True
    use_temporal_memory: bool = True
    use_uncertainty: bool = True
    # --- settings ---
    build_chapters: bool = True
    frame_max_side: int = 768
    jpeg_quality: int = 80
    reasoning: str = "off"

    @property
    def label(self) -> str:
        tags = []
        if not self.adaptive_observation:
            tags.append("no-adaptive")
        if not self.use_temporal_memory:
            tags.append("no-memory")
        if not self.use_uncertainty:
            tags.append("no-uncertainty")
        return "full" if not tags else "ablation:" + "+".join(tags)


@dataclass
class AgentResult:
    question: str
    answer: str
    confidence: float
    evidence: list[dict[str, Any]] = field(default_factory=list)
    remaining_uncertainty: str = ""
    used_timestamps: list[float] = field(default_factory=list)
    stop_reason: str = ""
    steps: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)
    coarse: dict[str, Any] = field(default_factory=dict)
    memory: TemporalMemory | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    config_label: str = "full"

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
            "remaining_uncertainty": self.remaining_uncertainty,
            "used_timestamps": [round(ts, 2) for ts in self.used_timestamps],
            "stop_reason": self.stop_reason,
            "steps": self.steps,
            "config": self.config_label,
            "coarse": self.coarse,
            "metrics": self.metrics,
            "trace": self.trace,
        }


SYSTEM_PROMPT = """You are Skim. You answer questions about a long lecture video by \
selectively looking at it, not by watching all of it.

Video: {name}
Duration: {duration}
Transcript: {segments} segments, language {language}

Available evidence channels:
- the spoken transcript (cheap, always available, time-stamped);
- the video frames (expensive: each look costs vision-model tokens), which carry
  slides, formulas, handwritten notes, code and diagrams that speech misses.

How to work:
1. Start from the coarse candidate windows already computed for you below.
2. `search_transcript` / `read_transcript` / `list_chapters` are free -- use them freely.
3. Call `inspect_video` only when the transcript is insufficient, when the question
   refers to visual material (slides / formulas / code / diagrams / handwriting), or
   when a window has suspiciously little speech. Prefer NARROW windows (20-90s).
4. Never re-inspect a window that is already covered -- the memory tool will refuse it.
5. Call `finish` as soon as the evidence is sufficient. Do not keep exploring.

Budget: at most {max_steps} steps; at most {max_frames} frames per `inspect_video` call.
Answer in the same language as the question."""


FIRST_USER = """Question: {question}

Coarse candidate windows derived from the transcript (cheap, may be wrong):
{candidates}

Table of contents:
{chapters}

{notes}
Decide the single most informative next action. Reply with one tool call."""


class SkimAgent:
    """Orchestrates indexing, the agent loop and the final synthesis."""

    def __init__(
        self,
        client: LLMClient,
        asset: VideoAsset,
        config: AgentConfig | None = None,
    ):
        self.client = client
        self.asset = asset
        self.config = config or AgentConfig()
        self.index = TranscriptIndex(asset.transcript)
        self.frame_store = FrameStore(
            asset.path,
            asset.cache_dir / "frames",
            max_side=self.config.frame_max_side,
            jpeg_quality=self.config.jpeg_quality,
        )
        self.observer = VisualObserver(
            client,
            self.frame_store,
            max_frames_per_call=self.config.max_frames_per_call,
            jpeg_quality=self.config.jpeg_quality,
        )
        self._chapters: list[Chapter] | None = None

    # -------------------------------------------------------------- chapters
    @property
    def chapters(self) -> list[Chapter]:
        if self._chapters is None:
            if self.config.build_chapters:
                self._chapters = build_chapters(self.asset, self.client)
            else:
                self._chapters = []
        return self._chapters

    # ------------------------------------------------------------------- run
    def run(self, question: str, *, progress=None) -> AgentResult:
        """Run one question end-to-end.

        A fresh :class:`UsageTracker` is installed for the duration of the run so
        that per-question metrics (frames, VLM calls, visual tokens) are exact.
        """
        started = time.perf_counter()
        self.client.usage = UsageTracker()
        return self._run_inner(question, started, progress)

    def _run_inner(self, question: str, started: float, progress) -> AgentResult:
        config = self.config
        memory = TemporalMemory(duration=self.asset.duration)

        def notify(message: str) -> None:
            logger.info(message)
            if progress is not None:
                progress(message)

        # --- stage 1: cheap coarse localisation -----------------------------
        notify("Coarse localisation over the transcript ...")
        coarse: CoarseResult = coarse_localize(
            question, self.index, self.chapters, max_windows=4
        )
        logger.info(
            "coarse: %d candidate windows, needs_vision=%s, terms=%s",
            len(coarse.windows),
            coarse.needs_vision,
            coarse.terms[:6],
        )

        if config.use_temporal_memory:
            # remember that the cheap pass already "read" those windows
            for window in coarse.windows[:2]:
                memory.add(
                    Observation(
                        start=window.start,
                        end=window.end,
                        method="transcript",
                        relevance=window.relevance,
                        confidence=0.25,
                        notes=f"transcript hit: {window.reason}",
                        evidence=window.snippets[:2],
                        step=0,
                    )
                )

        toolbox = ToolBox(
            question=question,
            index=self.index,
            chapters=self.chapters,
            observer=self.observer,
            memory=memory,
            frame_store=self.frame_store,
            use_memory=config.use_temporal_memory,
            adaptive=config.adaptive_observation,
            max_frames_per_call=config.max_frames_per_call,
        )

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    name=self.asset.path.name,
                    duration=f"{self.asset.duration:.0f}s ({format_timestamp(self.asset.duration)})",
                    segments=len(self.asset.transcript),
                    language=self.asset.transcript.language,
                    max_steps=config.max_steps,
                    max_frames=config.max_frames_per_call,
                ),
            },
            {
                "role": "user",
                "content": FIRST_USER.format(
                    question=question,
                    candidates=self._render_candidates(coarse),
                    chapters=self._render_chapters(),
                    notes=("\n".join(f"note: {n}" for n in coarse.notes) + "\n") if coarse.notes else "",
                ),
            },
        ]

        trace: list[dict[str, Any]] = []
        stop_reason = "budget_exhausted"
        steps_taken = 0
        last_evidence = memory.evidence_count()
        last_useful = toolbox.useful_calls
        stagnant = 0
        nudged = False

        for step in range(1, config.max_steps + 1):
            steps_taken = step
            toolbox.step = step
            notify(f"Agent step {step}/{config.max_steps}")

            response = self.client.chat(
                messages,
                tools=TOOL_SCHEMAS,
                max_tokens=4096,
                reasoning=config.reasoning,
            )
            messages.append(response.to_message())

            if not response.tool_calls:
                # Nudge once instead of silently ending the loop.
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Respond with a tool call. If the gathered evidence is already "
                            "sufficient, call `finish`."
                        ),
                    }
                )
                trace.append({"step": step, "action": "no_tool_call", "text": response.text[:400]})
                stagnant += 1
                if stagnant >= config.patience:
                    stop_reason = "no_progress"
                    break
                continue

            finished = False
            for call in response.tool_calls:
                # Measure how much of the requested window was still unexplored
                # *before* the call, otherwise the observation is already in
                # memory and the ratio would always read zero.
                pre_new_ratio: float | None = None
                if call.name == "inspect_video":
                    try:
                        pre_new_ratio = round(
                            memory.new_ratio(
                                float(call.arguments.get("start", 0.0)),
                                float(call.arguments.get("end", 0.0)),
                            ),
                            3,
                        )
                    except (TypeError, ValueError):
                        pre_new_ratio = None

                result = toolbox.dispatch(call.name, call.arguments)
                messages.append(result.as_message(call.id))
                trace.append(
                    {
                        "step": step,
                        "action": call.name,
                        "arguments": call.arguments,
                        "cost": result.cost,
                        "summary": result.content[:400],
                    }
                )
                if call.name == "inspect_video" and not result.payload.get("refused"):
                    observation = result.payload.get("observation") or {}
                    trace[-1]["relevance"] = observation.get("relevance")
                    trace[-1]["confidence"] = observation.get("confidence")
                    trace[-1]["frames"] = result.payload.get("frames")
                    trace[-1]["new_ratio"] = pre_new_ratio
                if call.name == "finish":
                    finished = True

            if finished:
                stop_reason = "finish"
                break

            # --- stopping policy -------------------------------------------
            # "Progress" means either new vision evidence or new material from a
            # cheap tool (a transcript window the agent had not read yet).
            evidence_now = memory.evidence_count()
            useful_now = toolbox.useful_calls
            progressed = evidence_now > last_evidence or useful_now > last_useful
            stagnant = 0 if progressed else stagnant + 1
            last_evidence = evidence_now
            last_useful = useful_now

            if config.use_uncertainty:
                # A single window reporting "I can answer this" is a *local*
                # judgement, not a global one, so it is offered to the agent as
                # a bounded hint rather than used to override its decision.
                # Without this switch the agent simply keeps working until the
                # budget or the stagnation guard stops it.
                confident = [o for o in memory.answered_by() if o.confidence >= config.min_confidence]
                if confident and not memory.open_questions and not nudged:
                    nudged = True
                    best = max(confident, key=lambda o: o.confidence)
                    hint = (
                        f"Evidence gathered at {best.label} looks sufficient "
                        f"(relevance {best.relevance:.2f}, confidence {best.confidence:.2f}) and no "
                        "open questions remain. If you agree, call `finish` now with the answer and "
                        "its evidence. If you believe something is still missing, continue searching "
                        "instead."
                    )
                    messages.append({"role": "user", "content": hint})
                    trace.append({"step": step, "action": "stop_hint", "summary": hint})
                    continue
                if stagnant >= config.patience:
                    stop_reason = "no_progress"
                    break
            elif stagnant >= config.patience + 2:
                # the non-uncertainty ablation still needs *some* guard against
                # infinite loops, but it is deliberately more tolerant
                stop_reason = "no_progress"
                break

        # --- stage 3: always produce an answer ------------------------------
        if toolbox.final is not None:
            answer = toolbox.final["answer"]
            confidence = toolbox.final["confidence"]
            evidence = toolbox.final["evidence"]
            remaining = toolbox.final.get("remaining_uncertainty", "")
        else:
            notify("Synthesising the final answer from gathered evidence ...")
            synthesised = self._synthesise(question, coarse, memory, toolbox.transcript_windows)
            answer = synthesised.get("answer", "")
            confidence = float(synthesised.get("confidence", 0.0) or 0.0)
            evidence = synthesised.get("evidence", []) or []
            remaining = synthesised.get("remaining_uncertainty", "")

        latency = time.perf_counter() - started
        metrics = {
            **self.client.usage.summary(),
            "memory_stats": memory.stats(),
            "coverage_ratio": round(memory.coverage_ratio(), 4),
            "video_duration_s": round(self.asset.duration, 2),
            "wall_latency_s": round(latency, 2),
            "steps": steps_taken,
            "stop_reason": stop_reason,
        }

        used_timestamps = [round(ts, 1) for ts in memory.frame_timestamps]
        result = AgentResult(
            question=question,
            answer=answer,
            confidence=confidence,
            evidence=evidence,
            remaining_uncertainty=remaining,
            used_timestamps=used_timestamps,
            stop_reason=stop_reason,
            steps=steps_taken,
            trace=trace,
            coarse=coarse.as_dict(),
            memory=memory,
            metrics=metrics,
            config_label=config.label,
        )
        return result

    # ---------------------------------------------------------------- helpers
    def _render_candidates(self, coarse: CoarseResult) -> str:
        if not coarse.windows:
            return "(none -- the transcript gave no usable signal)"
        lines = []
        for window in coarse.windows:
            lines.append(
                f"- {format_timestamp(window.start)}-{format_timestamp(window.end)} "
                f"(relevance {window.relevance:.2f}): {window.reason}"
            )
            if window.snippets:
                lines.append(f"    e.g. \"{window.snippets[0][:160]}\"")
        return "\n".join(lines)

    def _render_chapters(self) -> str:
        if not self.chapters:
            return "(not available)"
        return "\n".join(chapter.as_prompt_line() for chapter in self.chapters[:20])

    def _synthesise(
        self,
        question: str,
        coarse: CoarseResult,
        memory: TemporalMemory,
        read_windows: list[tuple[float, float]] | None = None,
    ) -> dict[str, Any]:
        """Final answer, grounded in everything gathered so far."""
        # Prefer windows the agent actually read over the cheap coarse guesses.
        windows: list[tuple[float, float]] = list(read_windows or [])
        for window in coarse.windows[:3]:
            if not any(
                abs(window.start - existing[0]) < 5 and abs(window.end - existing[1]) < 5
                for existing in windows
            ):
                windows.append((window.start, window.end))

        transcript_excerpts: list[str] = []
        for start, end in windows[:5]:
            text = self.asset.transcript.text_between(start, end)
            if text:
                transcript_excerpts.append(
                    f"[{format_timestamp(start)}-{format_timestamp(end)}] {text[:900]}"
                )

        prompt = f"""Answer the question using ONLY the evidence below. If the evidence is
insufficient, say so explicitly and give the best partial answer.

Question: {question}

What was inspected and found:
{memory.digest()}

Relevant transcript excerpts:
{chr(10).join(transcript_excerpts) or '(none)'}

Return JSON:
{{
  "answer": "...",
  "confidence": 0.0,
  "evidence": [{{"timestamp": 0.0, "quote": "..."}}],
  "remaining_uncertainty": ""
}}
Answer in the same language as the question. Cite the timestamps you rely on."""

        try:
            payload = self.client.chat_json(
                [
                    {"role": "system", "content": "You answer questions about lecture videos with JSON only."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4096,
                reasoning=self.config.reasoning,
            )
        except Exception as exc:
            logger.warning("synthesis failed: %s", exc)
            return {
                "answer": "Unable to produce a grounded answer within the budget.",
                "confidence": 0.0,
                "evidence": [],
                "remaining_uncertainty": str(exc),
            }

        evidence: list[dict[str, Any]] = []
        for item in payload.get("evidence") or []:
            if isinstance(item, dict):
                try:
                    timestamp = float(item.get("timestamp", 0.0))
                except (TypeError, ValueError):
                    timestamp = 0.0
                evidence.append({"timestamp": timestamp, "quote": str(item.get("quote", "")).strip()})
            elif isinstance(item, str):
                evidence.append({"timestamp": 0.0, "quote": item})

        try:
            confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0

        return {
            "answer": str(payload.get("answer", "")).strip(),
            "confidence": confidence,
            "evidence": evidence,
            "remaining_uncertainty": str(payload.get("remaining_uncertainty", "")).strip(),
        }

    # ------------------------------------------------------------- utilities
    def question_timeline(self, question: str) -> dict[str, Any]:
        """Cheap preview used by the UI before running the agent."""
        coarse = coarse_localize(question, self.index, self.chapters, max_windows=4)
        return {
            "question": question,
            "candidates": coarse.as_dict(),
            "chapters": [chapter.as_dict() for chapter in self.chapters],
        }

    def to_json(self, result: AgentResult) -> str:
        return json.dumps(result.as_dict(), ensure_ascii=False, indent=2)