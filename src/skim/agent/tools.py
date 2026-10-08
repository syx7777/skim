"""Tool surface exposed to the agent LLM.

Cheap first, expensive last:

``search_transcript`` / ``read_transcript`` / ``list_chapters`` cost nothing but
tokens, while ``inspect_video`` is the only tool that spends VLM budget -- and it
consults Temporal Memory before doing so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..indexing.chapters import Chapter
from ..indexing.transcript_index import TranscriptIndex
from ..logging_utils import get_logger
from ..media.frames import FrameStore
from ..media.transcript import format_timestamp
from ..memory.temporal_memory import Observation, TemporalMemory
from .observer import VisualObserver

logger = get_logger("tools")


@dataclass
class ToolResult:
    name: str
    content: str
    payload: dict[str, Any] = field(default_factory=dict)
    cost: str = "cheap"  # cheap | vlm

    def as_message(self, tool_call_id: str) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": self.content}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_transcript",
            "description": (
                "Keyword/BM25 search over the lecture transcript. Free. Use it to "
                "locate the rough time range of a topic before looking at video."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search terms."},
                    "top_k": {"type": "integer", "description": "Max results (default 8)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_transcript",
            "description": "Read the verbatim transcript inside a time window. Free.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "number", "description": "Start time in seconds."},
                    "end": {"type": "number", "description": "End time in seconds."},
                },
                "required": ["start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_chapters",
            "description": "List the automatically derived table of contents with time ranges. Free.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_video",
            "description": (
                "EXPENSIVE: sample frames from a time window and ask the vision model what is "
                "shown (slides, formulas, handwritten notes, code). Only call this when the "
                "transcript is not enough, and prefer narrow windows. Re-inspecting a window "
                "that is already covered is refused unless force=true."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "number", "description": "Window start in seconds."},
                    "end": {"type": "number", "description": "Window end in seconds."},
                    "num_frames": {
                        "type": "integer",
                        "description": "Frames to sample (1-4). More frames = more cost.",
                    },
                    "focus": {
                        "type": "string",
                        "description": "What to look for, e.g. 'the GRPO objective formula'.",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Re-inspect even if this window is already covered.",
                    },
                },
                "required": ["start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_status",
            "description": "Show which time ranges were already inspected, the evidence found and open questions. Free.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Stop searching and deliver the final answer. Call this as soon as the evidence "
                "is sufficient, and cite the timestamps you rely on."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "The answer, in the user's language."},
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "timestamp": {"type": "number"},
                                "quote": {"type": "string"},
                            },
                            "required": ["timestamp", "quote"],
                        },
                        "description": "Supporting items with their timestamps.",
                    },
                    "confidence": {"type": "number", "description": "0-1 confidence in the answer."},
                    "remaining_uncertainty": {"type": "string"},
                },
                "required": ["answer", "confidence"],
            },
        },
    },
]


class ToolBox:
    """Executes tool calls against the video the agent is reasoning about."""

    def __init__(
        self,
        *,
        question: str,
        index: TranscriptIndex,
        chapters: list[Chapter],
        observer: VisualObserver,
        memory: TemporalMemory,
        frame_store: FrameStore,
        use_memory: bool = True,
        adaptive: bool = True,
        max_frames_per_call: int = 4,
        step: int = 0,
    ):
        self.question = question
        self.index = index
        self.chapters = chapters
        self.observer = observer
        self.memory = memory
        self.store = frame_store
        self.use_memory = use_memory
        self.adaptive = adaptive
        self.max_frames_per_call = max_frames_per_call
        self.step = step
        self.final: dict[str, Any] | None = None
        self.calls: list[dict[str, Any]] = []
        #: Tool calls that actually returned new material. The agent loop uses
        #: this (not just VLM evidence) as its progress signal -- otherwise a
        #: transcript-only investigation looks like "no progress" and the
        #: stagnation guard stops the run before it has read anything.
        self.useful_calls = 0
        #: Windows the agent explicitly read, so the final synthesis can quote
        #: them even when they were not in the cheap coarse candidate list.
        self.transcript_windows: list[tuple[float, float]] = []

    # -------------------------------------------------------------- dispatch
    def dispatch(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return ToolResult(name=name, content=f"ERROR: unknown tool '{name}'")
        try:
            result = handler(**arguments)
        except TypeError as exc:
            result = ToolResult(name=name, content=f"ERROR: bad arguments for {name}: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("tool %s failed", name)
            result = ToolResult(name=name, content=f"ERROR: {name} failed: {exc}")
        self.calls.append({"name": name, "cost": result.cost, "arguments": arguments})
        if self._is_useful(result):
            self.useful_calls += 1
        return result

    @staticmethod
    def _is_useful(result: ToolResult) -> bool:
        """Did this call return material the agent did not already have?"""
        if result.payload.get("refused"):
            return False
        if result.name == "search_transcript":
            return bool(result.payload.get("hits"))
        if result.name == "read_transcript":
            return bool(result.payload.get("segments"))
        if result.name == "list_chapters":
            return bool(result.payload.get("chapters"))
        if result.name == "inspect_video":
            observation = result.payload.get("observation") or {}
            return bool(observation.get("evidence")) or float(observation.get("relevance") or 0) > 0.2
        # memory_status and finish are bookkeeping, not evidence gathering.
        return False

    # ----------------------------------------------------------------- tools
    def _tool_search_transcript(self, query: str, top_k: int = 8) -> ToolResult:
        hits = self.index.search(query, top_k=int(top_k))
        if not hits:
            return ToolResult(
                name="search_transcript",
                content=f"No transcript matches for {query!r}.",
                payload={"hits": []},
            )
        lines = [
            f"{format_timestamp(hit.start)}-{format_timestamp(hit.end)} (score {hit.score:.2f}): {hit.segment.text}"
            for hit in hits
        ]
        return ToolResult(
            name="search_transcript",
            content="Transcript matches for "
            f"{query!r}:\n"
            + "\n".join(lines[: int(top_k)]),
            payload={"hits": [hit.as_dict() for hit in hits]},
        )

    def _tool_read_transcript(self, start: float, end: float) -> ToolResult:
        start, end = float(start), float(end)
        segments = self.index.transcript.window(start, end)
        if not segments:
            return ToolResult(
                name="read_transcript",
                content=(
                    f"No speech between {format_timestamp(start)} and {format_timestamp(end)} "
                    "-- this window may contain visual-only content."
                ),
                payload={"segments": []},
            )
        self.transcript_windows.append((start, end))
        text = "\n".join(segment.as_prompt_line() for segment in segments)
        return ToolResult(
            name="read_transcript",
            content=f"Transcript {format_timestamp(start)}-{format_timestamp(end)}:\n{text}",
            payload={"segments": [segment.to_dict() for segment in segments]},
        )

    def _tool_list_chapters(self) -> ToolResult:
        if not self.chapters:
            return ToolResult(name="list_chapters", content="No chapters available.")
        lines = [
            f"{index}. [{format_timestamp(c.start)}-{format_timestamp(c.end)}] {c.title}"
            + (f" -- {c.summary}" if c.summary else "")
            for index, c in enumerate(self.chapters, start=1)
        ]
        return ToolResult(
            name="list_chapters",
            content="Table of contents:\n" + "\n".join(lines),
            payload={"chapters": [c.as_dict() for c in self.chapters]},
        )

    def _tool_memory_status(self) -> ToolResult:
        stats = self.memory.stats()
        content = (
            "Inspected intervals: "
            + self.memory.coverage_digest()
            + f"\nCoverage: {stats['coverage_ratio'] * 100:.1f}% of the video, "
            + f"{stats['frames_viewed']} frames, {stats['vlm_looks']} vision calls.\n"
            + "Findings so far:\n"
            + self.memory.digest()
            + "\nOpen questions:\n"
            + self.memory.open_questions_digest()
        )
        return ToolResult(name="memory_status", content=content, payload=stats)

    def _tool_inspect_video(
        self,
        start: float,
        end: float,
        num_frames: int | None = None,
        focus: str | None = None,
        force: bool = False,
    ) -> ToolResult:
        start, end = float(start), max(float(end), float(start) + 1.0)
        if self.store.duration:
            end = min(end, self.store.duration)

        if self.use_memory and not force and self.memory.is_covered(start, end):
            return ToolResult(
                name="inspect_video",
                content=(
                    f"REFUSED: {format_timestamp(start)}-{format_timestamp(end)} is already covered "
                    f"({self.memory.new_ratio(start, end) * 100:.0f}% unexplored). "
                    "Existing findings:\n"
                    + self.memory.digest(max_items=4)
                    + "\nPick an unexplored window or call finish if the evidence is sufficient."
                ),
                payload={"refused": True, "reason": "already_covered"},
            )

        relevance_hint = 0.5
        budget = int(num_frames) if num_frames else None
        if budget is not None:
            # even in non-adaptive mode the caller's explicit request is honoured,
            # but it is clamped to the configured per-call ceiling
            budget = max(1, min(budget, self.max_frames_per_call))
        elif not self.adaptive:
            budget = min(2, self.max_frames_per_call)

        outcome = self.observer.inspect(
            self.question,
            start,
            end,
            count=budget,
            relevance_hint=relevance_hint,
            focus=focus,
            context=self.memory.digest(max_items=5) if self.use_memory else "(not tracking history)",
            adaptive=self.adaptive,
            step=self.step,
        )
        self.memory.add(outcome.observation)

        observation = outcome.observation
        if self.use_memory and observation.notes:
            # remember what is still unknown so later steps can chase it
            missing = outcome.payload.get("still_missing")
            if isinstance(missing, str) and missing.strip():
                self.memory.add_open_question(missing.strip())

        content = (
            f"Vision result for {format_timestamp(start)}-{format_timestamp(end)} "
            f"({outcome.frames} frames, relevance {observation.relevance:.2f}, "
            f"confidence {observation.confidence:.2f}):\n"
            f"summary: {observation.notes or '(none)'}\n"
            f"on_screen_text: {observation.visual_text or '(none)'}\n"
            f"knowledge_points: {', '.join(observation.knowledge_points) or '(none)'}\n"
            "evidence:\n"
            + ("\n".join(f"  - {item}" for item in observation.evidence) or "  (none)")
            + (
                f"\nsuggested next window: {outcome.suggest_next}"
                if outcome.suggest_next
                else ""
            )
        )
        return ToolResult(
            name="inspect_video",
            content=content,
            payload={
                "observation": observation.as_dict(),
                "suggest_next": outcome.suggest_next,
                "frames": outcome.frames,
            },
            cost="vlm",
        )

    def _tool_finish(
        self,
        answer: str,
        confidence: float = 0.0,
        evidence: list[dict[str, Any]] | None = None,
        remaining_uncertainty: str = "",
    ) -> ToolResult:
        items: list[dict[str, Any]] = []
        for item in evidence or []:
            if isinstance(item, dict):
                try:
                    timestamp = float(item.get("timestamp", 0.0))
                except (TypeError, ValueError):
                    timestamp = 0.0
                items.append({"timestamp": timestamp, "quote": str(item.get("quote", "")).strip()})
            elif isinstance(item, str):
                items.append({"timestamp": 0.0, "quote": item})
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.0

        self.final = {
            "answer": answer.strip(),
            "confidence": confidence,
            "evidence": items,
            "remaining_uncertainty": remaining_uncertainty.strip(),
        }
        return ToolResult(
            name="finish",
            content="Search finished. Final answer recorded.",
            payload=self.final,
        )

    # ------------------------------------------------------------------ misc
    def seed_memory_from_transcript(self, start: float, end: float, note: str) -> None:
        """Record a cheap transcript-only look so the agent knows it happened."""
        segments = self.index.transcript.window(start, end)
        self.memory.add(
            Observation(
                start=start,
                end=end,
                method="transcript",
                knowledge_points=[],
                evidence=[segment.text for segment in segments[:4]],
                relevance=0.4,
                confidence=0.2,
                notes=note,
                step=self.step,
            )
        )

    def as_json(self) -> str:
        return json.dumps(self.calls, ensure_ascii=False)