"""Reference methods that the agent is compared against.

* ``subtitle_rag``  -- transcript RAG only, never looks at the video.
* ``fixed_fps``     -- uniform frame sampling over the whole video, then one
  VLM pass per batch of frames. This is the "watch everything at a fixed rate"
  strategy the agent is meant to beat on cost.
"""

from __future__ import annotations

import time
from typing import Any

from ..indexing.transcript_index import TranscriptIndex
from ..llm.client import LLMClient, UsageTracker
from ..logging_utils import get_logger
from ..media.ingest import VideoAsset
from ..media.transcript import format_timestamp

logger = get_logger("baselines")


def _blank_metrics(usage: UsageTracker, *, duration: float, latency: float, coverage: float) -> dict[str, Any]:
    return {
        **usage.summary(),
        "coverage_ratio": round(coverage, 4),
        "video_duration_s": round(duration, 2),
        "wall_latency_s": round(latency, 2),
        "steps": 1,
    }


def run_subtitle_rag(
    asset: VideoAsset,
    client: LLMClient,
    question: str,
    *,
    top_k: int = 8,
) -> dict[str, Any]:
    """Answer from transcript retrieval only -- zero frames, zero visual tokens."""
    started = time.perf_counter()
    client.usage = UsageTracker()

    index = TranscriptIndex(asset.transcript)
    hits = index.search(question, top_k=top_k)
    context = "\n".join(
        f"[{format_timestamp(hit.start)}-{format_timestamp(hit.end)}] {hit.segment.text}" for hit in hits
    ) or "(no transcript matches)"

    payload = client.chat_json(
        [
            {
                "role": "system",
                "content": "You answer questions about lecture videos using only transcript excerpts. JSON only.",
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\nTranscript excerpts:\n{context}\n\n"
                    'Return JSON: {"answer": "...", "confidence": 0.0, '
                    '"evidence": [{"timestamp": 0.0, "quote": ""}], "remaining_uncertainty": ""}'
                ),
            },
        ],
        max_tokens=2048,
        reasoning="off",
    )

    latency = time.perf_counter() - started
    evidence = payload.get("evidence") or []
    timestamps = [
        float(item.get("timestamp", 0.0))
        for item in evidence
        if isinstance(item, dict) and item.get("timestamp") is not None
    ]
    if not timestamps:
        timestamps = [hit.start for hit in hits[:3]]

    return {
        "question": question,
        "answer": str(payload.get("answer", "")).strip(),
        "confidence": float(payload.get("confidence", 0.0) or 0.0),
        "evidence": evidence,
        "remaining_uncertainty": str(payload.get("remaining_uncertainty", "")),
        "used_timestamps": timestamps,
        "stop_reason": "single_pass",
        "steps": 1,
        "coarse": {"method": "subtitle_rag", "hits": [hit.as_dict() for hit in hits]},
        "metrics": _blank_metrics(client.usage, duration=asset.duration, latency=latency, coverage=0.0),
        "trace": [{"step": 1, "action": "transcript_rag", "summary": context[:400]}],
        "config": "baseline:subtitle_rag",
    }


def run_fixed_fps(
    asset: VideoAsset,
    client: LLMClient,
    question: str,
    *,
    interval_s: float = 30.0,
    batch_size: int = 4,
    max_frames: int = 60,
) -> dict[str, Any]:
    """Uniform temporal sampling + VLM, with no adaptivity and no memory."""
    started = time.perf_counter()
    client.usage = UsageTracker()

    from ..media.frames import FrameStore

    store = FrameStore(asset.path, asset.cache_dir / "frames")
    duration = asset.duration
    count = min(max_frames, max(1, int(duration // max(1.0, interval_s)) + 1))
    timestamps = [round(index * interval_s, 1) for index in range(count)]
    timestamps = [ts for ts in timestamps if 0 <= ts < max(duration, interval_s)]

    notes: list[str] = []
    for batch_start in range(0, len(timestamps), batch_size):
        batch = timestamps[batch_start : batch_start + batch_size]
        frames = store.get_many(batch)
        if not frames:
            continue
        attachments = [frame.to_attachment() for frame in frames]
        listing = ", ".join(f"t={frame.timestamp:.1f}s" for frame in frames)
        try:
            payload = client.vision_json(
                (
                    f"Question about this lecture video: {question}\n\n"
                    f"These frames come from a uniform sampling pass ({listing}).\n"
                    "Report anything relevant to the question; ignore irrelevant frames.\n"
                    'Return JSON: {"relevant": bool, "on_screen_text": "", "notes": "", '
                    '"evidence": [{"timestamp": 0.0, "quote": ""}]}'
                ),
                attachments,
                system="You inspect lecture-video frames. JSON only.",
                max_tokens=2048,
                reasoning="off",
            )
            if payload.get("relevant"):
                notes.append(
                    f"[{listing}] {payload.get('on_screen_text', '')} {payload.get('notes', '')}".strip()
                )
            for item in payload.get("evidence") or []:
                if isinstance(item, dict) and item.get("quote"):
                    notes.append(f"[t={item.get('timestamp')}] {item['quote']}")
        except Exception as exc:  # pragma: no cover - network path
            logger.warning("fixed_fps VLM batch failed: %s", exc)

    coverage = min(1.0, (len(timestamps) * 1.0) / max(1.0, duration / interval_s))

    payload = client.chat_json(
        [
            {
                "role": "system",
                "content": "You answer questions about lecture videos. JSON only.",
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Frames were sampled every {interval_s:.0f}s across the video. Notes:\n"
                    + ("\n".join(notes[:120]) or "(nothing relevant was found)")
                    + '\n\nReturn JSON: {"answer": "...", "confidence": 0.0, '
                    '"evidence": [{"timestamp": 0.0, "quote": ""}], "remaining_uncertainty": ""}'
                ),
            },
        ],
        max_tokens=3072,
        reasoning="off",
    )

    latency = time.perf_counter() - started
    evidence = payload.get("evidence") or []
    used = [float(item.get("timestamp", 0.0)) for item in evidence if isinstance(item, dict)]
    if not used:
        used = timestamps

    return {
        "question": question,
        "answer": str(payload.get("answer", "")).strip(),
        "confidence": float(payload.get("confidence", 0.0) or 0.0),
        "evidence": evidence,
        "remaining_uncertainty": str(payload.get("remaining_uncertainty", "")),
        "used_timestamps": used,
        "stop_reason": "single_pass",
        "steps": 1,
        "coarse": {"method": "fixed_fps", "interval_s": interval_s, "sampled": len(timestamps)},
        "metrics": _blank_metrics(client.usage, duration=duration, latency=latency, coverage=coverage),
        "trace": [{"step": 1, "action": "fixed_fps", "summary": f"{len(timestamps)} frames sampled"}],
        "config": "baseline:fixed_fps",
    }


BASELINES = {
    "subtitle_rag": run_subtitle_rag,
    "fixed_fps": run_fixed_fps,
}