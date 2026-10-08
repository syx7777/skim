"""Chapter / topic segmentation of a lecture, produced by the LLM.

Chapters are the second cheap signal used for coarse localisation: a question
like "where does he compare GRPO and PPO?" can often be answered at
*chapter* granularity without looking at a single frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..llm.client import LLMClient
from ..logging_utils import get_logger
from ..media.ingest import VideoAsset
from ..media.transcript import format_timestamp
from .text import keywords, tokenize

logger = get_logger("chapters")

MAX_BLOCKS = 380


@dataclass
class Chapter:
    index: int
    start: float
    end: float
    title: str
    summary: str = ""
    keywords: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "range": f"{format_timestamp(self.start)} - {format_timestamp(self.end)}",
            "title": self.title,
            "summary": self.summary,
            "keywords": self.keywords,
        }

    def as_prompt_line(self) -> str:
        return (
            f"[{self.start:.0f}-{self.end:.0f}] {self.title}"
            + (f" | {self.summary}" if self.summary else "")
        )


def _blocks(asset: VideoAsset, target_blocks: int = MAX_BLOCKS) -> list[tuple[float, str]]:
    """Compress the transcript into ``[timestamp, text]`` blocks."""
    segments = [seg for seg in asset.transcript.segments if seg.text.strip()]
    if not segments:
        return []

    duration = asset.duration or segments[-1].end
    block_seconds = max(20.0, duration / max(1, target_blocks))

    blocks: list[tuple[float, str]] = []
    current_start = segments[0].start
    buffer: list[str] = []
    for segment in segments:
        if segment.start - current_start > block_seconds and buffer:
            blocks.append((current_start, " ".join(buffer)))
            current_start = segment.start
            buffer = []
        buffer.append(segment.text.strip())
    if buffer:
        blocks.append((current_start, " ".join(buffer)))
    return blocks


_SYSTEM = (
    "You segment lecture transcripts into a table of contents. "
    "You always answer with JSON only."
)

_PROMPT = """Below is a transcript of a lecture video, split into blocks. Each block is
prefixed with its start time in seconds.

{blocks}

Split the lecture into {n} topical chapters (consecutive, non-overlapping, covering the whole video).
For each chapter give:
- "start": start time in seconds (use the block timestamps)
- "end": end time in seconds
- "title": short title in the same language as the transcript (max 12 words)
- "summary": one sentence describing what is taught
- "keywords": 3-6 salient terms (keep technical acronyms verbatim)

Respond with JSON: {{"chapters": [{{"start": 0, "end": 120, "title": "...", "summary": "...", "keywords": ["..."]}}]}}
"""


def build_chapters(
    asset: VideoAsset,
    client: LLMClient,
    *,
    force: bool = False,
    max_chapters: int = 14,
) -> list[Chapter]:
    """Return the chapter list, cached in the per-video cache directory."""
    cache_path: Path = asset.cache_dir / "chapters.json"
    if cache_path.is_file() and not force:
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            chapters = [
                Chapter(
                    index=int(item.get("index", position)),
                    start=float(item["start"]),
                    end=float(item["end"]),
                    title=str(item.get("title", "")),
                    summary=str(item.get("summary", "")),
                    keywords=list(item.get("keywords", [])),
                )
                for position, item in enumerate(payload.get("chapters", []))
            ]
            if chapters:
                logger.info("loaded %d cached chapters", len(chapters))
                return chapters
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("chapter cache unusable (%s); rebuilding", exc)

    blocks = _blocks(asset)
    if not blocks:
        return []

    rendered = "\n".join(f"{start:.0f}s: {text}" for start, text in blocks)
    target = min(max_chapters, max(3, len(blocks) // 12))
    prompt = _PROMPT.format(blocks=rendered, n=target)

    try:
        payload = client.chat_json(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=6144,
            reasoning="off",
        )
        raw_chapters = payload.get("chapters", payload if isinstance(payload, list) else [])
    except Exception as exc:
        logger.warning("chaptering failed (%s); using uniform chapters", exc)
        raw_chapters = []

    duration = asset.duration or blocks[-1][0]
    chapters: list[Chapter] = []
    for position, item in enumerate(raw_chapters):
        try:
            start = max(0.0, float(item.get("start", 0)))
            end = float(item.get("end", start))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        chapters.append(
            Chapter(
                index=position,
                start=start,
                end=min(end, duration) if duration else end,
                title=str(item.get("title", "")).strip() or f"Section {position + 1}",
                summary=str(item.get("summary", "")).strip(),
                keywords=[str(k) for k in item.get("keywords", [])][:8],
            )
        )

    if not chapters:
        chapters = _uniform_chapters(asset, buckets=6)

    chapters.sort(key=lambda chapter: chapter.start)
    for position, chapter in enumerate(chapters):
        chapter.index = position

    cache_path.write_text(
        json.dumps({"chapters": [c.as_dict() for c in chapters]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("built %d chapters", len(chapters))
    return chapters


def _uniform_chapters(asset: VideoAsset, buckets: int = 6) -> list[Chapter]:
    """Deterministic fallback when the LLM is unavailable."""
    duration = asset.duration or (asset.transcript.segments[-1].end if asset.transcript.segments else 0.0)
    if duration <= 0:
        return []
    step = duration / buckets
    return [
        Chapter(
            index=index,
            start=index * step,
            end=min(duration, (index + 1) * step),
            title=f"Section {index + 1}",
        )
        for index in range(buckets)
    ]


def rank_chapters(question: str, chapters: list[Chapter], *, top_k: int = 3) -> list[tuple[Chapter, float]]:
    """Score chapters against the question by term overlap."""
    terms = tokenize(question) + keywords(question, max_terms=8)
    if not terms or not chapters:
        return []
    unique_terms = list(dict.fromkeys(terms))

    scored: list[tuple[Chapter, float]] = []
    for chapter in chapters:
        haystack = " ".join([chapter.title, chapter.summary, " ".join(chapter.keywords)]).lower()
        matched = [term for term in unique_terms if term in haystack]
        if not matched:
            continue
        scored.append((chapter, len(matched) / len(unique_terms)))
    scored.sort(key=lambda item: -item[1])
    return scored[:top_k]