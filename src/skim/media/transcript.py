"""Transcript data model plus SRT/VTT parsing and persistence."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

_TIMESTAMP = re.compile(
    r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})[.,](?P<ms>\d{1,3})"
)


def parse_timestamp(value: str) -> float:
    """``00:01:02,500`` / ``00:01:02.500`` -> seconds."""
    match = _TIMESTAMP.search(value)
    if not match:
        raise ValueError(f"unrecognised timestamp: {value!r}")
    return (
        int(match.group("h")) * 3600
        + int(match.group("m")) * 60
        + int(match.group("s"))
        + int(match.group("ms").ljust(3, "0")) / 1000.0
    )


def format_timestamp(seconds: float) -> str:
    """Seconds -> ``HH:MM:SS`` (used in prompts and reports)."""
    seconds = max(0.0, float(seconds))
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def as_prompt_line(self) -> str:
        tag = f"{self.speaker}: " if self.speaker else ""
        return f"[{format_timestamp(self.start)} - {format_timestamp(self.end)}] {tag}{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Transcript:
    """An ordered list of time-stamped utterances."""

    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str = "unknown"
    source: str = "unknown"  # asr | subtitle | manual
    duration: float = 0.0

    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self):
        return iter(self.segments)

    def text_between(self, start: float, end: float) -> str:
        """Concatenate everything that overlaps ``[start, end]``."""
        parts = [
            seg.text
            for seg in self.segments
            if seg.end > start and seg.start < end and seg.text.strip()
        ]
        return " ".join(parts).strip()

    def window(self, start: float, end: float) -> list[TranscriptSegment]:
        return [seg for seg in self.segments if seg.end > start and seg.start < end]

    def full_text(self) -> str:
        return " ".join(seg.text.strip() for seg in self.segments if seg.text.strip())

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "source": self.source,
            "duration": self.duration,
            "segments": [seg.to_dict() for seg in self.segments],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Transcript":
        segments = [
            TranscriptSegment(
                start=float(item["start"]),
                end=float(item["end"]),
                text=str(item.get("text", "")),
                speaker=item.get("speaker"),
            )
            for item in payload.get("segments", [])
        ]
        return cls(
            segments=segments,
            language=payload.get("language", "unknown"),
            source=payload.get("source", "unknown"),
            duration=float(payload.get("duration", 0.0)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "Transcript":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------- parsing


def parse_srt(content: str) -> list[TranscriptSegment]:
    """Parse SubRip (.srt)."""
    blocks = re.split(r"\r?\n\r?\n+", content.strip())
    segments: list[TranscriptSegment] = []
    for block in blocks:
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        left, _, right = lines[timing_index].partition("-->")
        try:
            start = parse_timestamp(left)
            end = parse_timestamp(right)
        except ValueError:
            continue
        text = " ".join(lines[timing_index + 1 :]).strip()
        text = re.sub(r"<[^>]+>", "", text)  # strip <i> etc.
        if text:
            segments.append(TranscriptSegment(start=start, end=end, text=text))
    return segments


def parse_vtt(content: str) -> list[TranscriptSegment]:
    """Parse WebVTT (.vtt)."""
    body = re.sub(r"^WEBVTT.*?\n\n", "", content.strip(), flags=re.DOTALL)
    blocks = re.split(r"\r?\n\r?\n+", body)
    segments: list[TranscriptSegment] = []
    for block in blocks:
        lines = [line for line in block.splitlines() if line.strip()]
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        left, _, right = lines[timing_index].partition("-->")
        try:
            start = parse_timestamp(left)
            end = parse_timestamp(right.split("align:")[0])
        except ValueError:
            continue
        text = " ".join(lines[timing_index + 1 :]).strip()
        text = re.sub(r"<[^>]+>", "", text)
        if text:
            segments.append(TranscriptSegment(start=start, end=end, text=text))
    return segments


def load_subtitle(path: Path) -> Transcript:
    """Load an .srt / .vtt / .json transcript from disk."""
    path = Path(path)
    content = path.read_text(encoding="utf-8", errors="ignore")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return Transcript.from_dict(json.loads(content))
    if suffix == ".vtt":
        segments = parse_vtt(content)
        source = "subtitle"
    else:
        segments = parse_srt(content)
        source = "subtitle"
    segments.sort(key=lambda seg: seg.start)
    duration = max((seg.end for seg in segments), default=0.0)
    return Transcript(segments=segments, source=source, duration=duration)


def merge_segments(
    segments: list[TranscriptSegment],
    max_gap: float = 1.2,
    max_duration: float = 18.0,
) -> list[TranscriptSegment]:
    """Glue ASR fragments into sentence-sized chunks.

    Whisper emits very short segments; merging them keeps the BM25 index and the
    prompt windows meaningful without losing timestamp resolution.
    """
    merged: list[TranscriptSegment] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if merged:
            previous = merged[-1]
            gap = segment.start - previous.end
            same_speaker = previous.speaker == segment.speaker
            if gap <= max_gap and same_speaker and (segment.end - previous.start) <= max_duration:
                previous.end = max(previous.end, segment.end)
                previous.text = (previous.text + " " + text).strip()
                continue
        merged.append(TranscriptSegment(start=segment.start, end=segment.end, text=text, speaker=segment.speaker))
    return merged