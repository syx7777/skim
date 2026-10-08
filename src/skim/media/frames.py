"""Frame planning, extraction, caching and encoding for the VLM."""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..logging_utils import get_logger
from . import ffmpeg

logger = get_logger("frames")


@dataclass
class SampledFrame:
    """One extracted frame plus its position in the video."""

    timestamp: float
    image: Image.Image
    label: str = ""

    def to_attachment(self, quality: int = 80):
        """Encode for the API. Imported lazily to avoid a circular import."""
        from ..llm.types import ImageAttachment

        data_url, width, height = to_data_url(self.image, quality=quality)
        label = self.label or f"t={self.timestamp:.1f}s"
        return ImageAttachment(
            data_url=data_url,
            timestamp=self.timestamp,
            width=width,
            height=height,
            label=label,
        )


def to_data_url(image: Image.Image, quality: int = 80) -> tuple[str, int, int]:
    """JPEG-encode a PIL image as a data URL, returning its dimensions."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}", image.width, image.height


def plan_timestamps(
    start: float,
    end: float,
    count: int,
    *,
    edge_margin: float = 0.25,
    max_count: int = 12,
    duration: float | None = None,
) -> list[float]:
    """Evenly distribute ``count`` sample points across ``[start, end]``.

    Density is decided by the caller (this is the "task-relevance-driven"
    part); this function only turns a budget into concrete timestamps.
    """
    count = max(1, min(int(count), max_count))
    start = max(0.0, float(start))
    end = float(end)
    if duration is not None:
        end = min(end, float(duration))
    if end <= start:
        return [round(start, 2)]

    span = end - start
    margin = min(edge_margin, span / 4) if span > 1.0 else 0.0
    low, high = start + margin, end - margin
    if count == 1:
        timestamps = [(low + high) / 2]
    else:
        step = (high - low) / (count - 1)
        timestamps = [low + index * step for index in range(count)]
    # Rounding to 0.1s keeps the on-disk cache effective across runs.
    return sorted({round(min(max(ts, 0.0), end), 1) for ts in timestamps})


def coarse_grid(
    duration: float,
    *,
    interval_s: float = 60.0,
    max_frames: int = 24,
) -> list[float]:
    """A sparse uniform grid over the whole video (cheap visual overview)."""
    if duration <= 0:
        return []
    interval = max(interval_s, duration / max(1, max_frames))
    timestamps = [round(index * interval, 1) for index in range(int(duration // interval) + 1)]
    return [ts for ts in timestamps if ts < duration][:max_frames]


class FrameStore:
    """Per-video frame extraction with a disk cache.

    Re-visiting the same timestamp across agent steps never hits ffmpeg twice,
    which matters because the Temporal Memory guarantees we skip repeated
    windows but the agent may still overlap windows slightly.
    """

    def __init__(
        self,
        video_path: str | Path,
        cache_dir: str | Path,
        *,
        max_side: int = 768,
        jpeg_quality: int = 80,
    ):
        self.video_path = Path(video_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_side = max_side
        self.jpeg_quality = jpeg_quality
        self._info = ffmpeg.probe(self.video_path)
        self.extracted = 0  # frames actually decoded (cost signal)

    @property
    def info(self) -> ffmpeg.VideoInfo:
        return self._info

    @property
    def duration(self) -> float:
        return self._info.duration

    def _cache_path(self, timestamp: float) -> Path:
        return self.cache_dir / f"frame_t{timestamp:09.3f}_s{self.max_side}.jpg"

    def get(self, timestamp: float) -> SampledFrame:
        """Frame at ``timestamp`` (0.1s resolution), cached on disk."""
        timestamp = round(float(timestamp), 1)
        path = self._cache_path(timestamp)
        if path.is_file():
            with Image.open(path) as handle:
                image = handle.convert("RGB")
        else:
            image = ffmpeg.extract_frame(self.video_path, timestamp, max_side=self.max_side)
            self.extracted += 1
            image.save(path, format="JPEG", quality=self.jpeg_quality, optimize=True)
        return SampledFrame(timestamp=timestamp, image=image)

    def get_many(self, timestamps: list[float]) -> list[SampledFrame]:
        frames: list[SampledFrame] = []
        for timestamp in timestamps:
            try:
                frames.append(self.get(timestamp))
            except ffmpeg.FFmpegError as exc:
                logger.warning("frame extraction failed at %.1fs: %s", timestamp, exc)
        return frames

    def window(
        self,
        start: float,
        end: float,
        count: int,
        *,
        max_count: int = 12,
    ) -> list[SampledFrame]:
        """Plan + extract ``count`` frames inside a candidate window."""
        timestamps = plan_timestamps(
            start, end, count, max_count=max_count, duration=self.duration
        )
        return self.get_many(timestamps)

    def coverage(self, timestamps: list[float], bucket_s: float = 5.0) -> int:
        """Number of distinct time buckets touched (used for coverage metrics)."""
        return len({int(ts // bucket_s) for ts in timestamps})

    def window_stats(self, timestamps: list[float]) -> dict[str, float]:
        if not timestamps:
            return {"count": 0, "span": 0.0, "unique": 0}
        return {
            "count": len(timestamps),
            "span": round(max(timestamps) - min(timestamps), 2),
            "unique": len(set(timestamps)),
        }


def dense_factor(relevance: float, base: int = 3, maximum: int = 10) -> int:
    """Map a relevance score in ``[0, 1]`` to a frame budget."""
    relevance = max(0.0, min(1.0, float(relevance)))
    return int(round(base + (maximum - base) * math.sqrt(relevance)))