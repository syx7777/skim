"""ffmpeg access.

The host has no system ffmpeg, so a static binary shipped by ``imageio-ffmpeg``
is used for everything: probing, audio extraction (for ASR) and frame
extraction (for the VLM). All calls go through :func:`run_ffmpeg` which keeps
arguments as a list, so Windows paths with spaces and non-ASCII characters work.
"""

from __future__ import annotations

import functools
import io
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..logging_utils import get_logger

logger = get_logger("ffmpeg")


class FFmpegError(RuntimeError):
    pass


@functools.lru_cache(maxsize=1)
def ffmpeg_bin() -> str:
    """Absolute path to the bundled ffmpeg executable."""
    try:
        import imageio_ffmpeg
    except ImportError as exc:  # pragma: no cover
        raise FFmpegError(
            "imageio-ffmpeg is not installed. Run `uv sync` in the project root."
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(args: list[str], *, binary_output: bool = False, timeout: int = 900):
    """Run ffmpeg with the given arguments."""
    command = [ffmpeg_bin(), *args]
    logger.debug("ffmpeg %s", " ".join(args))
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="ignore")
        tail = "\n".join(stderr.strip().splitlines()[-8:])
        raise FFmpegError(f"ffmpeg failed ({completed.returncode}):\n{tail}")
    return completed.stdout if binary_output else completed.stderr.decode("utf-8", errors="ignore")


@dataclass
class VideoInfo:
    path: str
    duration: float
    fps: float
    width: int
    height: int
    has_audio: bool

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 16 / 9


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2})\.(\d+)")
_VIDEO_RE = re.compile(r"Stream #\d+:\d+.*?: Video: .*?(\d{2,5})x(\d{2,5})")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps")


def probe(path: str | Path) -> VideoInfo:
    """Return duration / resolution / fps / audio presence."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"video not found: {path}")

    # Preferred: imageio-ffmpeg's metadata (no ffprobe needed).
    try:
        import imageio_ffmpeg

        generator = imageio_ffmpeg.read_frames(str(path))
        meta = next(generator)
        generator.close()
        size = meta.get("size") or (0, 0)
        info = VideoInfo(
            path=str(path),
            duration=float(meta.get("duration") or 0.0),
            fps=float(meta.get("fps") or 0.0),
            width=int(size[0] or 0),
            height=int(size[1] or 0),
            has_audio=bool(meta.get("audio_codec")),
        )
        if info.duration > 0 and info.width > 0:
            return info
    except Exception as exc:  # pragma: no cover - fallback path
        logger.debug("imageio-ffmpeg probe failed (%s); falling back to stderr parsing", exc)

    text = run_ffmpeg(["-hide_banner", "-i", str(path)])
    duration = 0.0
    if match := _DURATION_RE.search(text):
        duration = (
            int(match.group(1)) * 3600
            + int(match.group(2)) * 60
            + int(match.group(3))
            + int(match.group(4).ljust(2, "0")[:2]) / 100
        )
    width = height = 0
    if match := _VIDEO_RE.search(text):
        width, height = int(match.group(1)), int(match.group(2))
    fps = 0.0
    if match := _FPS_RE.search(text):
        fps = float(match.group(1))
    return VideoInfo(
        path=str(path),
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        has_audio="Audio:" in text,
    )


def scaled_size(width: int, height: int, max_side: int | None) -> tuple[int, int]:
    """Downscale so the longest side is ``max_side``; dimensions stay even."""
    if not max_side or max(width, height) <= max_side:
        width, height = width, height
    else:
        ratio = max_side / max(width, height)
        width, height = int(width * ratio), int(height * ratio)
    width = max(2, width - (width % 2))
    height = max(2, height - (height % 2))
    return width, height


def extract_audio(path: str | Path, out_wav: str | Path, sample_rate: int = 16000) -> Path:
    """Extract a mono 16 kHz PCM WAV, which is what Whisper-family models want."""
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-y",
            "-loglevel", "error",
            "-i", str(path),
            "-vn",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            str(out_wav),
        ]
    )
    return out_wav


def extract_frame(
    path: str | Path,
    timestamp: float,
    max_side: int | None = 768,
) -> Image.Image:
    """Grab a single frame at ``timestamp`` seconds using fast input seeking."""
    info = _cached_probe(str(path))
    width, height = scaled_size(info.width, info.height, max_side)
    command = [
        "-loglevel", "error",
        "-ss", f"{max(0.0, timestamp):.3f}",
        "-i", str(path),
        "-frames:v", "1",
    ]
    if width and height:
        command += ["-vf", f"scale={width}:{height}"]
    command += ["-f", "image2pipe", "-vcodec", "png", "-"]
    payload = run_ffmpeg(command, binary_output=True, timeout=120)
    if not payload:
        raise FFmpegError(f"no frame decoded at t={timestamp:.2f}s of {path}")
    return Image.open(io.BytesIO(payload)).convert("RGB")


def extract_frames(
    path: str | Path,
    timestamps: list[float],
    max_side: int | None = 768,
) -> list[Image.Image]:
    """Extract several timestamps one by one.

    A single pass per timestamp keeps seeking exact; with typical candidate
    windows (tens of frames) this is fast enough and far simpler than
    re-implementing an index.
    """
    frames: list[Image.Image] = []
    for timestamp in timestamps:
        try:
            frames.append(extract_frame(path, timestamp, max_side=max_side))
        except FFmpegError as exc:
            logger.warning("skipping frame at %.2fs: %s", timestamp, exc)
    return frames


def extract_uniform_frames(
    path: str | Path,
    fps: float,
    out_dir: str | Path,
    max_side: int | None = 768,
    start: float = 0.0,
    end: float | None = None,
    quality: int = 4,
) -> list[tuple[float, Path]]:
    """Decode the whole (or a slice of a) video to JPEGs at a fixed fps.

    Used by the fixed-frequency baseline, where a single ffmpeg pass is much
    cheaper than thousands of individual seeks. Returned timestamps are
    approximations: frame *i* is assumed to sit at ``start + i / fps``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = _cached_probe(str(path))
    width, height = scaled_size(info.width, info.height, max_side)

    command = ["-y", "-loglevel", "error", "-ss", f"{max(0.0, start):.3f}"]
    if end is not None:
        command += ["-to", f"{max(0.0, end):.3f}"]
    command += ["-i", str(path), "-vf", f"fps={fps},scale={width}:{height}"]
    command += ["-q:v", str(quality), str(out_dir / "frame_%06d.jpg")]
    run_ffmpeg(command)

    files = sorted(out_dir.glob("frame_*.jpg"))
    return [(start + index / fps, file) for index, file in enumerate(files)]


@functools.lru_cache(maxsize=16)
def _cached_probe(path: str) -> VideoInfo:
    return probe(path)