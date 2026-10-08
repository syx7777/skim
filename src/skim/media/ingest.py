"""Video ingestion: identity, probing, transcripts and caching."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..config import Settings, get_layout, get_settings
from ..logging_utils import get_logger
from . import ffmpeg
from .asr import ASRBackend, build_backend
from .download import ResolvedSource, resolve_source
from .transcript import Transcript, load_subtitle

logger = get_logger("ingest")

ProgressFn = Callable[[str, float], None]


@dataclass
class VideoAsset:
    """Everything the agent needs about one lecture video."""

    video_id: str
    path: Path
    info: ffmpeg.VideoInfo
    transcript: Transcript
    cache_dir: Path
    source: str = ""
    title: str = ""
    is_remote: bool = False

    @property
    def duration(self) -> float:
        return self.info.duration

    @property
    def display_name(self) -> str:
        return self.title or self.path.name


def video_id_for(path: str | Path) -> str:
    """Stable id derived from the absolute path, file size and mtime."""
    path = Path(path).resolve()
    stat = path.stat()
    digest = hashlib.sha1(
        f"{path}|{stat.st_size}|{int(stat.st_mtime)}".encode("utf-8")
    ).hexdigest()
    return digest[:12]


def _emit(progress: ProgressFn | None, message: str, fraction: float) -> None:
    logger.info(message)
    if progress is not None:
        progress(message, fraction)


def ingest(
    source: str | Path,
    *,
    settings: Settings | None = None,
    subtitle: str | Path | None = None,
    backend: ASRBackend | None = None,
    force: bool = False,
    max_height: int | None = None,
    progress: ProgressFn | None = None,
) -> VideoAsset:
    """Prepare a lecture for question answering.

    ``source`` may be an http(s) URL (Bilibili / YouTube / ...) or a local path.
    URLs are downloaded once into ``data/downloads/`` and cached, so the same
    link is never fetched twice.

    Order of preference for the transcript:

    1. cached ``transcript.json`` from a previous run,
    2. an explicitly supplied subtitle file (``.srt`` / ``.vtt`` / ``.json``),
    3. local ASR over the extracted audio track.

    Frames are *not* pre-extracted here -- they are pulled lazily per candidate
    window so that no compute is spent on parts of the video nobody asks about.
    """
    settings = settings or get_settings()
    layout = get_layout(settings)

    resolved: ResolvedSource = resolve_source(
        source,
        settings=settings,
        download_dir=layout.downloads,
        max_height=max_height or settings.video_max_height,
        force=force,
        progress=progress,
    )
    video_path = resolved.path

    video_id = video_id_for(video_path)
    cache_dir = layout.video_cache(video_id)

    _emit(progress, f"Probing {video_path.name} ...", 0.02)
    info = ffmpeg.probe(video_path)
    logger.info(
        "video: %.1fs, %dx%d, %.2f fps, audio=%s",
        info.duration,
        info.width,
        info.height,
        info.fps,
        info.has_audio,
    )
    if info.duration >= 1800:
        logger.warning(
            "long lecture (%.0f min): speech recognition over the whole audio track "
            "is the slowest stage -- see the README if you want to trade accuracy for speed",
            info.duration / 60,
        )

    transcript_path = cache_dir / "transcript.json"
    meta_path = cache_dir / "meta.json"

    transcript: Transcript | None = None
    if transcript_path.is_file() and not force:
        transcript = Transcript.load(transcript_path)
        _emit(progress, f"Loaded cached transcript ({len(transcript)} segments)", 0.5)

    if transcript is None and subtitle is not None:
        _emit(progress, f"Reading subtitle {Path(subtitle).name} ...", 0.2)
        transcript = load_subtitle(Path(subtitle))
        transcript.duration = transcript.duration or info.duration
        transcript.save(transcript_path)

    if transcript is None:
        if not info.has_audio:
            raise RuntimeError(
                f"{video_path.name} has no audio track; pass --subtitle <file> instead."
            )
        audio_path = cache_dir / "audio.wav"
        if not audio_path.is_file() or force:
            _emit(progress, "Extracting audio track ...", 0.15)
            ffmpeg.extract_audio(video_path, audio_path)
        _emit(progress, "Running speech recognition (this is the slow part) ...", 0.25)
        transcript = (backend or build_backend(settings)).transcribe(
            audio_path, duration=info.duration
        )
        transcript.duration = transcript.duration or info.duration
        transcript.save(transcript_path)

    meta_path.write_text(
        json.dumps(
            {
                "video_id": video_id,
                "path": str(video_path.resolve()),
                "name": video_path.name,
                "title": resolved.title or video_path.name,
                "source": resolved.source,
                "is_remote": resolved.is_remote,
                "uploader": resolved.uploader,
                "webpage_url": resolved.webpage_url,
                "duration": info.duration,
                "fps": info.fps,
                "width": info.width,
                "height": info.height,
                "has_audio": info.has_audio,
                "transcript_source": transcript.source,
                "transcript_segments": len(transcript),
                "language": transcript.language,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _emit(progress, f"Ready: {len(transcript)} transcript segments", 1.0)

    return VideoAsset(
        video_id=video_id,
        path=video_path,
        info=info,
        transcript=transcript,
        cache_dir=cache_dir,
        source=resolved.source,
        title=resolved.title,
        is_remote=resolved.is_remote,
    )


def load_asset(video_id: str, *, settings: Settings | None = None) -> VideoAsset:
    """Re-open a previously ingested video from its cache directory."""
    settings = settings or get_settings()
    layout = get_layout(settings)
    cache_dir = layout.video_cache(video_id)
    meta_path = cache_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"no cached metadata for video id {video_id}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return VideoAsset(
        video_id=video_id,
        path=Path(meta["path"]),
        info=ffmpeg.probe(meta["path"]),
        transcript=Transcript.load(cache_dir / "transcript.json"),
        cache_dir=cache_dir,
        source=meta.get("source", ""),
        title=meta.get("title", ""),
        is_remote=bool(meta.get("is_remote", False)),
    )