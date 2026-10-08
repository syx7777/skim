"""Fetch lecture videos from a URL (Bilibili, YouTube, ...) with yt-dlp.

A lecture is usually referenced by a link rather than a local file, so the
engine accepts either. ``resolve_source`` is the single entry point:

* a URL is downloaded once into ``data/downloads/`` and cached (a sidecar JSON
  records which source produced which file, so the same link is never fetched
  twice);
* a local path is returned unchanged.

Format selection is deliberately opinionated:

* **H.264 (``avc1``) rather than AV1.** The engine decodes hundreds of
  single-timestamp frames with the bundled ffmpeg; AV1 software decoding is both
  slower and a portability risk, and Bilibili serves H.264 at every quality tier.
* **Height capped (default 720p).** Blackboard/formula content must stay legible,
  but 1080p roughly doubles the download for little readable gain after the
  frames are downscaled for the VLM anyway.
* **Low-bitrate audio.** ASR downsamples to mono 16 kHz, so the 120 kbps track
  Bilibili prefers by default is pure waste.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlsplit

from ..config import Settings, apply_proxy, get_settings
from ..logging_utils import get_logger
from . import ffmpeg

logger = get_logger("download")

ProgressFn = Callable[[str, float], None]

URL_PREFIXES = ("http://", "https://")

#: Query parameters that carry no meaning for *which* video is being requested.
#: Bilibili in particular appends tracking junk (`spm_id_from`, `vd_source`, ...)
#: that differs between the address-bar URL and a shared URL, so the cache would
#: otherwise miss for the same lecture.
_TRACKING_PARAMS = {
    "vd_source",
    "from",
    "from_source",
    "from_spmid",
    "share_source",
    "share_medium",
    "share_plat",
    "share_tag",
    "share_session_id",
    "bboverlay",
    "seid",
    "buvid",
    "unique_k",
    "broadcast_type",
    "is_room_feed",
    "timestamp",
    "spmid",
}

#: Prefer H.264 at a bounded height, with a small audio track.
_FORMAT_TEMPLATE = (
    "bv*[height<={height}][vcodec^=avc1]+ba[abr<={abr}]"
    "/bv*[height<={height}][vcodec^=avc1]+ba"
    "/bv*[height<={height}]+ba"
    "/b[height<={height}]"
    "/b"
)


class DownloadError(RuntimeError):
    pass


def is_url(value: str | Path | None) -> bool:
    """True when ``value`` looks like an http(s) URL."""
    return isinstance(value, str) and value.strip().lower().startswith(URL_PREFIXES)


def normalize_url(url: str) -> str:
    """Canonical form of a video URL, used as the download-cache key.

    Drops tracking parameters (everything under ``spm_*`` plus an explicit
    list), sorts what remains, and lowercases scheme/host so that
    ``.../BV1CAxaeHEeH/?spm_id_from=...&p=106`` and ``.../BV1CAxaeHEeH/?p=106``
    resolve to the same cached file.
    """
    parts = urlsplit(url.strip())
    if not parts.scheme or not parts.netloc:
        return url.strip()

    kept = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_PARAMS and not key.lower().startswith("spm_")
    ]
    kept.sort()

    base = f"{parts.scheme.lower()}://{parts.netloc.lower()}{parts.path}"
    query = urlencode(kept)
    return f"{base}?{query}" if query else base


@dataclass
class ResolvedSource:
    """A local media file plus the metadata that produced it."""

    path: Path
    is_remote: bool = False
    title: str = ""
    source: str = ""
    duration: float = 0.0
    uploader: str = ""
    webpage_url: str = ""
    media_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return self.title or self.path.name


def _sidecar_path(download_dir: Path, url: str) -> Path:
    """Cache key for a link, ignoring tracking params so equivalent URLs collide."""
    digest = hashlib.sha1(normalize_url(url).encode("utf-8")).hexdigest()[:12]
    return download_dir / f"{digest}.source.json"


def _load_cached(download_dir: Path, url: str) -> ResolvedSource | None:
    sidecar = _sidecar_path(download_dir, url)
    if not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    path = Path(payload.get("path", ""))
    if not path.is_file():
        return None
    logger.info("reusing downloaded media for this link: %s", path.name)
    return ResolvedSource(
        path=path,
        is_remote=True,
        title=payload.get("title", ""),
        source=url,
        duration=float(payload.get("duration", 0.0) or 0.0),
        uploader=payload.get("uploader", ""),
        webpage_url=payload.get("webpage_url", url),
        media_id=payload.get("media_id", ""),
    )


def _save_sidecar(download_dir: Path, url: str, resolved: ResolvedSource) -> None:
    _sidecar_path(download_dir, url).write_text(
        json.dumps(
            {
                "source": url,
                "normalized_source": normalize_url(url),
                "path": str(resolved.path),
                "title": resolved.title,
                "duration": resolved.duration,
                "uploader": resolved.uploader,
                "webpage_url": resolved.webpage_url,
                "media_id": resolved.media_id,
                "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _build_ydl_options(
    download_dir: Path,
    *,
    settings: Settings,
    max_height: int,
    max_audio_kbps: int,
    progress: ProgressFn | None,
) -> dict[str, Any]:
    import yt_dlp  # imported lazily so the rest of the engine works without it

    def hook(status: dict[str, Any]) -> None:
        if progress is None:
            return
        state = status.get("status")
        if state == "downloading":
            total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
            done = status.get("downloaded_bytes") or 0
            fraction = (done / total) if total else 0.0
            speed = (status.get("speed") or 0) / 1_000_000
            progress(
                f"Downloading {fraction * 100:.0f}% ({speed:.1f} MB/s)",
                min(0.95, fraction),
            )
        elif state == "finished":
            progress("Download finished, merging streams ...", 0.96)

    options: dict[str, Any] = {
        "format": _FORMAT_TEMPLATE.format(height=max_height, abr=max_audio_kbps),
        "outtmpl": str(download_dir / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        # The engine's own ffmpeg does the DASH merge -- no system ffmpeg needed.
        "ffmpeg_location": ffmpeg.ffmpeg_bin(),
        "noplaylist": True,          # a `?p=106` episode link must not pull the series
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
        "retries": 5,
        "fragment_retries": 5,
        # Stalls are detected quickly so the outer loop can ask for a new mirror.
        "socket_timeout": 30,
        # Deliberately 1. Bilibili serves each DASH stream as a single m4s file,
        # and parallel range requests through a local proxy intermittently hang
        # with zero bytes received (measured: the audio track stalls at 0 B with
        # 4 workers, but streams steadily at ~90 KiB/s with 1). Reliability beats
        # peak throughput here -- a stalled download blocks the whole pipeline.
        "concurrent_fragment_downloads": 1,
        # Never clobber a completed file: this is what lets a re-run (or a
        # mirror-failover retry) skip the streams that already finished.
        "overwrites": False,
        "continuedl": True,
    }
    if settings.proxy:
        options["proxy"] = settings.proxy
    _ = yt_dlp  # keep the import meaningful for callers/tests
    return options


def fetch_media(
    url: str,
    download_dir: Path,
    *,
    settings: Settings | None = None,
    max_height: int = 720,
    max_audio_kbps: int = 100,
    force: bool = False,
    progress: ProgressFn | None = None,
) -> ResolvedSource:
    """Download ``url`` into ``download_dir`` and return the local file."""
    settings = settings or get_settings()
    apply_proxy(settings)

    url = url.strip()
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    if not force:
        cached = _load_cached(download_dir, url)
        if cached is not None:
            if progress is not None:
                progress(f"Reusing cached download: {cached.path.name}", 0.05)
            return cached

    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover
        raise DownloadError(
            "yt-dlp is not installed. Run `uv sync` in the project root."
        ) from exc

    logger.info("resolving %s", url)
    if progress is not None:
        progress("Resolving video link ...", 0.01)

    options = _build_ydl_options(
        download_dir,
        settings=settings,
        max_height=max_height,
        max_audio_kbps=max_audio_kbps,
        progress=progress,
    )

    # Bilibili hands out a different CDN mirror on every extraction, and some of
    # them stall through a proxy. `socket_timeout` abandons a dead mirror, and
    # this loop re-extracts (i.e. asks for a new mirror) rather than retrying the
    # same dead URL. Already-finished streams are reused, so a retry is cheap.
    attempts = 3
    info: dict[str, Any] | None = None
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=True)
            last_error = None
            break
        except Exception as exc:  # noqa: BLE001 - yt-dlp raises many types
            last_error = exc
            if attempt >= attempts:
                break
            logger.warning(
                "download attempt %d/%d failed (%s); retrying with a fresh mirror",
                attempt,
                attempts,
                exc,
            )
            if progress is not None:
                progress(f"Retrying download ({attempt}/{attempts}) ...", 0.01)
            time.sleep(2 * attempt)

    if last_error is not None:
        raise DownloadError(f"could not download {url}: {last_error}") from last_error

    if info is None:
        raise DownloadError(f"no media found at {url}")
    if info.get("_type") == "playlist":
        entries = [entry for entry in (info.get("entries") or []) if entry]
        if not entries:
            raise DownloadError(f"playlist at {url} contained no videos")
        info = entries[0]

    path = _downloaded_path(info)
    if path is None or not path.is_file():
        raise DownloadError(f"download reported success but no file was produced for {url}")

    resolved = ResolvedSource(
        path=path,
        is_remote=True,
        title=str(info.get("title") or path.stem),
        source=url,
        duration=float(info.get("duration") or 0.0),
        uploader=str(info.get("uploader") or ""),
        webpage_url=str(info.get("webpage_url") or url),
        media_id=str(info.get("id") or ""),
        extra={
            "resolution": f"{info.get('width')}x{info.get('height')}",
            "fps": info.get("fps"),
            "vcodec": info.get("vcodec"),
        },
    )
    _save_sidecar(download_dir, url, resolved)
    logger.info(
        "downloaded '%s' (%.0fs, %s) -> %s",
        resolved.title,
        resolved.duration,
        resolved.extra.get("resolution"),
        path.name,
    )
    if progress is not None:
        progress(f"Downloaded {path.name}", 1.0)
    return resolved


def _downloaded_path(info: dict[str, Any]) -> Path | None:
    """Work out where yt-dlp actually put the merged file."""
    for requested in info.get("requested_downloads") or []:
        candidate = requested.get("filepath") or requested.get("_filename")
        if candidate:
            path = Path(candidate)
            if path.is_file():
                return path
            for suffix in (".mp4", ".mkv", ".webm"):
                sibling = path.with_suffix(suffix)
                if sibling.is_file():
                    return sibling

    filename = info.get("filepath") or info.get("_filename")
    if filename:
        path = Path(filename)
        if path.is_file():
            return path
        for suffix in (".mp4", ".mkv", ".webm"):
            sibling = path.with_suffix(suffix)
            if sibling.is_file():
                return sibling
    return None


def probe_remote(url: str, *, settings: Settings | None = None) -> dict[str, Any]:
    """Metadata-only lookup (no download). Handy for UIs and sanity checks."""
    settings = settings or get_settings()
    apply_proxy(settings)
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover
        raise DownloadError("yt-dlp is not installed. Run `uv sync`.") from exc

    options: dict[str, Any] = {"quiet": True, "no_warnings": True, "noplaylist": True}
    if settings.proxy:
        options["proxy"] = settings.proxy
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "title": info.get("title"),
        "duration": info.get("duration"),
        "uploader": info.get("uploader"),
        "webpage_url": info.get("webpage_url"),
        "media_id": info.get("id"),
    }


def resolve_source(
    source: str | Path,
    *,
    settings: Settings | None = None,
    download_dir: Path | None = None,
    max_height: int = 720,
    force: bool = False,
    progress: ProgressFn | None = None,
) -> ResolvedSource:
    """Accept a URL *or* a local path and always return a local file."""
    settings = settings or get_settings()

    if is_url(source):
        directory = Path(download_dir) if download_dir else Path(settings.data_dir) / "downloads"
        return fetch_media(
            str(source),
            directory,
            settings=settings,
            max_height=max_height,
            force=force,
            progress=progress,
        )

    path = Path(str(source))
    if not path.is_file():
        raise FileNotFoundError(f"media not found: {path}")
    return ResolvedSource(path=path, is_remote=False, title=path.name, source=str(path))