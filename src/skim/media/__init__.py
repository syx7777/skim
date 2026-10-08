"""Media layer: remote download, ffmpeg, frames, ASR and transcripts."""

from .download import DownloadError, ResolvedSource, fetch_media, is_url, resolve_source
from .ingest import VideoAsset, ingest, load_asset, video_id_for
from .transcript import (
    Transcript,
    TranscriptSegment,
    format_timestamp,
    load_subtitle,
    merge_segments,
)

__all__ = [
    "DownloadError",
    "ResolvedSource",
    "fetch_media",
    "is_url",
    "resolve_source",
    "VideoAsset",
    "ingest",
    "load_asset",
    "video_id_for",
    "Transcript",
    "TranscriptSegment",
    "format_timestamp",
    "load_subtitle",
    "merge_segments",
]