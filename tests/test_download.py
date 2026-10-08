"""Remote-source handling: URL detection, caching and file resolution.

These are all offline: no network and no yt-dlp invocation.
"""

from __future__ import annotations

import json

import pytest

from skim.media.download import (
    _FORMAT_TEMPLATE,
    ResolvedSource,
    _downloaded_path,
    _load_cached,
    _save_sidecar,
    _sidecar_path,
    is_url,
    normalize_url,
    resolve_source,
)

URL = "https://www.bilibili.com/video/BV1CAxaeHEeH/?p=106"
#: The same episode as it appears in a browser address bar.
URL_TRACKED = (
    "https://www.bilibili.com/video/BV1CAxaeHEeH/"
    "?spm_id_from=333.788.videopod.episodes"
    "&vd_source=359a4f9902e6f6dc9ae9e813bae03a59&p=106"
)


@pytest.mark.parametrize(
    "value,expected",
    [
        (URL, True),
        ("https://youtu.be/abc", True),
        ("http://example.com/v.mp4", True),
        ("HTTPS://EXAMPLE.COM/v", True),
        (r"D:\videos\lecture.mp4", False),
        ("./lecture.mp4", False),
        ("lecture.mp4", False),
        ("", False),
        (None, False),
    ],
)
def test_is_url(value, expected):
    assert is_url(value) is expected


def test_format_template_prefers_h264_and_caps_height():
    """Regression guard: AV1 is slow/risky to decode frame-by-frame, so H.264 first."""
    rendered = _FORMAT_TEMPLATE.format(height=720, abr=100)
    assert rendered.index("avc1") < rendered.index("/bv*[height<=720]+ba")
    assert "height<=720" in rendered
    assert "abr<=100" in rendered
    # always has a last-resort fallback so an unavailable tier cannot hard-fail
    assert rendered.rstrip().endswith("/b")


def test_sidecar_path_is_stable_and_url_specific(tmp_path):
    first = _sidecar_path(tmp_path, URL)
    assert first == _sidecar_path(tmp_path, URL)
    assert first != _sidecar_path(tmp_path, URL + "&p=107")
    assert first.name.endswith(".source.json")


def test_normalize_url_strips_tracking_params():
    assert normalize_url(URL_TRACKED) == URL


def test_normalize_url_keeps_meaningful_params():
    assert normalize_url("https://x.com/v?b=2&a=1") == "https://x.com/v?a=1&b=2"


def test_normalize_url_is_idempotent():
    once = normalize_url(URL_TRACKED)
    assert normalize_url(once) == once


def test_normalize_url_lowercases_scheme_and_host_only():
    assert normalize_url("HTTPS://Example.COM/Path?q=1") == "https://example.com/Path?q=1"


def test_normalize_url_tolerates_non_urls():
    assert normalize_url("not a url") == "not a url"


def test_equivalent_urls_share_one_cache_entry(tmp_path):
    """The address-bar URL and the shared URL must not download twice."""
    assert _sidecar_path(tmp_path, URL) == _sidecar_path(tmp_path, URL_TRACKED)


def test_resolve_source_returns_local_paths_untouched(tmp_path):
    video = tmp_path / "lecture.mp4"
    video.write_bytes(b"x")
    resolved = resolve_source(video)
    assert resolved.is_remote is False
    assert resolved.path == video
    assert resolved.title == "lecture.mp4"


def test_resolve_source_rejects_missing_local_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_source(tmp_path / "nope.mp4")


def test_sidecar_roundtrip_enables_reuse(tmp_path):
    media = tmp_path / "video.mp4"
    media.write_bytes(b"x" * 10)
    resolved = ResolvedSource(
        path=media,
        is_remote=True,
        title="定积分的换元法",
        source=URL,
        duration=5332.18,
        uploader="宋浩老师官方",
        webpage_url=URL,
        media_id="BV1CAxaeHEeH_p106",
    )
    _save_sidecar(tmp_path, URL, resolved)

    cached = _load_cached(tmp_path, URL)
    assert cached is not None
    assert cached.path == media
    assert cached.title == "定积分的换元法"
    assert cached.duration == pytest.approx(5332.18)
    assert cached.is_remote is True


def test_cache_is_ignored_when_the_file_vanished(tmp_path):
    media = tmp_path / "video.mp4"
    media.write_bytes(b"x")
    _save_sidecar(tmp_path, URL, ResolvedSource(path=media, is_remote=True, source=URL))
    media.unlink()
    assert _load_cached(tmp_path, URL) is None


def test_cache_is_ignored_on_corrupt_sidecar(tmp_path):
    _sidecar_path(tmp_path, URL).write_text("{not json", encoding="utf-8")
    assert _load_cached(tmp_path, URL) is None


def test_missing_sidecar_is_not_an_error(tmp_path):
    assert _load_cached(tmp_path, URL) is None


def test_downloaded_path_prefers_requested_download(tmp_path):
    target = tmp_path / "out.mp4"
    target.write_bytes(b"x")
    info = {"requested_downloads": [{"filepath": str(target)}]}
    assert _downloaded_path(info) == target


def test_downloaded_path_falls_back_to_merged_suffix(tmp_path):
    """yt-dlp may report a pre-merge name; the merged .mp4 is what exists."""
    reported = tmp_path / "out.webm"
    merged = tmp_path / "out.mp4"
    merged.write_bytes(b"x")
    assert _downloaded_path({"filepath": str(reported)}) == merged


def test_downloaded_path_returns_none_when_nothing_exists(tmp_path):
    assert _downloaded_path({"filepath": str(tmp_path / "gone.mp4")}) is None


def test_sidecar_records_the_source_url(tmp_path):
    media = tmp_path / "v.mp4"
    media.write_bytes(b"x")
    _save_sidecar(tmp_path, URL, ResolvedSource(path=media, is_remote=True, source=URL))
    payload = json.loads(_sidecar_path(tmp_path, URL).read_text(encoding="utf-8"))
    assert payload["source"] == URL
    assert payload["path"] == str(media)