"""Live tests for the remote (URL) source path.

Opt in with:

    uv run pytest -m network tests/test_remote_network.py

Deliberately metadata-only: downloading a real 1.5-hour lecture takes minutes
and hundreds of megabytes, which is not appropriate for a test suite. The full
download path is covered offline in ``test_download.py`` and exercised manually
via ``skim ingest <url>``.
"""

from __future__ import annotations

import pytest

from skim.config import get_settings
from skim.media.download import (
    DownloadError,
    is_url,
    normalize_url,
    probe_remote,
    resolve_source,
)

pytestmark = pytest.mark.network

#: A real, stable, lecture-length Bilibili episode (宋浩《高等数学》 p106).
LECTURE_URL = "https://www.bilibili.com/video/BV1CAxaeHEeH/?p=106"
#: The same episode copied out of a browser address bar (tracking params).
LECTURE_URL_TRACKED = (
    "https://www.bilibili.com/video/BV1CAxaeHEeH/"
    "?spm_id_from=333.788.videopod.episodes"
    "&vd_source=359a4f9902e6f6dc9ae9e813bae03a59&p=106"
)


def test_probe_remote_reads_lecture_metadata():
    """yt-dlp can reach the site through the configured proxy and parse the page."""
    info = probe_remote(LECTURE_URL, settings=get_settings())
    assert info["duration"] and info["duration"] > 3600, "expected a lecture-length video"
    assert info["title"]
    assert info["media_id"] == "BV1CAxaeHEeH_p106", "the ?p= episode selector must apply"


def test_probe_remote_treats_tracked_and_plain_urls_alike():
    plain = probe_remote(LECTURE_URL, settings=get_settings())
    tracked = probe_remote(LECTURE_URL_TRACKED, settings=get_settings())
    assert plain["media_id"] == tracked["media_id"]


def test_resolve_source_rejects_a_dead_link(tmp_path):
    with pytest.raises(DownloadError):
        resolve_source(
            "https://www.bilibili.com/video/BV0000000000/",
            settings=get_settings(),
            download_dir=tmp_path,
        )


def test_url_detection_and_normalisation_agree():
    assert is_url(LECTURE_URL_TRACKED)
    assert normalize_url(LECTURE_URL_TRACKED) == normalize_url(LECTURE_URL)