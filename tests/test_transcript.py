"""Transcript parsing, merging and timestamp helpers."""

from __future__ import annotations

from skim.media.transcript import (
    Transcript,
    TranscriptSegment,
    format_timestamp,
    load_subtitle,
    merge_segments,
    parse_srt,
    parse_vtt,
)

SRT = """1
00:00:00,000 --> 00:00:04,500
Hello and welcome.

2
00:00:04,500 --> 00:00:09,250
<i>Today</i> we cover GRPO.
"""

VTT = """WEBVTT

00:00:00.000 --> 00:00:04.500
Hello and welcome.

00:00:04.500 --> 00:00:09.250 align:start position:0%
Today we cover GRPO.
"""


def test_parse_srt_strips_markup():
    segments = parse_srt(SRT)
    assert len(segments) == 2
    assert segments[0].start == 0.0
    assert segments[0].end == 4.5
    assert segments[1].text == "Today we cover GRPO."


def test_parse_vtt():
    segments = parse_vtt(VTT)
    assert len(segments) == 2
    assert segments[1].start == 4.5
    assert segments[1].end == 9.25
    assert segments[1].text == "Today we cover GRPO."


def test_load_subtitle_dispatch(tmp_path):
    srt_path = tmp_path / "a.srt"
    srt_path.write_text(SRT, encoding="utf-8")
    transcript = load_subtitle(srt_path)
    assert transcript.source == "subtitle"
    assert len(transcript) == 2
    assert transcript.duration == 9.25


def test_format_timestamp_rounds_down_at_hour():
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(3661.9) == "01:01:01"


def test_text_between_and_window(transcript):
    assert "critic" in transcript.text_between(12, 24)
    assert "PPO" not in transcript.text_between(48, 60)
    assert len(transcript.window(0, 30)) == 3


def test_merge_segments_glues_close_fragments():
    fragments = [
        TranscriptSegment(0.0, 1.0, "hello"),
        TranscriptSegment(1.2, 2.0, "world"),
        TranscriptSegment(30.0, 31.0, "later"),
    ]
    merged = merge_segments(fragments, max_gap=1.2)
    assert len(merged) == 2
    assert merged[0].text == "hello world"
    assert merged[0].end == 2.0


def test_merge_segments_respects_max_duration():
    fragments = [TranscriptSegment(i * 4.0, i * 4.0 + 3.9, f"part{i}") for i in range(10)]
    merged = merge_segments(fragments, max_gap=1.2, max_duration=18.0)
    assert len(merged) > 1
    assert all(segment.duration <= 20.0 for segment in merged)


def test_transcript_roundtrip(tmp_path, transcript):
    path = tmp_path / "t.json"
    transcript.save(path)
    reloaded = Transcript.load(path)
    assert len(reloaded) == len(transcript)
    assert reloaded.segments[4].text == transcript.segments[4].text
    assert reloaded.duration == transcript.duration