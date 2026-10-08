"""Tokenisation, BM25 retrieval and coarse localisation."""

from __future__ import annotations

from skim.indexing.chapters import Chapter
from skim.indexing.coarse import (
    coarse_localize,
    question_needs_vision,
    transcript_density,
)
from skim.indexing.text import contains_term, keywords, tokenize
from skim.indexing.transcript_index import TranscriptIndex


def test_tokenize_keeps_acronyms_and_chinese():
    tokens = tokenize("GRPO 和 PPO 的区别，用组内归一化")
    assert "grpo" in tokens
    assert "ppo" in tokens
    assert "归一化" in tokens


def test_keywords_ranks_acronyms_first():
    terms = keywords("老师在哪一段讲了 GRPO 和 PPO 的区别")
    assert terms[0] in {"grpo", "ppo"}


def test_contains_term_is_case_insensitive():
    assert contains_term("GRPO is great", "grpo")
    assert contains_term("我们用组内归一化", "归一化")
    assert not contains_term("PPO", "GRPO")


def test_index_search_returns_relevant_segment(transcript):
    index = TranscriptIndex(transcript)
    hits = index.search("critic 网络")
    assert hits
    assert "critic" in hits[0].segment.text


def test_index_search_can_be_restricted_to_window(transcript):
    """The window filter is overlap-based: a segment spanning the boundary counts."""
    index = TranscriptIndex(transcript)
    hits = index.search("GRPO", start=40.0, end=90.0)
    assert hits
    assert all(hit.segment.end > 40.0 and hit.segment.start < 90.0 for hit in hits)
    # nothing that ends before the window starts
    assert all(hit.start >= 36.0 for hit in hits)


def test_index_locate_term(transcript):
    index = TranscriptIndex(transcript)
    found = index.locate_term("clip")
    assert len(found) == 1
    assert found[0].start == 24.0


def test_coarse_windows_are_bounded_and_padded(transcript):
    index = TranscriptIndex(transcript)
    windows = index.coarse_windows("GRPO 组内归一化", max_span=40.0)
    assert windows
    for window in windows:
        assert window.end - window.start <= 40.0 + 2 * 12.0 + 1e-6


def test_coarse_localize_prefers_chapter_match(transcript):
    index = TranscriptIndex(transcript)
    chapters = [
        Chapter(index=0, start=0.0, end=36.0, title="回顾 PPO", summary="clip 与优势估计"),
        Chapter(index=1, start=36.0, end=84.0, title="GRPO 原理", summary="组内归一化替代 critic"),
        Chapter(index=2, start=84.0, end=108.0, title="对比与小结", summary="GRPO vs PPO"),
    ]
    result = coarse_localize("GRPO 为什么不需要 critic？", index, chapters)
    assert result.windows
    assert result.matched_chapters
    assert result.matched_chapters[0][0].title == "GRPO 原理"


def test_question_needs_vision_heuristic():
    assert question_needs_vision("这张 PPT 上的公式是什么？")
    assert question_needs_vision("what does the code snippet on the slide do?")
    assert not question_needs_vision("老师讲了什么学习率？")


def test_transcript_density_flags_visual_only_sections(transcript):
    dense = transcript_density(transcript, 12.0, 24.0)
    # segment 6 is a single short sentence spread over 12 seconds
    sparse = transcript_density(transcript, 72.0, 84.0)
    assert sparse < dense


def test_search_on_empty_transcript_is_safe():
    from skim.media.transcript import Transcript

    index = TranscriptIndex(Transcript(segments=[], duration=0.0))
    assert index.search("anything") == []
    assert index.coarse_windows("anything") == []