"""Temporal Memory: coverage bookkeeping and prompt digests."""

from __future__ import annotations

from skim.memory.temporal_memory import Observation, TemporalMemory


def _visual(start: float, end: float, **kwargs) -> Observation:
    return Observation(start=start, end=end, method="frames", **kwargs)


def test_new_ratio_is_one_for_unexplored_windows():
    memory = TemporalMemory(duration=600.0)
    assert memory.new_ratio(100.0, 200.0) == 1.0


def test_new_ratio_accounts_for_partial_overlap():
    memory = TemporalMemory(duration=600.0)
    memory.add(_visual(100.0, 200.0))
    assert memory.new_ratio(150.0, 250.0) == 0.5
    assert memory.new_ratio(100.0, 200.0) == 0.0


def test_transcript_reads_do_not_block_visual_inspection():
    """Reading the transcript is free; it must not stop the agent looking at frames."""
    memory = TemporalMemory(duration=600.0)
    memory.add(Observation(start=0.0, end=300.0, method="transcript"))
    assert memory.new_ratio(0.0, 300.0) == 1.0
    assert not memory.is_covered(0.0, 300.0)
    assert memory.coverage_ratio() == 0.0


def test_is_covered_uses_threshold():
    memory = TemporalMemory(duration=600.0)
    memory.add(_visual(0.0, 80.0))
    assert memory.is_covered(0.0, 100.0, threshold=0.6)       # 80% covered
    assert not memory.is_covered(0.0, 100.0, threshold=0.9)   # only 80% covered


def test_adjacent_observations_merge():
    memory = TemporalMemory(duration=600.0)
    memory.add(_visual(0.0, 50.0))
    memory.add(_visual(50.0, 100.0))
    assert memory.checked_intervals == [(0.0, 100.0)]


def test_coverage_ratio_and_stats():
    memory = TemporalMemory(duration=100.0)
    memory.add(_visual(0.0, 25.0, timestamps=[5.0, 15.0], frames=2, relevance=0.8, confidence=0.9))
    stats = memory.stats()
    assert stats["frames_viewed"] == 2
    assert stats["vlm_looks"] == 1
    assert memory.coverage_ratio() == 0.25


def test_scan_and_frames_both_count_as_visual():
    memory = TemporalMemory(duration=100.0)
    memory.add(Observation(start=0.0, end=10.0, method="scan"))
    memory.add(_visual(20.0, 30.0))
    assert memory.vlm_looks() == 2
    assert memory.coverage_ratio() == 0.2


def test_digest_and_open_questions():
    memory = TemporalMemory(duration=100.0)
    memory.add(
        _visual(
            10.0,
            40.0,
            evidence=["[00:00:20] A_i = (r - mu) / sigma"],
            visual_text="A_i = (r - mu) / sigma",
            knowledge_points=["GRPO advantage"],
            relevance=0.9,
        )
    )
    memory.add_open_question("what is eps?")
    digest = memory.digest()
    assert "A_i" in digest
    assert "what is eps?" in memory.open_questions_digest()


def test_best_confidence_and_answered_by():
    memory = TemporalMemory(duration=100.0)
    memory.add(_visual(0.0, 10.0, confidence=0.3, answers_question=False))
    memory.add(_visual(10.0, 20.0, confidence=0.9, answers_question=True))
    assert memory.best_confidence() == 0.9
    assert len(memory.answered_by()) == 1


def test_roundtrip():
    memory = TemporalMemory(duration=100.0)
    memory.add(_visual(0.0, 10.0, timestamps=[1.0, 2.0], frames=2, evidence=["x"]))
    memory.add_open_question("q")
    restored = TemporalMemory.from_dict(memory.to_dict())
    assert restored.duration == 100.0
    assert len(restored.observations) == 1
    assert restored.frame_timestamps == [1.0, 2.0]
    assert restored.open_questions == ["q"]