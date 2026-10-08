"""Frame planning/encoding and the agent's tool surface."""

from __future__ import annotations

from PIL import Image

from skim.agent.tools import ToolBox
from skim.indexing.transcript_index import TranscriptIndex
from skim.media.frames import (
    coarse_grid,
    dense_factor,
    plan_timestamps,
    to_data_url,
)
from skim.memory.temporal_memory import TemporalMemory


# ------------------------------------------------------------------- planning
def test_plan_timestamps_count_and_bounds():
    stamps = plan_timestamps(100.0, 200.0, 5)
    assert len(stamps) == 5
    assert stamps == sorted(stamps)
    assert all(100.0 <= ts <= 200.0 for ts in stamps)


def test_plan_timestamps_handles_degenerate_windows():
    assert plan_timestamps(50.0, 50.0, 4) == [50.0]
    assert plan_timestamps(60.0, 50.0, 4) == [60.0]


def test_plan_timestamps_clamps_to_max_count():
    assert len(plan_timestamps(0.0, 100.0, 999, max_count=8)) <= 8


def test_plan_timestamps_respects_duration():
    stamps = plan_timestamps(0.0, 500.0, 4, duration=200.0)
    assert max(stamps) <= 200.0


def test_coarse_grid_is_sparse_and_bounded():
    grid = coarse_grid(3600.0, interval_s=60.0, max_frames=24)
    assert len(grid) <= 24
    assert grid == sorted(grid)
    assert all(0 <= ts < 3600.0 for ts in grid)


def test_dense_factor_scales_with_relevance():
    assert dense_factor(0.0, base=2, maximum=8) == 2
    assert dense_factor(1.0, base=2, maximum=8) == 8
    assert dense_factor(0.5, base=2, maximum=8) < 8


def test_to_data_url_encodes_jpeg():
    image = Image.new("RGB", (64, 32), "white")
    url, width, height = to_data_url(image, quality=70)
    assert url.startswith("data:image/jpeg;base64,")
    assert (width, height) == (64, 32)


# ---------------------------------------------------------------------- tools
def _toolbox(transcript, observer, memory, *, use_memory=True, adaptive=True) -> ToolBox:
    return ToolBox(
        question="GRPO 的优势怎么算？",
        index=TranscriptIndex(transcript),
        chapters=[],
        observer=observer,
        memory=memory,
        frame_store=observer.store,
        use_memory=use_memory,
        adaptive=adaptive,
    )


def test_inspect_video_refuses_covered_window(transcript, fake_observer):
    memory = TemporalMemory(duration=108.0)
    box = _toolbox(transcript, fake_observer, memory, use_memory=True)

    first = box.dispatch("inspect_video", {"start": 60.0, "end": 90.0})
    assert first.cost == "vlm"
    assert not first.payload.get("refused")
    assert len(fake_observer.calls) == 1

    second = box.dispatch("inspect_video", {"start": 62.0, "end": 88.0})
    assert second.payload.get("refused") is True
    assert len(fake_observer.calls) == 1, "a covered window must not reach the VLM"


def test_inspect_video_force_overrides_memory(transcript, fake_observer):
    memory = TemporalMemory(duration=108.0)
    box = _toolbox(transcript, fake_observer, memory, use_memory=True)
    box.dispatch("inspect_video", {"start": 60.0, "end": 90.0})
    forced = box.dispatch("inspect_video", {"start": 62.0, "end": 88.0, "force": True})
    assert not forced.payload.get("refused")
    assert len(fake_observer.calls) == 2


def test_memory_ablation_allows_reinspection(transcript, fake_observer):
    """With Temporal Memory disabled the agent may re-inspect -- that is the ablation."""
    memory = TemporalMemory(duration=108.0)
    box = _toolbox(transcript, fake_observer, memory, use_memory=False)
    box.dispatch("inspect_video", {"start": 60.0, "end": 90.0})
    again = box.dispatch("inspect_video", {"start": 60.0, "end": 90.0})
    assert not again.payload.get("refused")
    assert len(fake_observer.calls) == 2


def test_cheap_tools_do_not_touch_the_observer(transcript, fake_observer):
    memory = TemporalMemory(duration=108.0)
    box = _toolbox(transcript, fake_observer, memory)
    assert box.dispatch("search_transcript", {"query": "critic"}).payload["hits"]
    assert "PPO" in box.dispatch("read_transcript", {"start": 0, "end": 30}).content
    assert box.dispatch("memory_status", {}).payload["observations"] == 0
    assert fake_observer.calls == []


def test_useful_calls_tracks_transcript_reads(transcript, fake_observer):
    """A transcript read is progress.

    Regression: the stagnation guard used to watch VLM evidence only, so a
    transcript-only investigation looked like "no progress" and the agent
    stopped before reading anything.
    """
    box = _toolbox(transcript, fake_observer, TemporalMemory(duration=108.0))
    assert box.useful_calls == 0

    box.dispatch("search_transcript", {"query": "GRPO"})
    assert box.useful_calls == 1

    box.dispatch("read_transcript", {"start": 40, "end": 70})
    assert box.useful_calls == 2

    # an empty window returns nothing useful
    box.dispatch("read_transcript", {"start": 0, "end": 0})
    assert box.useful_calls == 2


def test_useful_calls_ignores_bookkeeping_and_refusals(transcript, fake_observer):
    memory = TemporalMemory(duration=108.0)
    box = _toolbox(transcript, fake_observer, memory)
    box.dispatch("memory_status", {})
    assert box.useful_calls == 0

    box.dispatch("inspect_video", {"start": 60.0, "end": 90.0})
    after_first = box.useful_calls
    refused = box.dispatch("inspect_video", {"start": 62.0, "end": 88.0})
    assert refused.payload.get("refused") is True
    assert box.useful_calls == after_first, "a refused window is not progress"


def test_read_windows_are_recorded_for_synthesis(transcript, fake_observer):
    box = _toolbox(transcript, fake_observer, TemporalMemory(duration=108.0))
    box.dispatch("read_transcript", {"start": 36, "end": 72})
    assert box.transcript_windows == [(36.0, 72.0)]
    box.dispatch("read_transcript", {"start": 0, "end": 0})
    assert box.transcript_windows == [(36.0, 72.0)], "empty reads are not recorded"


def test_unknown_tool_returns_error(transcript, fake_observer):
    box = _toolbox(transcript, fake_observer, TemporalMemory(duration=108.0))
    assert "unknown tool" in box.dispatch("nope", {}).content


def test_finish_records_the_final_answer(transcript, fake_observer):
    box = _toolbox(transcript, fake_observer, TemporalMemory(duration=108.0))
    result = box.dispatch(
        "finish",
        {"answer": "组内归一化", "confidence": 1.4, "evidence": [{"timestamp": 50, "quote": "x"}]},
    )
    assert box.final is not None
    assert box.final["confidence"] == 1.0, "confidence must be clamped to [0, 1]"
    assert result.payload["answer"] == "组内归一化"


def test_inspect_video_rejects_bad_arguments(transcript, fake_observer):
    box = _toolbox(transcript, fake_observer, TemporalMemory(duration=108.0))
    assert "ERROR" in box.dispatch("inspect_video", {"nope": 1}).content


def test_frame_budget_is_fixed_in_non_adaptive_mode(transcript, fake_observer):
    from skim.agent.observer import VisualObserver

    adaptive = VisualObserver(fake_observer, fake_observer.store, max_frames_per_call=4)
    assert adaptive.frame_budget(1.0, adaptive=True) == 4
    assert adaptive.frame_budget(1.0, adaptive=False) == 2