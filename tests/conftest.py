"""Shared fixtures: synthetic transcripts and a fake vision observer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from skim.media.transcript import Transcript, TranscriptSegment
from skim.memory.temporal_memory import Observation


@pytest.fixture
def transcript() -> Transcript:
    """A small two-topic lecture with an explicitly visual-only section."""
    segments = [
        TranscriptSegment(0, 12, "今天讲强化学习里的策略优化，先回顾 PPO。"),
        TranscriptSegment(12, 24, "PPO 需要用一个 critic 网络来估计优势函数 A。"),
        TranscriptSegment(24, 36, "clip 的作用是限制一次策略更新的幅度，避免不稳定。"),
        TranscriptSegment(36, 48, "接下来我们看 GRPO 的核心思想。"),
        TranscriptSegment(48, 60, "GRPO 对同一个问题采样一组回答，做组内归一化。"),
        TranscriptSegment(60, 72, "这样就不再需要 critic 网络了。"),
        TranscriptSegment(72, 84, "大家看这张幻灯片上的公式。"),
        TranscriptSegment(84, 96, "下面我们对比一下 GRPO 和 PPO。"),
        TranscriptSegment(96, 108, "最后总结一下，并给出实验建议。"),
    ]
    return Transcript(segments=segments, language="zh", source="manual", duration=108.0)


@pytest.fixture
def visual_only_observation() -> Observation:
    return Observation(
        start=60.0,
        end=90.0,
        method="frames",
        timestamps=[70.0, 80.0],
        frames=2,
        knowledge_points=["GRPO objective"],
        evidence=["[00:01:10] A_i = (r_i - mean(r)) / (std(r) + eps)"],
        visual_text="A_i = (r_i - mean(r)) / (std(r) + eps)",
        relevance=0.9,
        answers_question=True,
        confidence=0.9,
    )


@pytest.fixture
def fake_observer(visual_only_observation):
    """Duck-typed stand-in for :class:`VisualObserver` (no ffmpeg, no network)."""
    from skim.agent.observer import InspectionOutcome

    class FakeObserver:
        def __init__(self) -> None:
            self.store = SimpleNamespace(duration=108.0)
            self.calls: list[tuple[float, float]] = []

        def frame_budget(self, relevance, *, adaptive=True):
            return 2

        def inspect(self, question, start, end, **kwargs):
            self.calls.append((start, end))
            observation = Observation(**{**visual_only_observation.as_dict(), "start": start, "end": end})
            observation.method = "frames"
            return InspectionOutcome(
                observation=observation,
                payload={"relevance": 0.9, "confidence": 0.9, "still_missing": ""},
                frames=2,
            )

    return FakeObserver()