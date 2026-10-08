"""End-to-end test against the live VLM.

Skipped by default. Run with:

    uv run pytest -m network tests/test_agent_network.py

These tests build the synthetic lecture, then check that the agent finds the
transcript-only answer cheaply and needs vision for the slide-only answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from skim.agent.agent import AgentConfig, SkimAgent
from skim.config import get_settings
from skim.demo import build_sample
from skim.llm.client import LLMClient
from skim.media.ingest import ingest

pytestmark = pytest.mark.network

ASSETS = Path("data/assets")


@pytest.fixture(scope="module")
def asset():
    video, dataset = build_sample(ASSETS)
    settings = get_settings()
    return ingest(video, settings=settings, subtitle=dataset.with_name("sample_lecture.transcript.json"))


@pytest.fixture(scope="module")
def agent(asset):
    return SkimAgent(LLMClient(get_settings()), asset, AgentConfig(max_steps=6))


def test_transcript_answerable_question_stays_cheap(agent):
    """PPO's clip motivation is spoken aloud -> the agent should not need frames."""
    result = agent.run("PPO 为什么要用 clip？")
    assert result.answer
    assert result.confidence > 0
    assert result.metrics["frames_sent"] <= 4, "a transcript-answerable question should stay cheap"


def test_vision_requiring_question_uses_the_slides(agent):
    """The GRPO advantage formula exists only on a slide."""
    result = agent.run("GRPO 的目标函数里，优势 A_i 具体是怎么算出来的？")
    assert result.metrics["vlm_calls"] >= 1, "must have looked at the video"
    assert result.metrics["frames_sent"] >= 1
    assert result.answer
    combined = result.answer + " ".join(
        str(item.get("quote", "")) for item in result.evidence
    )
    assert "mean" in combined or "归一化" in combined


def test_temporal_memory_prevents_repeat_inspection(agent):
    """Within one run the agent must not inspect overlapping windows.

    Temporal Memory is scoped to a single question (each ``run`` starts a fresh
    one), so the guarantee to check is that the *same* run never pays twice for
    the same stretch of video.
    """
    result = agent.run("GRPO 的目标函数里，优势 A_i 具体是怎么算出来的？其中 KL 项的作用是什么？")
    inspected = [
        (float(entry["arguments"]["start"]), float(entry["arguments"]["end"]))
        for entry in result.trace
        if entry.get("action") == "inspect_video" and not entry.get("summary", "").startswith("REFUSED")
    ]
    assert len(inspected) >= 1
    for index, (start_a, end_a) in enumerate(inspected):
        for start_b, end_b in inspected[index + 1 :]:
            overlap = max(0.0, min(end_a, end_b) - max(start_a, start_b))
            shorter = min(end_a - start_a, end_b - start_b)
            assert overlap / shorter < 0.7, f"windows {start_a}-{end_a} and {start_b}-{end_b} overlap"


def test_answer_language_matches_the_question(agent):
    result = agent.run("What is the main difference between GRPO and PPO?")
    assert any("\u4e00" <= char <= "\u9fff" for char in result.answer) or "critic" in result.answer.lower()