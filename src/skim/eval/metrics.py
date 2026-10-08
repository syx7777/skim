"""Metrics for comparing the agent against the baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..llm.client import LLMClient
from ..logging_utils import get_logger

logger = get_logger("metrics")

#: Outcome counters reported for every method.
NUMERIC_KEYS = [
    "frames_sent",
    "vlm_calls",
    "visual_tokens",
    "coverage_ratio",
    "wall_latency_s",
    "total_tokens",
]

JUDGE_SYSTEM = (
    "You grade answers to questions about a lecture video. "
    "Be strict about factual correctness but tolerant about wording, length and language. "
    "Answer with JSON only."
)

JUDGE_PROMPT = """Question: {question}

Reference answer: {gold}

Candidate answer: {pred}

Grade the candidate:
- "score": 1.0 if it conveys the same key facts as the reference; 0.5 if partially
  correct (right idea, missing or wrong details); 0.0 if wrong or evasive.
- "correct": true only when score >= 0.8
- "reason": one short sentence

Respond with JSON: {{"score": 0.0, "correct": false, "reason": ""}}"""


@dataclass
class JudgeVerdict:
    score: float
    correct: bool
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"score": round(self.score, 3), "correct": self.correct, "reason": self.reason}


def judge_answer(
    client: LLMClient,
    question: str,
    prediction: str,
    gold: str,
) -> JudgeVerdict:
    """LLM-as-judge scoring of one answer."""
    if not gold:
        return JudgeVerdict(score=0.0, correct=False, reason="no reference answer")
    if not prediction.strip():
        return JudgeVerdict(score=0.0, correct=False, reason="empty answer")

    try:
        payload = client.chat_json(
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(question=question, gold=gold, pred=prediction[:4000]),
                },
            ],
            max_tokens=1024,
            reasoning="off",
        )
    except Exception as exc:
        logger.warning("judge failed: %s", exc)
        return JudgeVerdict(score=0.0, correct=False, reason=f"judge error: {exc}")

    try:
        score = max(0.0, min(1.0, float(payload.get("score", 0.0))))
    except (TypeError, ValueError):
        score = 0.0
    correct = bool(payload.get("correct", score >= 0.8))
    return JudgeVerdict(score=score, correct=correct, reason=str(payload.get("reason", "")))


def temporal_hit(
    used_timestamps: list[float],
    gold_window: list[float] | None,
    *,
    tolerance: float = 15.0,
) -> float | None:
    """1.0 if the method looked inside the reference window, else 0.0.

    This is the localisation half of the evaluation: it separates "answered
    correctly" from "answered correctly *for the right reason*".
    """
    if not gold_window or len(gold_window) != 2 or not used_timestamps:
        return None
    start, end = float(gold_window[0]) - tolerance, float(gold_window[1]) + tolerance
    return 1.0 if any(start <= ts <= end for ts in used_timestamps) else 0.0


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean of every numeric metric across runs, plus the answer accuracy."""
    if not rows:
        return {}

    summary: dict[str, Any] = {"n": len(rows)}
    scores = [row.get("judge", {}).get("score", 0.0) for row in rows]
    summary["answer_score"] = round(sum(scores) / len(scores), 4)
    summary["accuracy"] = round(
        sum(1 for row in rows if row.get("judge", {}).get("correct")) / len(rows), 4
    )

    hits = [row["temporal_hit"] for row in rows if row.get("temporal_hit") is not None]
    if hits:
        summary["temporal_hit_rate"] = round(sum(hits) / len(hits), 4)

    for key in NUMERIC_KEYS:
        values = [
            row.get("metrics", {}).get(key)
            for row in rows
            if isinstance(row.get("metrics", {}).get(key), (int, float))
        ]
        if values:
            summary[key] = round(sum(values) / len(values), 4)
    return summary