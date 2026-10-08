"""Benchmark runner: agent vs. baselines vs. ablations.

Produces the numbers described in the project brief -- answer accuracy next to
average frames viewed, VLM calls, visual tokens, video coverage and latency --
and runs the three ablations (Adaptive Observation, Temporal Memory,
uncertainty-based stopping).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..agent.agent import AgentConfig, SkimAgent
from ..llm.client import LLMClient
from ..logging_utils import get_logger
from ..media.download import is_url
from ..media.ingest import VideoAsset, ingest
from .baselines import run_fixed_fps, run_subtitle_rag
from .metrics import aggregate, judge_answer, temporal_hit

logger = get_logger("runner")


#: Named method -> factory building a callable ``(asset, client, row) -> result``.
METHODS: dict[str, str] = {
    "agent": "The full agent: adaptive observation + temporal memory + uncertainty stopping.",
    "agent-no-adaptive": "Ablation: fixed frame density, no relevance-driven sampling.",
    "agent-no-memory": "Ablation: no Temporal Memory, windows may be re-inspected.",
    "agent-no-uncertainty": "Ablation: never stops early on confidence.",
    "subtitle_rag": "Baseline: transcript-only RAG, no video.",
    "fixed_fps": "Baseline: uniform frame sampling across the whole video.",
}

AGENT_ABLATION_FLAGS = {
    "agent": {},
    "agent-no-adaptive": {"adaptive_observation": False},
    "agent-no-memory": {"use_temporal_memory": False},
    "agent-no-uncertainty": {"use_uncertainty": False},
}


@dataclass
class Dataset:
    video: str  # an http(s) link or a local path
    questions: list[dict[str, Any]] = field(default_factory=list)
    subtitle: Path | None = None
    name: str = ""

    @property
    def video_label(self) -> str:
        return self.video

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        base = path.parent

        raw_video = str(payload["video"])
        if is_url(raw_video):
            video = raw_video  # a lecture referenced by link
        else:
            candidate = Path(raw_video)
            video = str(candidate if candidate.is_absolute() else (base / candidate).resolve())

        subtitle = payload.get("subtitle")
        subtitle_path = None
        if subtitle:
            candidate = Path(subtitle)
            subtitle_path = candidate if candidate.is_absolute() else (base / candidate).resolve()
        questions = [q for q in payload.get("questions", []) if q.get("question")]
        return cls(
            video=video,
            questions=questions,
            subtitle=subtitle_path,
            name=payload.get("name", path.stem),
        )


def _agent_config(method: str, base: dict[str, Any]) -> AgentConfig:
    overrides = dict(AGENT_ABLATION_FLAGS[method])
    overrides.update(base)
    return AgentConfig(**overrides)


def run_evaluation(
    dataset: Dataset,
    client: LLMClient,
    *,
    methods: list[str] | None = None,
    agent_options: dict[str, Any] | None = None,
    fps_interval_s: float = 30.0,
    judge: bool = True,
    output_dir: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run every requested method over every question and return a report."""
    methods = methods or ["agent", "subtitle_rag", "fixed_fps"]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; available: {sorted(METHODS)}")

    def notify(message: str) -> None:
        logger.info(message)
        if progress is not None:
            progress(message)

    notify(f"Preparing lecture {dataset.video_label} ...")
    asset: VideoAsset = ingest(dataset.video, subtitle=dataset.subtitle)
    notify(f"Ready: {asset.display_name} ({asset.duration:.0f}s, {len(asset.transcript)} segments)")

    report: dict[str, Any] = {
        "dataset": dataset.name,
        "video": str(dataset.video),
        "video_title": asset.display_name,
        "video_duration_s": round(asset.duration, 2),
        "transcript_segments": len(asset.transcript),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "questions": len(dataset.questions),
        "methods": {},
    }

    for method in methods:
        notify(f"=== method: {method} ===")
        rows: list[dict[str, Any]] = []

        agent: SkimAgent | None = None
        if method in AGENT_ABLATION_FLAGS:
            agent = SkimAgent(client, asset, _agent_config(method, agent_options or {}))

        for question in dataset.questions:
            text = question["question"]
            notify(f"[{method}] {text[:80]}")
            started = time.perf_counter()

            if method in AGENT_ABLATION_FLAGS:
                assert agent is not None
                result = agent.run(text).as_dict()
            elif method == "subtitle_rag":
                result = run_subtitle_rag(asset, client, text)
            elif method == "fixed_fps":
                result = run_fixed_fps(asset, client, text, interval_s=fps_interval_s)
            else:  # pragma: no cover - guarded above
                raise ValueError(method)

            elapsed = time.perf_counter() - started
            row: dict[str, Any] = {
                "id": question.get("id"),
                "question": text,
                "gold_answer": question.get("gold_answer", ""),
                "prediction": result.get("answer", ""),
                "stop_reason": result.get("stop_reason"),
                "steps": result.get("steps"),
                "used_timestamps": result.get("used_timestamps", []),
                "metrics": result.get("metrics", {}),
                "runtime_s": round(elapsed, 2),
                "config": result.get("config", method),
            }
            row["temporal_hit"] = temporal_hit(
                result.get("used_timestamps", []), question.get("gold_window")
            )
            if judge:
                verdict = judge_answer(
                    client, text, result.get("answer", ""), question.get("gold_answer", "")
                )
                row["judge"] = verdict.as_dict()
                notify(f"    -> score={verdict.score:.2f} ({verdict.reason[:60]})")
            rows.append(row)

        report["methods"][method] = {
            "description": METHODS[method],
            "summary": aggregate(rows),
            "rows": rows,
        }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
        notify(f"Report written to {output_dir}")

    return report


def render_markdown(report: dict[str, Any]) -> str:
    """Human-readable comparison table."""
    lines = [
        f"# Skim evaluation - {report.get('dataset', '')}",
        "",
        f"- video: `{report.get('video')}` ({report.get('video_duration_s')}s, "
        f"{report.get('transcript_segments')} transcript segments)",
        f"- questions: {report.get('questions')}",
        f"- created: {report.get('created_at')}",
        "",
        "## Summary",
        "",
        "| method | accuracy | answer score | frames sent | VLM calls | visual tokens | coverage | latency (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for method, payload in report.get("methods", {}).items():
        summary = payload.get("summary", {})
        lines.append(
            "| {method} | {acc} | {score} | {frames} | {vlm} | {vtok} | {cov} | {lat} |".format(
                method=method,
                acc=_pct(summary.get("accuracy")),
                score=_num(summary.get("answer_score")),
                frames=_num(summary.get("frames_sent")),
                vlm=_num(summary.get("vlm_calls")),
                vtok=_num(summary.get("visual_tokens")),
                cov=_pct(summary.get("coverage_ratio")),
                lat=_num(summary.get("wall_latency_s")),
            )
        )

    hit_rows = [
        (method, payload.get("summary", {}).get("temporal_hit_rate"))
        for method, payload in report.get("methods", {}).items()
        if payload.get("summary", {}).get("temporal_hit_rate") is not None
    ]
    if hit_rows:
        lines += ["", "### Localisation", "", "| method | looked inside the reference window |", "| --- | --- |"]
        for method, rate in hit_rows:
            lines.append(f"| {method} | {_pct(rate)} |")

    lines += ["", "## Per-question detail", ""]
    for method, payload in report.get("methods", {}).items():
        lines += [f"### {method}", "", payload.get("description", ""), ""]
        lines += ["| id | question | correct | score | stop reason | steps | frames |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for row in payload.get("rows", []):
            judge = row.get("judge", {})
            correct = "yes" if judge.get("correct") else "no"
            lines.append(
                f"| {row.get('id', '')} | {row.get('question', '')[:70]} | {correct} | "
                f"{_num(judge.get('score'))} | {row.get('stop_reason', '')} | {row.get('steps', '')} | "
                f"{_num(row.get('metrics', {}).get('frames_sent'))} |"
            )
        lines.append("")
    return "\n".join(lines)


def _num(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _pct(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"