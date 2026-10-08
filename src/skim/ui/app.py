"""Gradio web UI: upload a lecture, index it, then ask questions."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from ..agent.agent import AgentConfig, SkimAgent
from ..config import Settings, apply_proxy, get_settings
from ..indexing.chapters import build_chapters
from ..llm.client import LLMClient
from ..logging_utils import get_logger, setup_logging
from ..media.download import is_url
from ..media.ingest import ingest
from ..media.transcript import format_timestamp

logger = get_logger("ui")


def _as_path(value: Any) -> str | None:
    """Normalise whatever a Gradio file component hands back into a path."""
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        return value.get("path") or value.get("name")
    return getattr(value, "name", None)


def _empty_state() -> dict[str, Any]:
    return {"asset": None, "agent": None, "client": None, "chapters": []}


def build_app(settings: Settings | None = None):
    """Construct the Gradio Blocks app."""
    import gradio as gr

    settings = settings or get_settings()
    setup_logging(settings.log_level)
    apply_proxy(settings)

    # ------------------------------------------------------------------ handlers
    def process(source_value, subtitle_value, max_height, state):
        state = state or _empty_state()
        source = (source_value or "").strip()
        if not source:
            return (
                state,
                "**Paste a lecture link first** "
                "(e.g. a Bilibili or YouTube URL), or a local file path.",
                [],
                None,
            )
        if not is_url(source) and not Path(source).is_file():
            return state, f"**Not a valid link or local file:** `{source}`", [], None

        subtitle_path = _as_path(subtitle_value)
        try:
            asset = ingest(
                source,
                settings=settings,
                subtitle=subtitle_path,
                max_height=int(max_height) if max_height else None,
            )
        except Exception as exc:
            logger.exception("ingest failed")
            return state, f"**Ingest failed:** {exc}", [], None

        client = state.get("client") or LLMClient(settings)
        agent = SkimAgent(client, asset, AgentConfig())

        chapters: list[dict[str, Any]] = []
        chapter_note = "_(chaptering skipped)_"
        try:
            built = build_chapters(asset, client)
            chapters = [
                [c.index + 1, format_timestamp(c.start), format_timestamp(c.end), c.title, c.summary]
                for c in built
            ]
            chapter_note = f"{len(built)} chapters"
        except Exception as exc:  # pragma: no cover - network path
            logger.warning("chaptering failed: %s", exc)
            chapter_note = f"chaptering failed: {exc}"

        state.update({"asset": asset, "agent": agent, "client": client, "chapters": chapters})
        source_line = (
            f"- source: [{asset.title or 'link'}]({asset.source})\n" if asset.is_remote else ""
        )
        status = (
            f"**Ready.** {asset.display_name}  \n"
            + source_line
            + f"- duration: `{format_timestamp(asset.duration)}` ({asset.duration:.0f}s)\n"
            f"- resolution: `{asset.info.width}x{asset.info.height}` @ {asset.info.fps:.2f} fps\n"
            f"- transcript: `{len(asset.transcript)}` segments ({asset.transcript.source}, "
            f"lang={asset.transcript.language})\n"
            f"- {chapter_note}\n"
            f"- local file: `{asset.path}`"
        )
        return state, status, chapters, str(asset.path)

    def ask(question, max_steps, use_adaptive, use_memory, use_uncertainty, state):
        state = state or _empty_state()
        agent: SkimAgent | None = state.get("agent")
        if agent is None:
            return "**Index a video first (step 1).**", [], {}, ""
        if not question or not question.strip():
            return "**Enter a question.**", [], {}, ""

        agent.config = AgentConfig(
            max_steps=int(max_steps),
            adaptive_observation=bool(use_adaptive),
            use_temporal_memory=bool(use_memory),
            use_uncertainty=bool(use_uncertainty),
        )
        try:
            result = agent.run(question)
        except Exception as exc:
            logger.exception("agent run failed")
            return f"**Run failed:** {exc}", [], {}, ""

        answer_md = (
            f"### Answer\n\n{result.answer or '_(empty)_'}\n\n"
            f"**confidence** `{result.confidence:.2f}` &nbsp;|&nbsp; "
            f"**stop reason** `{result.stop_reason}` &nbsp;|&nbsp; **steps** `{result.steps}`"
        )
        if result.remaining_uncertainty:
            answer_md += f"\n\n> remaining uncertainty: {result.remaining_uncertainty}"

        evidence = [
            [
                format_timestamp(float(item["timestamp"])) if isinstance(item.get("timestamp"), (int, float)) else "-",
                str(item.get("quote", "")),
            ]
            for item in result.evidence
        ]

        metrics = dict(result.metrics)
        trace_md = _render_trace(result)

        return answer_md, evidence, metrics, trace_md

    # -------------------------------------------------------------------- layout
    with gr.Blocks(title="Skim") as app_ui:
        gr.Markdown(
            "# Skim\n"
            "Paste a **lecture link** (Bilibili / YouTube) and ask questions about it. "
            "The agent decides *where* to look instead of watching everything: cheap "
            "transcript search first, then vision only on the windows that matter."
        )
        state = gr.State(_empty_state())

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Load a lecture")
                source_in = gr.Textbox(
                    label="Lecture link or local path",
                    placeholder="https://www.bilibili.com/video/BV.../?p=106",
                    lines=2,
                )
                gr.Markdown(
                    "<small>Paste a Bilibili / YouTube link and the engine downloads it "
                    "(720p by default, cached under <code>data/downloads/</code>). "
                    "A local file path also works.</small>",
                    sanitize_html=False,
                )
                subtitle_in = gr.File(
                    label="Optional transcript (.srt / .vtt / .json) — skips ASR",
                    file_types=[".srt", ".vtt", ".json"],
                )
                with gr.Accordion("Download settings", open=False):
                    max_height_in = gr.Dropdown(
                        choices=[360, 480, 720, 1080],
                        value=720,
                        label="Max video height",
                        info="720p keeps blackboard handwriting and formulas legible.",
                    )
                process_btn = gr.Button("Download & index", variant="primary")
                status_out = gr.Markdown()
                preview_out = gr.Video(label="Resolved video", interactive=False, height=240)
                chapters_out = gr.Dataframe(
                    headers=["#", "start", "end", "title", "summary"],
                    label="Table of contents",
                    wrap=True,
                    interactive=False,
                )

            with gr.Column(scale=2):
                gr.Markdown("### 2. Ask")
                question_in = gr.Textbox(
                    label="Question",
                    placeholder="老师在哪一段讲了 GRPO 和 PPO 的区别？",
                    lines=2,
                )
                with gr.Accordion("Agent settings / ablations", open=False):
                    max_steps_in = gr.Slider(1, 16, value=8, step=1, label="Max steps")
                    adaptive_in = gr.Checkbox(value=True, label="Adaptive observation (relevance-driven frame density)")
                    memory_in = gr.Checkbox(value=True, label="Temporal Memory (avoid re-inspecting)")
                    uncertainty_in = gr.Checkbox(value=True, label="Uncertainty-based early stopping")
                ask_btn = gr.Button("Ask", variant="primary")

                answer_out = gr.Markdown()
                evidence_out = gr.Dataframe(
                    headers=["time", "evidence"], label="Evidence", wrap=True, interactive=False
                )
                with gr.Accordion("Cost & behaviour", open=False):
                    metrics_out = gr.JSON(label="Metrics")
                    trace_out = gr.Markdown()

        process_btn.click(
            process,
            inputs=[source_in, subtitle_in, max_height_in, state],
            outputs=[state, status_out, chapters_out, preview_out],
        )
        ask_btn.click(
            ask,
            inputs=[question_in, max_steps_in, adaptive_in, memory_in, uncertainty_in, state],
            outputs=[answer_out, evidence_out, metrics_out, trace_out],
        )
        question_in.submit(
            ask,
            inputs=[question_in, max_steps_in, adaptive_in, memory_in, uncertainty_in, state],
            outputs=[answer_out, evidence_out, metrics_out, trace_out],
        )

    return app_ui


def _render_trace(result) -> str:
    """Render the agent's decision trace as a compact markdown list."""
    rows = [
        "| step | action | cost | detail |",
        "| --- | --- | --- | --- |",
    ]
    for entry in result.trace:
        detail = html.escape(str(entry.get("summary", "")).replace("\n", " ")[:160])
        rows.append(
            f"| {entry.get('step', '')} | `{entry.get('action', '')}` | {entry.get('cost', '')} | {detail} |"
        )
    memory_stats = result.metrics.get("memory_stats", {})
    footer = (
        f"\n\n**coverage** {memory_stats.get('coverage_ratio', 0) * 100:.1f}% &nbsp;|&nbsp; "
        f"**frames** {memory_stats.get('frames_viewed', 0)} &nbsp;|&nbsp; "
        f"**VLM calls** {memory_stats.get('vlm_looks', 0)} &nbsp;|&nbsp; "
        f"**distinct intervals** {memory_stats.get('checked_intervals', 0)}"
    )
    return "\n".join(rows) + footer


if __name__ == "__main__":  # pragma: no cover
    build_app().launch()