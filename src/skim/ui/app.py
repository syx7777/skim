"""Gradio web UI: load a lecture, then ask questions about it."""

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

# Gradio 6 wants theme/css at ``launch()`` time, so they live here and are
# handed over by the CLI entry point (``skim ui``).
CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
footer { display: none !important; }

#skim-hero {
    text-align: center;
    padding: 26px 16px 14px;
}
#skim-hero h1 {
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 0 0 8px 0;
}
#skim-hero p {
    color: var(--body-text-color-subdued, #6b7280);
    font-size: 14px;
    line-height: 1.6;
    max-width: 760px;
    margin: 0 auto;
}

.skim-card {
    background: var(--block-background-fill, #ffffff);
    border: 1px solid var(--border-color-primary, #e5e7eb);
    border-radius: 16px;
    padding: 8px 18px 16px 18px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
}
.skim-step-title {
    border-left: 3px solid var(--color-accent, #4f46e5);
    padding-left: 10px;
    margin: 10px 0 14px 0;
    font-size: 15px;
    font-weight: 600;
    line-height: 1.4;
}
.skim-hint {
    color: var(--body-text-color-subdued, #6b7280);
    font-size: 12px;
    line-height: 1.55;
}
.skim-hint code {
    background: var(--color-accent-soft, #eef2ff);
    border-radius: 5px;
    padding: 0 4px;
}
"""

HERO = (
    "# Skim\n"
    "**Skim a long lecture instead of watching all of it.** Paste a Bilibili / YouTube "
    "link and ask questions — the agent decides *where* to look: cheap transcript search "
    "first, vision only on the windows that matter."
)


def build_theme():
    """Soft, low-contrast theme; system fonts only so it works offline."""
    import gradio as gr

    return gr.themes.Soft(
        primary_hue="indigo",
        neutral_hue="slate",
        radius_size="lg",
        font=["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        font_mono=["ui-monospace", "SFMono-Regular", "Consolas", "monospace"],
    )


def _empty_state() -> dict[str, Any]:
    return {"asset": None, "agent": None, "client": None, "chapters": []}


def build_app(settings: Settings | None = None):
    """Construct the Gradio Blocks app."""
    import gradio as gr

    settings = settings or get_settings()
    setup_logging(settings.log_level)
    apply_proxy(settings)

    # ------------------------------------------------------------------ handlers
    def process(source_value, max_height, state):
        state = state or _empty_state()
        source = (source_value or "").strip()
        if not source:
            return (
                state,
                "**Paste a lecture link first** "
                "(e.g. a Bilibili or YouTube URL), or a local file path.",
                [],
                None,
                None,
            )
        if not is_url(source) and not Path(source).is_file():
            return state, f"**Not a valid link or local file:** `{source}`", [], None, None

        try:
            asset = ingest(
                source,
                settings=settings,
                max_height=int(max_height) if max_height else None,
            )
        except Exception as exc:
            logger.exception("ingest failed")
            return state, f"**Ingest failed:** {exc}", [], None, None

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

        # Expose the produced transcript (from ASR or cache) as a downloadable .srt.
        transcript_file: Path | None = None
        transcript_note = ""
        if len(asset.transcript) > 0:
            candidate = asset.cache_dir / "transcript.srt"
            try:
                asset.transcript.save_srt(candidate)
                transcript_file = candidate
            except OSError as exc:  # pragma: no cover - disk failure
                logger.warning("could not write transcript: %s", exc)
        else:
            transcript_note = " — no speech detected, nothing to download"

        state.update({"asset": asset, "agent": agent, "client": client, "chapters": chapters})
        source_line = (
            f"- source: [{asset.title or 'link'}]({asset.source})\n" if asset.is_remote else ""
        )
        status = (
            f"**Ready — {asset.display_name}**  \n"
            + source_line
            + f"- duration: `{format_timestamp(asset.duration)}` ({asset.duration:.0f}s)\n"
            f"- resolution: `{asset.info.width}x{asset.info.height}` @ {asset.info.fps:.2f} fps\n"
            f"- transcript: `{len(asset.transcript)}` segments "
            f"({asset.transcript.source}, lang={asset.transcript.language}){transcript_note}\n"
            f"- {chapter_note}\n"
            f"- local file: `{asset.path}`"
        )
        return (
            state,
            status,
            chapters,
            str(asset.path),
            str(transcript_file) if transcript_file else None,
        )

    def ask(question, max_steps, use_adaptive, use_memory, use_uncertainty, state):
        state = state or _empty_state()
        agent: SkimAgent | None = state.get("agent")
        if agent is None:
            return "**Load a lecture first (step 1).**", [], {}, ""
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
        gr.Markdown(HERO, elem_id="skim-hero")
        state = gr.State(_empty_state())

        with gr.Row():
            # ---------------------------------------------------------- step 1
            with gr.Column(scale=1, elem_classes="skim-card"):
                gr.Markdown(
                    "<div class='skim-step-title'>1 · Load a lecture</div>",
                    sanitize_html=False,
                )
                source_in = gr.Textbox(
                    label="Lecture link or local path",
                    placeholder="https://www.bilibili.com/video/BV.../?p=106",
                    lines=2,
                )
                gr.Markdown(
                    "<small class='skim-hint'>Paste a Bilibili / YouTube link and the engine "
                    "downloads it (720p by default, cached under <code>data/downloads/</code>), "
                    "transcribes it and builds a table of contents. A local file path also works."
                    "</small>",
                    sanitize_html=False,
                )
                with gr.Accordion("Download settings", open=False):
                    max_height_in = gr.Dropdown(
                        choices=[360, 480, 720, 1080],
                        value=720,
                        label="Max video height",
                        info="720p keeps blackboard handwriting and formulas legible.",
                    )
                download_btn = gr.Button("Download", variant="primary")

                status_out = gr.Markdown()
                transcript_out = gr.File(
                    label="ASR transcript (.srt) — click to download",
                    interactive=False,
                )
                preview_out = gr.Video(label="Resolved video", interactive=False, height=220)
                with gr.Accordion("Table of contents", open=False):
                    chapters_out = gr.Dataframe(
                        headers=["#", "start", "end", "title", "summary"],
                        column_widths=["6%", "12%", "12%", "28%", "42%"],
                        wrap=True,
                        interactive=False,
                    )

            # ---------------------------------------------------------- step 2
            with gr.Column(scale=2, elem_classes="skim-card"):
                gr.Markdown(
                    "<div class='skim-step-title'>2 · Ask</div>",
                    sanitize_html=False,
                )
                question_in = gr.Textbox(
                    label="Question",
                    placeholder="老师在哪一段讲了 GRPO 和 PPO 的区别？",
                    lines=2,
                )
                with gr.Accordion("Agent settings / ablations", open=False):
                    max_steps_in = gr.Slider(1, 16, value=8, step=1, label="Max steps")
                    adaptive_in = gr.Checkbox(
                        value=True, label="Adaptive observation (relevance-driven frame density)"
                    )
                    memory_in = gr.Checkbox(
                        value=True, label="Temporal Memory (avoid re-inspecting)"
                    )
                    uncertainty_in = gr.Checkbox(
                        value=True, label="Uncertainty-based early stopping"
                    )
                ask_btn = gr.Button("Ask", variant="primary")

                answer_out = gr.Markdown()
                evidence_out = gr.Dataframe(
                    headers=["time", "evidence"], label="Evidence", wrap=True, interactive=False
                )
                with gr.Accordion("Cost & behaviour", open=False):
                    metrics_out = gr.JSON(label="Metrics")
                    trace_out = gr.Markdown()

        process_btn_inputs = [source_in, max_height_in, state]
        process_btn_outputs = [state, status_out, chapters_out, preview_out, transcript_out]
        download_btn.click(process, inputs=process_btn_inputs, outputs=process_btn_outputs)

        ask_inputs = [question_in, max_steps_in, adaptive_in, memory_in, uncertainty_in, state]
        ask_outputs = [answer_out, evidence_out, metrics_out, trace_out]
        ask_btn.click(ask, inputs=ask_inputs, outputs=ask_outputs)
        question_in.submit(ask, inputs=ask_inputs, outputs=ask_outputs)

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
    build_app().launch(theme=build_theme(), css=CSS)