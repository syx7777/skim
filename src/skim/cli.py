"""Command line interface (``skim``)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agent.agent import AgentConfig, SkimAgent
from .config import apply_proxy, get_layout, get_settings
from .eval.runner import Dataset, METHODS, run_evaluation
from .indexing.chapters import build_chapters
from .llm.client import LLMClient
from .logging_utils import get_logger, setup_logging
from .media.ingest import ingest as ingest_video
from .media.transcript import format_timestamp

app = typer.Typer(
    add_completion=False,
    help="Skim - ask questions about long lecture videos.",
    no_args_is_help=True,
)
console = Console()
logger = get_logger("cli")


def _bootstrap(level: str = "INFO"):
    settings = get_settings()
    setup_logging(level)
    apply_proxy(settings)
    return settings


# --------------------------------------------------------------------- ingest
@app.command()
def ingest(
    source: str = typer.Argument(
        ...,
        help="Lecture video link (Bilibili / YouTube / ...) or a local file path.",
    ),
    subtitle: Optional[Path] = typer.Option(None, "--subtitle", "-s", help="Optional .srt/.vtt/.json transcript."),
    max_height: Optional[int] = typer.Option(
        None, "--max-height", help="Height cap when downloading a link (default from SKIM_VIDEO_MAX_HEIGHT)."
    ),
    force: bool = typer.Option(False, "--force", help="Ignore caches: re-download and re-run ASR."),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Download (if a link) / transcribe / index a lecture, cached for later questions."""
    settings = _bootstrap(log_level)
    asset = ingest_video(
        source, settings=settings, subtitle=subtitle, force=force, max_height=max_height
    )
    table = Table(title="Video ready")
    table.add_column("field")
    table.add_column("value")
    table.add_row("title", asset.display_name)
    if asset.is_remote:
        table.add_row("source", asset.source)
    table.add_row("video id", asset.video_id)
    table.add_row("local file", str(asset.path))
    table.add_row("duration", f"{format_timestamp(asset.duration)} ({asset.duration:.0f}s)")
    table.add_row("resolution", f"{asset.info.width}x{asset.info.height} @ {asset.info.fps:.2f} fps")
    table.add_row("transcript source", asset.transcript.source)
    table.add_row("transcript segments", str(len(asset.transcript)))
    table.add_row("language", asset.transcript.language)
    table.add_row("cache dir", str(asset.cache_dir))
    console.print(table)


# ------------------------------------------------------------------------ ask
@app.command()
def ask(
    source: str = typer.Argument(
        ..., help="Lecture video link (Bilibili / YouTube / ...) or a local file path."
    ),
    question: str = typer.Argument(..., help="Question to answer."),
    subtitle: Optional[Path] = typer.Option(None, "--subtitle", "-s"),
    max_height: Optional[int] = typer.Option(None, "--max-height", help="Height cap when downloading a link."),
    max_steps: int = typer.Option(8, "--max-steps", help="Agent step budget."),
    max_frames: int = typer.Option(4, "--max-frames", help="Frames per inspect_video call."),
    min_confidence: float = typer.Option(0.7, "--min-confidence", help="Early-stop threshold."),
    no_adaptive: bool = typer.Option(False, "--no-adaptive", help="Ablation: fixed frame density."),
    no_memory: bool = typer.Option(False, "--no-memory", help="Ablation: no Temporal Memory."),
    no_uncertainty: bool = typer.Option(False, "--no-uncertainty", help="Ablation: never stop early."),
    json_out: Optional[Path] = typer.Option(None, "--json-out", help="Write the full run to a JSON file."),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Ask one question about a lecture (given as a link or a local file)."""
    settings = _bootstrap(log_level)

    asset = ingest_video(source, settings=settings, subtitle=subtitle, max_height=max_height)
    client = LLMClient(settings)
    config = AgentConfig(
        max_steps=max_steps,
        max_frames_per_call=max_frames,
        min_confidence=min_confidence,
        adaptive_observation=not no_adaptive,
        use_temporal_memory=not no_memory,
        use_uncertainty=not no_uncertainty,
    )
    agent = SkimAgent(client, asset, config)
    with console.status("Agent is searching the video ..."):
        result = agent.run(question)

    console.print(Panel(result.answer or "(no answer)", title=f"Answer (confidence {result.confidence:.2f})"))

    if result.evidence:
        evidence = Table(title="Evidence", show_lines=True)
        evidence.add_column("time")
        evidence.add_column("quote", overflow="fold")
        for item in result.evidence:
            stamp = item.get("timestamp")
            evidence.add_row(
                format_timestamp(float(stamp)) if isinstance(stamp, (int, float)) else "-",
                str(item.get("quote", "")),
            )
        console.print(evidence)

    metrics = Table(title="Cost & behaviour", show_header=False)
    metrics.add_column("metric")
    metrics.add_column("value")
    for key, value in result.metrics.items():
        if key == "memory_stats":
            continue
        metrics.add_row(key, str(value))
    console.print(metrics)

    console.print(f"[dim]stop reason: {result.stop_reason} | steps: {result.steps} | config: {result.config_label}[/dim]")
    if result.remaining_uncertainty:
        console.print(f"[yellow]remaining uncertainty: {result.remaining_uncertainty}[/yellow]")

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(agent.to_json(result), encoding="utf-8")
        console.print(f"[green]written {json_out}[/green]")


# ------------------------------------------------------------------- chapters
@app.command()
def chapters(
    source: str = typer.Argument(
        ..., help="Lecture video link (Bilibili / YouTube / ...) or a local file path."
    ),
    subtitle: Optional[Path] = typer.Option(None, "--subtitle", "-s"),
    max_height: Optional[int] = typer.Option(None, "--max-height", help="Height cap when downloading a link."),
    force: bool = typer.Option(False, "--force", help="Rebuild even if cached."),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Build (and print) the automatic table of contents."""
    settings = _bootstrap(log_level)
    asset = ingest_video(source, settings=settings, subtitle=subtitle, max_height=max_height)
    client = LLMClient(settings)
    items = build_chapters(asset, client, force=force)

    table = Table(title=f"Chapters ({len(items)})")
    table.add_column("#", justify="right")
    table.add_column("start")
    table.add_column("end")
    table.add_column("title")
    table.add_column("summary", overflow="fold")
    for chapter in items:
        table.add_row(
            str(chapter.index + 1),
            format_timestamp(chapter.start),
            format_timestamp(chapter.end),
            chapter.title,
            chapter.summary,
        )
    console.print(table)


# ------------------------------------------------------------------------ eval
@app.command()
def evaluate(
    dataset: Path = typer.Argument(..., exists=True, help="Dataset JSON (see README)."),
    methods: str = typer.Option(
        "agent,subtitle_rag,fixed_fps", "--methods", help=f"Comma separated. Available: {', '.join(METHODS)}"
    ),
    fps_interval: float = typer.Option(30.0, "--fps-interval", help="Sampling interval for the fixed_fps baseline."),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip LLM answer judging."),
    out: Optional[Path] = typer.Option(None, "--out", help="Directory for report.json/report.md."),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run the benchmark (baselines + ablations)."""
    settings = _bootstrap(log_level)
    layout = get_layout(settings)
    output_dir = out or (layout.runs / time.strftime("%Y%m%d-%H%M%S"))

    client = LLMClient(settings)
    selected = [m.strip() for m in methods.split(",") if m.strip()]
    report = run_evaluation(
        Dataset.load(dataset),
        client,
        methods=selected,
        fps_interval_s=fps_interval,
        judge=not no_judge,
        output_dir=output_dir,
        progress=lambda message: console.print(f"[dim]{message}[/dim]"),
    )

    table = Table(title="Summary")
    for column in ("method", "accuracy", "frames", "vlm calls", "visual tokens", "coverage", "latency"):
        table.add_column(column)
    for method, payload in report["methods"].items():
        summary = payload["summary"]
        table.add_row(
            method,
            f"{summary.get('accuracy', 0) * 100:.1f}%",
            f"{summary.get('frames_sent', 0):.1f}",
            f"{summary.get('vlm_calls', 0):.1f}",
            f"{summary.get('visual_tokens', 0):.0f}",
            f"{summary.get('coverage_ratio', 0) * 100:.1f}%",
            f"{summary.get('wall_latency_s', 0):.1f}s",
        )
    console.print(table)
    console.print(f"[green]report: {output_dir / 'report.md'}[/green]")


# --------------------------------------------------------------------- sample
@app.command()
def sample(
    out: Optional[Path] = typer.Option(None, "--out", help="Output directory for the demo assets."),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Generate a synthetic lecture video + QA dataset for a zero-setup demo."""
    _bootstrap(log_level)
    from .demo import build_sample

    settings = get_settings()
    layout = get_layout(settings)
    target = out or layout.assets
    video, dataset = build_sample(target, settings=settings)
    console.print(f"[green]video:[/green] {video}")
    console.print(f"[green]dataset:[/green] {dataset}")
    console.print(f'\nTry: [bold]skim ask "{video}" "老师在哪一段讲了 GRPO 和 PPO 的区别？"[/bold]')


# ------------------------------------------------------------------------- ui
@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7860, "--port"),
    share: bool = typer.Option(False, "--share"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Launch the web UI."""
    _bootstrap(log_level)
    try:
        from .ui.app import build_app
    except ImportError as exc:  # pragma: no cover
        console.print(f"[red]Gradio is not installed: {exc}[/red]")
        raise typer.Exit(code=1)
    build_app().launch(server_name=host, server_port=port, share=share, show_error=True)


# ----------------------------------------------------------------------- info
@app.command()
def info() -> None:
    """Show the resolved configuration (secrets are never printed)."""
    settings = get_settings()
    layout = get_layout(settings)
    table = Table(title="Configuration")
    table.add_column("key")
    table.add_column("value")
    table.add_row("base_url", settings.base_url)
    table.add_row("api_key", settings.masked_key)
    table.add_row("llm_model", settings.llm_model)
    table.add_row("vlm_model", settings.vlm_model)
    table.add_row("proxy", settings.proxy or "(none)")
    table.add_row("data_dir", str(layout.root.resolve()))
    table.add_row("whisper_model", settings.whisper_model)
    table.add_row("max_agent_steps", str(settings.max_agent_steps))
    console.print(table)


@app.command()
def cache() -> None:
    """List ingested videos in the local cache."""
    settings = get_settings()
    layout = get_layout(settings)
    table = Table(title="Cached videos")
    table.add_column("video id")
    table.add_column("name")
    table.add_column("duration")
    table.add_column("segments")
    found = False
    for meta_path in sorted(layout.cache.glob("*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        found = True
        table.add_row(
            meta.get("video_id", ""),
            meta.get("name", ""),
            format_timestamp(meta.get("duration", 0.0)),
            str(meta.get("transcript_segments", 0)),
        )
    if found:
        console.print(table)
    else:
        console.print("[dim]cache is empty[/dim]")


def main() -> None:  # pragma: no cover
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/yellow]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()