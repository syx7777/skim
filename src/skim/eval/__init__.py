"""Evaluation: metrics, baselines and the ablation runner."""

from .baselines import BASELINES, run_fixed_fps, run_subtitle_rag
from .metrics import aggregate, judge_answer, temporal_hit
from .runner import METHODS, Dataset, render_markdown, run_evaluation

__all__ = [
    "BASELINES",
    "run_fixed_fps",
    "run_subtitle_rag",
    "aggregate",
    "judge_answer",
    "temporal_hit",
    "METHODS",
    "Dataset",
    "render_markdown",
    "run_evaluation",
]