"""Synthetic lecture generator for a zero-setup demo and reproducible evaluation.

Produces:

* ``sample_lecture.mp4`` -- a slide-based lecture video,
* ``sample_lecture.transcript.json`` -- the matching transcript (so the demo does
  not require running ASR first),
* ``sample_lecture.dataset.json`` -- questions with reference answers and
  reference time windows.

The narration deliberately says *less* than the slides for the formula, table and
code sections. That is what makes the demo meaningful: those questions cannot be
answered from the transcript, so the agent has to decide to look at the video.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .config import Settings, apply_proxy, get_settings
from .logging_utils import get_logger
from .media import ffmpeg

logger = get_logger("demo")

WIDTH, HEIGHT = 1280, 720
FPS = 2

#: (duration_s, slide title, bullet lines, narration spoken by the teacher)
SLIDES: list[tuple[float, str, list[str], str]] = [
    (
        24.0,
        "强化学习中的策略优化",
        ["本节目标：理解 PPO 与 GRPO 的区别", "从 on-policy 目标函数开始", "关注实现代价与稳定性"],
        "同学们好，这一节我们讲强化学习里的策略优化。我们会先快速回顾 PPO，然后重点介绍 GRPO，"
        "也就是 Group Relative Policy Optimization。最后我们比较两者的区别，"
        "并看看在实际实现中的代价。",
    ),
    (
        26.0,
        "回顾 PPO",
        ["目标函数：L = E[ min(r*A, clip(r,1-e,1+e)*A) ]", "需要一个 critic 估计优势 A", "用 clip 限制策略更新幅度"],
        "我们先回顾 PPO。PPO 的目标函数包含一个重要性采样比率 r，以及用 clip 把更新限制在一个信任域里。"
        "这里的优势函数 A 通常由单独的 critic 网络，也就是价值网络来估计。"
        "clip 的作用是防止一次更新把策略带得太远。",
    ),
    (
        26.0,
        "PPO 的痛点",
        ["critic 与 policy 同规模，显存翻倍", "优势估计方差大、训练不稳定", "超参敏感：clip 系数、GAE lambda"],
        "PPO 的问题主要有三个。第一，critic 网络和策略网络规模相当，显存和计算开销基本翻倍。"
        "第二，优势估计的方差比较大，训练容易不稳定。第三，它对超参数比较敏感。"
        "这些都是我们希望改进的地方。",
    ),
    (
        26.0,
        "GRPO 的核心思想",
        ["采样一组回答（group），而不是单个样本", "用组内均值/标准差归一化得到优势", "不再需要 critic 网络"],
        "GRPO 的思路是这样的：对同一个问题采样一组回答，而不是单个回答，"
        "然后用这一组内部的均值和标准差做归一化，直接得到优势估计。"
        "这样我们就不需要额外的 critic 网络了，显存开销明显下降。",
    ),
    (
        26.0,
        "GRPO 的目标函数",
        [
            "A_i = (r_i - mean(r)) / (std(r) + eps)",
            "J = E[ min(rho_i*A_i, clip(rho_i,1-e,1+e)*A_i) ] - beta*KL(pi||pi_ref)",
            "KL 项使用无偏估计，beta 控制偏离参考策略的程度",
        ],
        "这里是 GRPO 的目标函数。大家看这张幻灯片，"
        "优势是通过组内归一化得到的，KL 项和 beta 的作用和刚才说的一样。"
        "具体的符号我们就不逐个念了，大家对照公式理解一下。",
    ),
    (
        26.0,
        "GRPO vs PPO",
        ["PPO：actor + critic + reward model", "GRPO：actor + reward model（无 critic）", "GRPO：优势来自组内比较，方差不依赖 value 估计"],
        "这张表把 GRPO 和 PPO 放在一起对比。大家注意看最后一行的区别，"
        "这是两者最本质的不同。这一页建议大家截图保存，复习的时候对照着看。",
    ),
    (
        26.0,
        "代码实现片段",
        [
            "loss = -(torch.min(ratio*adv, clamp(ratio)*adv)).mean()",
            "adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)",
            "loss += beta * kl_divergence",
        ],
        "这是实现层面的代码片段，大概十几行。"
        "关键就是这两行，第一行是优势的归一化，第二行是带 clip 的策略损失。"
        "大家自己动手写一遍会更清楚。",
    ),
    (
        24.0,
        "小结与实验建议",
        ["GRPO 用组内归一化替代 critic", "显存与实现复杂度都更低", "建议在相同 reward 设置下对比两者"],
        "最后做个小结。GRPO 用组内归一化替代了 critic，显存和实现复杂度都更低。"
        "建议大家在自己的任务上，用相同的 reward 设置对比一下这两种方法。这一节就到这里，谢谢大家。",
    ),
]

QUESTIONS: list[dict] = [
    {
        "id": "q1",
        "question": "老师在哪一段讲了 GRPO 和 PPO 的区别？区别的核心是什么？",
        "gold_answer": (
            "在对比表格那一段（约 02:10-02:36）。核心区别是：PPO 需要额外的 critic 网络来估计优势，"
            "而 GRPO 不需要 critic，优势来自同一组回答内部的归一化比较，因此显存和实现复杂度更低。"
        ),
        "gold_window": [130.0, 156.0],
        "requires_vision": False,
    },
    {
        "id": "q2",
        "question": "GRPO 的目标函数里，优势 A_i 具体是怎么算出来的？",
        "gold_answer": (
            "A_i = (r_i - mean(r)) / (std(r) + eps)，"
            "即用同一组回答的奖励减去组内均值，再除以组内标准差（加 eps 防止除零）。"
        ),
        "gold_window": [112.0, 138.0],
        "requires_vision": True,
    },
    {
        "id": "q3",
        "question": "PPO 为什么要用 clip？",
        "gold_answer": (
            "用来限制一次策略更新的幅度，把新策略约束在旧策略附近的信任域内，"
            "避免更新过大导致训练不稳定。"
        ),
        "gold_window": [28.0, 54.0],
        "requires_vision": False,
    },
    {
        "id": "q4",
        "question": "课程里给出的 GRPO 损失函数代码是怎么写的？",
        "gold_answer": (
            "先做优势归一化 adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)，"
            "损失为 loss = -(torch.min(ratio*adv, clamp(ratio)*adv)).mean()，"
            "再加上 beta * kl_divergence 这一项。"
        ),
        "gold_window": [164.0, 190.0],
        "requires_vision": True,
    },
    {
        "id": "q5",
        "question": "PPO 相比 GRPO 在训练代价上多出了什么？",
        "gold_answer": (
            "多出了一个与策略网络同规模的 critic（价值网络），显存和计算开销基本翻倍，"
            "而且优势估计方差大、对超参更敏感。"
        ),
        "gold_window": [54.0, 80.0],
        "requires_vision": False,
    },
]


# ------------------------------------------------------------------- rendering
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhl.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:  # pragma: no cover
                continue
    logger.warning("no CJK-capable font found; slide text may not render correctly")
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    """Character-based wrapping that works for CJK (no spaces)."""
    lines: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = char
    if current:
        lines.append(current)
    return lines


@dataclass
class SlidePlan:
    index: int
    start: float
    end: float
    title: str
    bullets: list[str]
    narration: str


def plan_slides() -> list[SlidePlan]:
    plans: list[SlidePlan] = []
    cursor = 0.0
    for index, (duration, title, bullets, narration) in enumerate(SLIDES, start=1):
        plans.append(
            SlidePlan(
                index=index,
                start=cursor,
                end=cursor + duration,
                title=title,
                bullets=bullets,
                narration=narration,
            )
        )
        cursor += duration
    return plans


def render_slide(plan: SlidePlan, out_path: Path) -> Path:
    """Draw one 1280x720 slide."""
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)

    title_font = _load_font(44)
    body_font = _load_font(30)
    footer_font = _load_font(22)

    # header band
    draw.rectangle([0, 0, WIDTH, 96], fill=(24, 38, 66))
    draw.text((48, 26), plan.title, font=title_font, fill="white")
    draw.text((WIDTH - 260, 36), f"Slide {plan.index}/{len(SLIDES)}", font=footer_font, fill=(180, 196, 220))

    # bullets
    y = 150
    for bullet in plan.bullets:
        draw.ellipse([56, y + 12, 70, y + 26], fill=(24, 38, 66))
        for line in _wrap(draw, bullet, body_font, WIDTH - 180):
            draw.text((92, y), line, font=body_font, fill=(28, 28, 32))
            y += 42
        y += 14

    draw.text((48, HEIGHT - 46), f"t = {plan.start:.0f}s", font=footer_font, fill=(120, 128, 140))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path


# -------------------------------------------------------------------- assembly
def _build_video(plans: list[SlidePlan], slide_paths: list[Path], out_path: Path) -> Path:
    """Concatenate slides into an H.264 mp4 using the bundled ffmpeg."""
    concat_file = out_path.parent / "slides.txt"
    # The concat demuxer resolves relative paths against the list file's own
    # directory, so absolute paths are the only reliable option here.
    absolute_slides = [slide.resolve().as_posix() for slide in slide_paths]
    lines: list[str] = []
    for plan, slide in zip(plans, absolute_slides):
        lines.append(f"file '{slide}'")
        lines.append(f"duration {plan.end - plan.start:.3f}")
    lines.append(f"file '{absolute_slides[-1]}'")  # concat quirk: repeat last
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_args = [
        "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-r", str(FPS),
        "-pix_fmt", "yuv420p",
    ]
    for codec in ("libx264", "mpeg4"):
        try:
            ffmpeg.run_ffmpeg([*base_args, "-c:v", codec, "-preset", "veryfast", str(out_path)])
            logger.info("encoded %s with %s", out_path.name, codec)
            return out_path
        except ffmpeg.FFmpegError as exc:
            logger.warning("codec %s failed (%s); trying next", codec, exc)
    raise ffmpeg.FFmpegError("could not encode the sample video with any available codec")


def _build_transcript(plans: list[SlidePlan], out_path: Path) -> Path:
    payload = {
        "language": "zh",
        "source": "manual",
        "duration": plans[-1].end,
        "segments": [
            {
                "start": round(plan.start, 3),
                "end": round(plan.end, 3),
                "text": plan.narration,
                "speaker": "teacher",
            }
            for plan in plans
        ],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def build_sample(
    out_dir: str | Path,
    *,
    settings: Settings | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    """Generate the sample video, transcript and dataset.

    Idempotent by default: existing assets are reused so that re-running the
    command does not change the file mtime and invalidate the ingest cache.

    Returns ``(video_path, dataset_path)``.
    """
    settings = settings or get_settings()
    apply_proxy(settings)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = out_dir / "sample_lecture.mp4"
    transcript_path = out_dir / "sample_lecture.transcript.json"
    dataset_path = out_dir / "sample_lecture.dataset.json"

    if not force and video_path.is_file() and transcript_path.is_file() and dataset_path.is_file():
        logger.info("reusing existing sample assets in %s", out_dir)
        return video_path, dataset_path

    plans = plan_slides()
    slide_dir = out_dir / "slides"
    slide_paths = [render_slide(plan, slide_dir / f"slide_{plan.index:02d}.png") for plan in plans]

    _build_video(plans, slide_paths, video_path)
    _build_transcript(plans, transcript_path)

    dataset_path.write_text(
        json.dumps(
            {
                "name": "synthetic-grpo-vs-ppo",
                "video": video_path.name,
                "subtitle": transcript_path.name,
                "questions": QUESTIONS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("sample assets written to %s", out_dir)
    return video_path, dataset_path