<div align="center">

# Skim

**面向长课程视频的「按需观看」问答 Agent**

给一段一两小时的网课链接或本地文件直接提问，Agent 自己判断**该看哪里、看多少、什么时候已经看够了** ——
而不是让 VLM 把整段视频从头看到尾。

[![Python](https://img.shields.io/badge/python-%E2%89%A5%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-2EA043.svg)](#-许可证)
[![uv](https://img.shields.io/badge/packaged%20with-uv-5C5CFF.svg)](https://docs.astral.sh/uv/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-F97316?logo=gradio&logoColor=white)](https://gradio.app/)
[![OpenAI compatible](https://img.shields.io/badge/API-OpenAI%20compatible-412991.svg)](https://platform.openai.com/docs/api-reference)

</div>

---

## 📖 项目简介

长视频问答最贵的资源是**视觉 token**。把两小时的课程均匀抽帧丢给 VLM，既慢又贵，而且大部分算力都花在了与问题无关的片段上。

Skim 的核心命题不是「让 VLM 看完整个视频」，而是「**让 Agent 自己判断应该看哪里、看多少、什么时候已经看够了**」。

用户给一段一两小时网课的**链接**（Bilibili / YouTube）或本地文件，直接提问，例如：

> 老师在哪一段讲了 GRPO 和 PPO 的区别？
> 这一步推导前面用了什么假设？

Agent 会自己决定：

- 先用**便宜信号**（字幕关键词检索、章节结构）做粗定位，把两小时压缩到几分钟的候选范围；
- 只在候选区间做**密集抽帧**，把昂贵的算力花在真正相关的片段上；
- 用 **Temporal Memory** 记录「看过哪些区间、发现了什么、还缺什么」，主动拒绝重复抽帧；
- 在**证据充分且置信度高**时提前停止，证据冲突或不足时继续看相关片段。

整条链路上，**只有 `inspect_video` 会消耗视觉 token**：粗定位完全免费，其余工具只花文本 token。因此「定位准不准」直接决定了成本。

---

## ✨ 核心功能

| 能力 | 说明 |
| --- | --- |
| 🔗 **链接直接输入** | 传入 Bilibili / YouTube 链接，引擎自动下载并缓存（默认 720p H.264），无需提前准备几十上百 GB 的本地文件 |
| 🎬 **多模态视频理解** | 字幕（ASR）+ 画面（PPT / 公式 / 板书 / 代码）双通道，弥补「关键信息不在讲解词里」的问题 |
| 🔎 **分阶段检索** | 低成本粗定位（BM25 + 章节匹配）→ 任务相关性驱动的动态抽帧 → VLM 理解 |
| 🎯 **任务相关性驱动采样** | 抽帧密度由「这段与问题有多相关」决定，而不是「画面变化有多大」 |
| 🧠 **Temporal Memory** | 记录已检查区间 / 知识点 / 证据 / 未解决问题，主动拦截重复抽帧 |
| 🛑 **不确定度驱动的早停** | 证据充分且置信度高就停；存在冲突或不确定就继续看相关片段 |
| 📊 **完整评测框架** | 与「字幕 RAG」「固定频率抽帧」等基线对比，并支持三项消融实验，输出 `report.json` / `report.md` |
| 🔁 **可复现实验** | 内置合成课程视频生成器，零素材即可跑通全流程与基准测试 |
| 🖥️ **Web UI** | Gradio 界面：粘贴链接 → 下载 / 转写 / 生成目录 → 提问，并可实时切换消融开关对比行为差异 |
| 🔐 **密钥安全** | 密钥只从环境变量或 git-ignored 的 `.env` 读取，`SecretStr` + 日志脱敏，绝不进入任何产物 |

---

## 🎯 适用场景与目标用户

**典型场景**

- 📚 **长课程复习与定位**：「这个知识点在哪一分钟讲的？」「这一步推导的前提是什么？」不必重看整节课。
- 🖼️ **画面型信息提取**：公式、板书、PPT、代码、对比表格 —— 这些内容往往**不在讲解词里**，只能靠画面获取。
- 🎓 **教学素材整理**：自动生成章节目录（含标题与摘要），用于课程大纲、复习提纲或出题。
- 💰 **成本 / 质量权衡研究**：在真实长视频上对比「按需观看」与「整段抽帧」「只用字幕」的准确率与视觉开销。

**目标用户**

| 用户 | 能获得什么 |
| --- | --- |
| **学生 / 自学者** | 用提问代替拖动进度条，快速定位长课程中的具体片段 |
| **教师 / 助教** | 自动章节化 + 摘要，快速产出课程结构与复习材料 |
| **研究者 / 工程师** | 一套可复现的评测框架，用于长视频 Agent、多模态检索与成本-精度权衡的实验 |
| **应用开发者** | 一个可嵌入的问答引擎：链接进、带时间戳引用的答案出 |

**不适用 / 非目标**

- ✖️ 通用视频摘要、剪辑或字幕生成工具（本项目只回答具体问题）；
- ✖️ 实时流媒体理解（面向已录制完成的长课程）；
- ✖️ 追求「无脑整段抽帧」的暴力方案 —— 成本与延迟正是本项目要优化掉的部分。

---

## 🏗️ 工作原理

```
                  链接 (Bilibili / YouTube) 或本地文件
                             │
        ┌────────────────────▼────────────────────┐
        │  Stage 0 · 获取媒体（仅链接时需要）          │
        │  yt-dlp 下载一次 → data/downloads/ 缓存     │
        │  · 优先 H.264，避免 AV1 逐帧解码慢/不兼容    │
        │  · 高度上限 720p，保证板书/公式可读          │
        │  · 低码率音轨（ASR 只需 16kHz 单声道）       │
        │  · 镜像卡死自动换源重试                      │
        └────────────────────┬────────────────────┘
                             │ 本地文件
        ┌────────────────────▼────────────────────┐
        │  Stage 1 · 粗定位（便宜，纯文本，0 帧 0 VLM）│
        │  · BM25 字幕检索 + 精确术语命中             │
        │  · LLM 自动章节化 (Table of Contents)       │
        │  · 合并/截断 → 候选时间窗 CandidateWindow    │
        └────────────────────┬────────────────────┘
                             │ 候选窗口 + 章节 + 术语
        ┌────────────────────▼────────────────────┐
        │  Stage 2 · Agent 循环（原生 tool calling）  │
        │                                          │
        │   ┌───────── 便宜工具 ─────────┐           │
        │   │ search_transcript          │           │
        │   │ read_transcript            │           │
        │   │ list_chapters              │           │
        │   │ memory_status              │           │
        │   └────────────┬───────────────┘           │
        │                │ 证据不足 / 问题涉及视觉内容 │
        │   ┌────────────▼───────────────┐           │
        │   │ inspect_video（唯一的贵工具）│           │
        │   │  ├ Temporal Memory 拦截重复  │           │
        │   │  ├ 按 relevance 决定抽帧密度 │           │
        │   │  ├ bundled ffmpeg 抽帧       │           │
        │   │  └ VLM 读取 PPT/公式/代码    │           │
        │   └────────────┬───────────────┘           │
        │                │ Observation 写回 Memory   │
        │                ▼                           │
        │   ┌────────────────────────────┐           │
        │   │ 停止判据                     │           │
        │   │  · answers_question 且高置信 → 停 │       │
        │   │  · 连续 N 步无新证据         → 停 │       │
        │   │  · 预算耗尽                 → 停 │       │
        │   └────────────────────────────┘           │
        └────────────────────┬────────────────────┘
                             │
        ┌────────────────────▼────────────────────┐
        │  Stage 3 · 综合回答                        │
        │  基于 Memory 中的证据 + 字幕片段生成答案，   │
        │  附带时间戳引用、置信度、剩余不确定性         │
        └─────────────────────────────────────────┘
```

---

## 🚀 快速开始

```bash
# 1) 生成一段示例课程视频 + 评测数据集（零素材，无需联网下载视频）
uv run skim sample

# 2) 直接提问（该问题只有幻灯片上才有答案；示例视频无音轨，故显式给出字幕）
uv run skim ask "data/assets/sample_lecture.mp4" "GRPO 的目标函数里，优势 A_i 具体是怎么算出来的？" `
    --subtitle "data/assets/sample_lecture.transcript.json"

# 3) 启动 Web UI
uv run skim ui      # 浏览器打开 http://127.0.0.1:7860
```

> 也可以直接给链接，跳过手动准备视频的步骤：
>
> ```bash
> uv run skim ask "https://www.bilibili.com/video/BV1CAxaeHEeH/?p=106" "定积分的换元法和分部积分法分别在什么时候用？"
> ```

---

## 🛠️ 安装与配置

### 环境要求

| 项目 | 要求 |
| --- | --- |
| 操作系统 | Windows / Linux / macOS |
| Python | ≥ 3.11（由 `uv` 自动管理，无需手动安装） |
| [uv](https://docs.astral.sh/uv/) | **必需**，用于环境与依赖管理 |
| ffmpeg | **无需单独安装**，由 `imageio-ffmpeg` 自带静态二进制 |
| 链接下载 | 由 `yt-dlp` 提供，随依赖一起安装 |
| 网络 | 需要能访问 LLM 接口；访问外网视频/模型仓库时通常需要代理 |
| GPU | 可选。ASR 默认走 CPU（int8），有 NVIDIA GPU 可切到 CUDA |

### 1. 安装依赖

```bash
cd skim
uv sync --all-extras
```

若代理较慢，可指定国内镜像加速：

```bash
uv sync --all-extras --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

> `uv` 自身下载依赖也需要走代理，首次安装前可先导出代理变量：
>
> ```powershell
> $env:HTTP_PROXY  = "http://127.0.0.1:7892"
> $env:HTTPS_PROXY = "http://127.0.0.1:7892"
> $env:ALL_PROXY   = "http://127.0.0.1:7892"
> ```
>
> ```bash
> export HTTP_PROXY=http://127.0.0.1:7892
> export HTTPS_PROXY=http://127.0.0.1:7892
> export ALL_PROXY=http://127.0.0.1:7892
> ```
>
> 程序运行时的代理由 `.env` 里的 `SKIM_PROXY` 负责（会自动注入 `httpx` 与 `huggingface_hub`），无需长期依赖 shell 变量。

### 2. 配置密钥

密钥**不会**写在代码里，只从**环境变量**或 git-ignored 的 `.env` 读取。

**方式 A（推荐，生产 / CI）：直接导出环境变量**

```powershell
$env:DEEPSEEK_API_KEY  = "sk-你的密钥"
$env:DEEPSEEK_BASE_URL = "https://api.deepseek.com"
```

**方式 B（本地开发）：复制模板并填写**

```bash
cp .env.example .env        # Windows: Copy-Item .env.example .env
# 然后编辑 .env，填入 DEEPSEEK_API_KEY
```

只要服务端兼容 OpenAI 接口即可替换 `DEEPSEEK_BASE_URL` 与模型名，不限于 DeepSeek。

### 3. 配置项一览

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | —（**必填**） | API 密钥（亦可用 `SKIM_API_KEY`） |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容端点（亦可用 `SKIM_BASE_URL`） |
| `SKIM_LLM_MODEL` | `deepseek-flash` | 推理模型 |
| `SKIM_VLM_MODEL` | `deepseek-flash` | 视觉模型（需支持 text + image 输入） |
| `SKIM_PROXY` | 空 | 出站代理，如 `http://127.0.0.1:7892`；留空则不改动环境变量 |
| `SKIM_DATA_DIR` | `data` | 数据根目录 |
| `SKIM_VIDEO_MAX_HEIGHT` | `720` | 链接下载时的清晰度上限（`360`/`480`/`720`/`1080`） |
| `SKIM_WHISPER_MODEL` | `small` | ASR 模型：`tiny`/`base`/`small`/`medium`/`large-v3` |
| `SKIM_WHISPER_DEVICE` | `cpu` | `cpu` 或 `cuda` |
| `SKIM_WHISPER_COMPUTE_TYPE` | `int8` | `int8`（CPU）/ `float16`（GPU） |
| `SKIM_MAX_AGENT_STEPS` | `8` | Agent 步数上限 |
| `SKIM_MAX_VLM_FRAMES` | `4` | 单次 `inspect_video` 抽帧上限 |
| `SKIM_REQUEST_TIMEOUT` | `180` | 单次 LLM 请求超时（秒） |
| `SKIM_LOG_LEVEL` | `INFO` | 日志级别 |
| `SKIM_ENV_FILE` | — | 显式指定 `.env` 路径（默认自动向上查找） |

### 🔐 密钥安全机制

1. **不落盘、不硬编码**：密钥只存在于进程环境或 git-ignored 的 `.env`；`.gitignore` 已屏蔽 `.env`、`*.key`、`secrets.json`。
2. **`SecretStr` 保护**：配置对象用 `pydantic.SecretStr` 承载密钥，`print(settings)` / `repr(settings)` 只输出 `**********`。
3. **日志脱敏**：`logging_utils.RedactionFilter` 从**每一条**日志中抹掉密钥（已知密钥逐字替换 + `sk-...` 正则兜底），控制台与文件 handler 均已挂载。
4. **展示用掩码**：需要展示时只用 `settings.masked_key`（形如 `sk-76b4...c4c6`）。
5. **不进入产物**：`report.json`、运行轨迹、缓存 meta 中都不包含密钥。

> 生产环境建议改用密钥管理服务（Vault / AWS Secrets Manager / K8s Secret）并通过环境变量注入，而不是 `.env` 文件。

---

## 📦 使用指南

### 命令行参考

`<SOURCE>` 处填**链接或本地路径**均可。

| 命令 | 作用 |
| --- | --- |
| `uv run skim sample [--out DIR]` | 生成合成课程视频 + 评测数据集 |
| `uv run skim ingest SOURCE [--subtitle SRT] [--max-height N] [--force]` | 下载（如为链接）/ 转写 / 建索引并缓存 |
| `uv run skim ask SOURCE "问题" [选项]` | 跑一次完整问答 |
| `uv run skim chapters SOURCE [--force]` | 生成并打印自动章节目录 |
| `uv run skim evaluate DATASET.json [--methods ...] [--out DIR]` | 运行基准与消融实验 |
| `uv run skim ui [--host H] [--port P] [--share]` | 启动 Web UI（默认 `127.0.0.1:7860`） |
| `uv run skim info` | 显示解析后的配置（密钥已掩码） |
| `uv run skim cache` | 列出已入库的视频 |

`skim ask` 常用选项：

```
--max-steps 8            Agent 步数上限
--max-frames 4           单次 inspect_video 抽帧上限
--min-confidence 0.7     早停置信度阈值
--no-adaptive            消融：关闭任务相关性驱动的抽帧密度
--no-memory              消融：关闭 Temporal Memory
--no-uncertainty         消融：关闭不确定度早停
--json-out PATH          导出完整运行记录（含决策轨迹）
```

### 典型用法

**① 直接给链接**

把链接交给引擎即可，无需提前下载到本地。下面这条是真实可用的测试链接：
宋浩老师《高等数学》p106「定积分的换元法与分部积分法」，**1 小时 28 分、纯黑板板书、且该视频没有字幕**，
因此会走完整的「下载 → ASR → 检索 → Agent」链路。

```bash
# 入库：下载链接（默认 720p H.264）→ 抽音频 → faster-whisper 转写
uv run skim ingest "https://www.bilibili.com/video/BV1CAxaeHEeH/?p=106"

# 提问（同一链接再传一次即可，第二次直接命中缓存）
uv run skim ask "https://www.bilibili.com/video/BV1CAxaeHEeH/?p=106" `
    "定积分的换元法和分部积分法分别在什么时候用？"

# 觉得慢可以先降低清晰度 / 换小模型
uv run skim ingest "https://www.bilibili.com/video/BV1CAxaeHEeH/?p=106" --max-height 480
```

> **耗时预期（1.5 小时课程，CPU）**：下载约 5–15 分钟（走代理时带宽有限），
> ASR 是主要瓶颈，`small` 模型通常需要 20–40 分钟。两者都有缓存，只需付一次成本。
> 想快速体验请用 `skim sample`，或设置 `SKIM_WHISPER_MODEL=base`。

**② 使用本地文件（可自带字幕）**

链接和本地路径共用同一套接口，本地文件会跳过下载这一步。

```bash
# 已有字幕可以直接给出，跳过 ASR（最快）
uv run skim ingest "D:/videos/lecture.mp4" --subtitle "D:/videos/lecture.srt"

uv run skim ask "D:/videos/lecture.mp4" "老师在哪一段讲了 GRPO 和 PPO 的区别？" `
    --json-out data/runs/my_run.json
```

### 🖥️ Web UI

```bash
uv run skim ui
# 浏览器打开 http://127.0.0.1:7860
```

界面分两步：

1. **1 · Load a lecture** —— 在输入框里**粘贴课程链接**（Bilibili / YouTube），或填写本地文件路径，点 **Download**。
   引擎会下载（折叠面板里可改清晰度上限）、转写并生成章节目录，并给出：
   - 状态摘要（时长、分辨率、字幕段数与来源、章节数）；
   - **可下载的 ASR 字幕文件**（`.srt`，点击即可下载，也能作为外部字幕复用）；
   - 本地视频预览与章节目录表（含每章标题与摘要）。
2. **2 · Ask** —— 输入问题，点 **Ask**，界面会给出：
   - 答案 + 置信度 + 停止原因 + 步数；
   - 证据表（时间戳 + 引用）；
   - 折叠面板中的成本指标（VLM 调用次数、抽帧数、视觉 token、覆盖率、延迟）与 Agent 决策轨迹。

折叠面板中还可以实时开关三项消融（自适应抽帧 / Temporal Memory / 不确定度早停），直接对比行为差异。

### 📄 数据集格式

```json
{
  "name": "my-lecture-qa",
  "video": "sample_lecture.mp4",
  "subtitle": "sample_lecture.transcript.json",
  "questions": [
    {
      "id": "q1",
      "question": "老师在哪一段讲了 GRPO 和 PPO 的区别？",
      "gold_answer": "在对比表格那一段（约 02:10-02:36）……",
      "gold_window": [130, 156],
      "requires_vision": false
    }
  ]
}
```

- `video` 可以是**链接**（`https://...`）也可以是本地路径。为相对路径时按数据集文件所在目录解析；写成链接则自动下载并缓存。
- `gold_window`（秒）用于计算**定位命中率**：Agent 实际抽帧的位置是否落在参考区间内。
- `requires_vision` 仅为说明性字段，不参与指标计算。

从 `.srt` 生成数据集骨架：

```bash
uv run python -c "from skim.media.transcript import load_subtitle; t=load_subtitle(r'lecture.srt'); print(len(t.segments))"
```

---

## 📊 评测与消融实验

```bash
uv run skim evaluate data/assets/sample_lecture.dataset.json \
    --methods agent,agent-no-adaptive,agent-no-memory,agent-no-uncertainty,subtitle_rag,fixed_fps \
    --fps-interval 30 \
    --out data/runs/benchmark
```

输出 `report.json`（机器可读）与 `report.md`（对比表格 + 逐题明细）。

### 对比方法

| method | 说明 |
| --- | --- |
| `agent` | 完整系统：自适应观察 + Temporal Memory + 不确定度早停 |
| `agent-no-adaptive` | 消融：抽帧密度固定，不随相关性变化 |
| `agent-no-memory` | 消融：无 Temporal Memory，可以重复查看同一区间 |
| `agent-no-uncertainty` | 消融：不因置信度高而提前停止 |
| `subtitle_rag` | 基线：只用字幕做 RAG，完全不看视频 |
| `fixed_fps` | 基线：整段视频固定频率抽帧后交给 VLM |

### 指标

| 指标 | 含义 |
| --- | --- |
| `accuracy` / `answer_score` | LLM-as-judge 判定的正确率 / 平均分 |
| `temporal_hit_rate` | 抽帧位置落在参考时间窗内的比例（定位质量） |
| `frames_sent` | 实际送入 VLM 的帧数 |
| `vlm_calls` | VLM 调用次数 |
| `visual_tokens` | **视觉调用的真实 prompt token 数**（来自 API 返回，图像占主导） |
| `coverage_ratio` | 视频被「看过」的时间占比（只统计视觉观察） |
| `wall_latency_s` | 端到端延迟 |

### 内置示例上的实测结果

在合成课程视频（229.5s、8 段字幕、5 道题，其中 2 道「答案只在画面上」）上：

| method | 准确率 | 抽帧数 | VLM 调用 | 视觉 token | 视频覆盖率 |
| --- | --- | --- | --- | --- | --- |
| **agent** | **100%** | **3.4** | **1.2** | **1696** | **17.4%** |
| agent-no-adaptive | 100% | 3.6 | 1.2 | 1742 | 14.5% |
| agent-no-memory | 100% | 3.4 | 1.2 | 1419 | 14.5% |
| agent-no-uncertainty | 100% | 3.6 | 1.2 | 1743 | 14.5% |
| subtitle_rag（只看字幕） | 40% | 0 | 0 | 0 | 0% |
| fixed_fps（整段均匀抽帧） | 80% | 8.0 | 2.0 | 2122 | 100% |

三条结论：

1. **视觉通道不可替代**：只用字幕的 RAG 在「公式 / 代码 / 对比表」类问题上只能拿到 40%，因为关键信息根本不在讲解词里。
2. **定位换成本**：Agent 用 3.4 帧达到 100%，而「整段均匀抽帧」要用 8.0 帧、覆盖 100% 的视频也只能到 80% —— 用 17.4% 的覆盖率换到更高准确率。
3. **消融差异在短视频上不明显**：这段示例只有约 4 分钟、8 段字幕，候选窗口本身就少，因此三项消融与完整系统差距很小。
   这些机制的价值随视频长度与检索难度上升而放大，在 1–2 小时的真实课程上（数千段字幕、数十个候选窗口、几百帧的基线开销）才会体现。

### 真实长视频上的实测

用上面的测试链接（宋浩《高等数学》p106，**1:28:52、纯黑板板书、无字幕**）跑完整链路，两个问题对比出「什么时候该看视频」：

| 问题 | VLM 调用 | 抽帧 | 视觉 token | 视频覆盖率 | 耗时 | 停止原因 |
| --- | --- | --- | --- | --- | --- | --- |
| 换元后上下限怎么变？老师举了什么例子？ | **0** | 0 | 0 | 0% | 13s | finish |
| 请读出黑板上的换元法定理（含公式） | 2 | 7 | 3157 | **3.7%** | 13s | finish |

- 第一个问题**字幕里就有答案**，Agent 直接判定不需要看视频，0 帧 0 视觉 token 收工；它同时如实报告的剩余不确定性是「语音识别把符号听错了（R4 / 杀鞋 之类）」。
- 第二个问题**只能来自黑板**。此时字幕恰好是坏的（「引入一个坏缘 第1个 5R2是党约一个A」），而 VLM 直接从画面读出
  `φ(α)=a, φ(β)=b`、`φ(t) 在 [α,β] 上有连续导数`、`∫_a^b f(x)dx = ∫_α^β f[φ(t)]φ'(t)dt`。
  **88 分钟的视频只看了 3.7%**，其余 96% 一帧未取。

> 说明：这次实跑用的是 `tiny` ASR 模型（受本机代理带宽限制，见 FAQ），所以字幕质量偏噪音 ——
> 反而更真实地暴露了「字幕不可全信、画面是独立证据」这一点。带宽允许时把 `SKIM_WHISPER_MODEL` 换成 `small` 会明显改善字幕。

### 完整基准实验（9 题，6 种方法）

素材：[西湖大学 WindyLab《强化学习的数学原理》p01](https://www.bilibili.com/video/BV1sd4y167NS/)
（36:03、纯幻灯片讲解、无字幕，视频自带 4 个官方章节）。
数据集 `data/assets/rl_lecture.dataset.json` 共 9 题，**5 题字幕可答 + 4 题只能看画面**。

```bash
uv run skim evaluate data/assets/rl_lecture.dataset.json \
  --methods agent,agent-no-adaptive,agent-no-memory,agent-no-uncertainty,subtitle_rag,fixed_fps \
  --fps-interval 60 --out data/runs/rl_benchmark
```

| method | 准确率 | 抽帧数 | VLM 调用 | 视觉 token | 视频覆盖率 | 延迟 | 定位命中 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **agent** | **88.9%** | **2.9** | **0.9** | **1350** | **3.8%** | **9.2s** | **100%** |
| agent-no-adaptive | 100% | 2.4 | 0.8 | 1124 | 2.5% | 9.6s | 100% |
| agent-no-memory | 88.9% | 2.4 | 0.7 | 935 | 3.1% | 8.1s | 100% |
| agent-no-uncertainty | 88.9% | 2.9 | 0.9 | 1332 | 3.3% | 9.9s | 100% |
| subtitle_rag（只看字幕） | 33.3% | 0 | 0 | 0 | 0% | 1.9s | 44.4% |
| fixed_fps（整段均匀抽帧） | 44.4% | 37.0 | 10.0 | 10001 | 100% | 24.7s | 77.8% |

结论：

1. **视觉通道不可替代**：只看字幕只有 33.3%，4 道画面题基本全丢。
2. **定位换成本**：Agent 用 **2.9 帧 / 1350 视觉 token / 3.8% 覆盖率**拿到 88.9%，而「整段均匀抽帧」用
   **37 帧 / 10001 视觉 token / 100% 覆盖率**只有 44.4% —— 视觉开销降到 **1/7.4**、视频覆盖率降到 **1/26**，准确率反而翻倍。
3. **定位命中率 100% vs 44.4%（字幕 RAG）**：说明它答对是因为真的看对了地方。
4. **消融差异不大（88.9%~100%）**：9 道题里 1 题就是 11 个百分点，这个量级下三项机制各自的影响被题目数淹没；
   要看出稳定差距需要更大的题集，评测框架已就绪，换数据集即可。

> **这次实验本身发现了一个真实缺陷并已修复**：修复前完整 agent 只有 77.8%，且两处失败都是 `no_progress` + **0 抽帧**
> —— 停滞判据只把「视觉证据增长」当作进展，于是纯字幕的调查会被误判成「没有进展」而过早收工。
> 改为把**「便宜工具返回了新内容」（新读到的字幕窗口）也计入进展**后，完整 agent 77.8% → 88.9%，且 9 题全部以 `finish` 正常结束。
> 这正是做实验的意义：不跑基准就发现不了这个判据写错了。

---

## 🧠 设计要点

### 1. 为什么是「任务相关性驱动」而不是「画面变化驱动」

曾尝试用帧间差异决定何时加密抽帧，但规则很不可靠：

- 老师站着不动讲一个关键公式 → 画面几乎无变化，却最重要；
- 切 PPT / 镜头移动 → 变化很大，但可能和用户问题无关。

因此改为：Agent 综合**用户问题 + 字幕内容 + 已找到的证据 + 还缺什么信息**，决定下一步看哪一段、抽多少帧。相关度越高，抽帧越密。

### 2. Temporal Memory 解决「重复看同一段」

Agent 很容易在多轮之后忘记某段已经查过。Memory 记录：

- 已检查的时间区间（**只统计真正花过 VLM 预算的视觉观察**；只读字幕不算「看过画面」，因此不会错误地阻止后续视觉检查）；
- 每段涉及的知识点、找到的证据、尚未解决的问题 `open_questions`。

下一轮决策前，`inspect_video` 会先检查 Memory，对已覆盖窗口直接**拒绝**并返回已有结论（可用 `force=true` 覆盖），直接减少重复的 VLM 调用。

### 3. 目标不是「帧越少越好」

一味追求少看会漏掉关键内容（某个概念可能只在 PPT 上出现几秒）。因此优化目标是**准确率与计算成本的权衡**：

- 当证据看起来足够（某窗口 `answers_question=true` 且置信度达标、且无遗留问题），系统把这一判断作为**提示**交给 Agent，请它决定是否 `finish`；
  决策权留在 Agent 手里，避免「单个窗口的局部自评」直接覆盖全局判断（实测中硬停确实会在「代码题」上过早收手）；
- 证据冲突 / 不足 → 允许继续查看相关片段；
- 连续若干步没有任何新证据 → 停止（`stop_reason = no_progress`）；
- 始终有一层「兜底综合」，保证预算耗尽时也能给出有依据的答案。

### 4. 为什么用 bundled ffmpeg

目标机器没有系统 ffmpeg。`imageio-ffmpeg` 提供静态二进制，探测、抽音频、抽帧全部走它，避免额外环境依赖。
ASR 侧则自己把音频解成 16 kHz 单声道 PCM 数组再交给 faster-whisper，绕开 PyAV 容器 API 的版本差异。

### 5. 链接下载的几个刻意选择

- **一次下完并缓存，而不是按需分段下载**：Bilibili 返回的是带签名的临时 URL（manifest 里 `deadline` 会过期），
  按需分段就得反复重新解析；且整段下载相对 20–40 分钟的 CPU ASR 只是零头，缓存后同一链接只付一次成本。
- **优先 H.264、限制 720p**：抽帧是上百次单时间点 seek，AV1 软解慢且有兼容风险；板书/公式在 768px 下 VLM 也能准确转写
  （实测同一帧 768px 用 279 token，1280px 用 635 token，转写质量接近），没有必要下 1080p。
- **URL 归一化后再做缓存键**：`?spm_id_from=...&vd_source=...&p=106` 与分享出来的 `?p=106` 是同一节课，
  归一化（丢掉跟踪参数并排序）保证不会重复下载。
- **镜像失败自动换源**：Bilibili 每次解析分配不同 CDN 镜像，个别镜像走代理会卡死；`socket_timeout` 放弃死连接 +
  外层重新解析换镜像 + 断点续传，重试不会白下已完成的流。

### 6. 章节摘要不会「悄悄消失」

生成章节依赖 LLM；偶发的连接失败会让引擎退回**均匀兜底**（按时间等分成 `Section 1..N`，无摘要）。
这种降级结果**不会写入缓存**，因此网络恢复后下一次运行会自动重试，而不会让一份没有摘要的目录永久留在本地缓存里。

---

## 📁 项目结构

```
skim/
├── pyproject.toml          uv 项目与依赖定义
├── .env.example            配置模板（复制为 .env 使用）
├── .gitignore
├── README.md
├── src/skim/               源码
│   ├── config.py           配置 / 密钥 / 路径布局（SecretStr 防泄漏）
│   ├── logging_utils.py    日志 + 密钥脱敏过滤器
│   ├── cli.py              命令行入口（skim）
│   ├── demo.py             合成课程视频 + 评测数据集生成器
│   ├── llm/                OpenAI 兼容客户端（重试、用量记账、JSON 兜底）
│   ├── media/              下载 / ffmpeg / 抽帧 / 字幕模型 / ASR / 入库编排
│   ├── indexing/           分词、BM25 检索、LLM 章节化、粗定位
│   ├── memory/             Temporal Memory（区间覆盖、去重、证据摘要）
│   ├── agent/              VLM 观察者、工具定义与执行、Agent 主循环
│   ├── eval/               指标、基线、基准与消融运行器
│   └── ui/app.py           Gradio Web UI
├── scripts/verify_asr.py   单独验证 ASR 链路
└── data/                   运行时目录（git-ignored）
    ├── assets/             合成视频、字幕、数据集
    ├── downloads/          链接下载下来的视频 + source.json 缓存映射
    ├── cache/<video_id>/   字幕、章节、抽帧缓存
    ├── models/             ASR 模型权重
    └── runs/               评测报告与运行记录
```

---

## ❓ 常见问题

**Q1. `uv sync` 卡住或报 `tunnel error` / `connection refused`**

`uv` 不会自动读取 `.env`，需要先把代理导出到 shell（见「安装与配置」）。若已 `uv sync` 过，可用 `uv run --no-sync ...` 跳过重新同步。

**Q2. 机器上本来就有 `HTTP_PROXY`，会不会冲突？**

程序运行时会用 `SKIM_PROXY` **覆盖** `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY`（含小写形式），并为 `localhost,127.0.0.1` 设置 `NO_PROXY`。
这是刻意为之：本机环境曾存在指向另一个端口的 `HTTP_PROXY`，如果沿用 `setdefault` 就会被它「赢」，表现为 LLM 调用正常、
但**下载 ASR 模型时连接被拒**，非常难排查。若确实想用系统代理，把 `.env` 里的 `SKIM_PROXY` 留空即可。

**Q3. 报 `Readme file does not exist: README.md`**

说明在项目根目录之外执行了构建。请先 `cd skim` 再执行 `uv sync`。

**Q4. VLM 返回空字符串 / `VisualObserver` 没有任何输出**

本项目默认的 `deepseek-flash` 是**推理模型**，reasoning token 会占用 `max_tokens`。
代码默认通过 `thinking={"type":"disabled"}` 关闭推理（更快更省）；若你手动调用并放宽了该设置，请把 `max_tokens` 提到 4096 以上。

**Q5. ASR 报 `TypeError: open() got an unexpected keyword argument 'metadata_errors'`**

这是 faster-whisper 与新版 PyAV 的兼容问题。本项目**不依赖 PyAV 解码**：音频先由 ffmpeg 转成 16 kHz 单声道 WAV，
再以 numpy 数组喂给模型（`media/asr.py`）。即使自行传入非 WAV 文件，`_decode()` 也会先用 ffmpeg 归一化。

**Q6. 首次 ASR 很慢 / 卡在下载模型**

首次运行会从 Hugging Face 下载模型到 `data/models/`，走 `SKIM_PROXY`。两个已处理的坑：

1. **代理掐断 TLS**：长传输偶发 `UNEXPECTED_EOF_WHILE_READING`，因此模型加载内置 3 次重试 —— 不会因为一次抖动丢掉前面十几分钟的下载与抽音频。
2. **单连接被限速**：本机代理对单条连接约 110 KB/s，`small`（约 480 MB）单流下载要一个多小时。
   因此程序会自动开启 Hugging Face 的 **Xet 高性能传输**（`HF_XET_HIGH_PERFORMANCE=1`，并行分块；实测 4 条并行 range 请求可达 ~460 KB/s）。
   注意 `HF_HUB_ENABLE_HF_TRANSFER` 在 `huggingface_hub` 1.x 已废弃，不要再用它。

想进一步加速可调小模型：`$env:SKIM_WHISPER_MODEL="base"`（约 145 MB）或 `tiny`（约 75 MB）。
Windows 下还会提示 symlink 警告，属正常现象，可用 `$env:HF_HUB_DISABLE_SYMLINKS_WARNING=1` 关闭。

**Q7. Windows 控制台出现 jieba 的 `SyntaxWarning`**

jieba 自身的正则字符串转义告警，不影响功能。已通过 `jieba.setLogLevel(60)` 压制其日志；该 warning 来自模块编译期，可安全忽略。

**Q8. Gradio 端口被占用**

`uv run skim ui --port 7861`。

**Q9. 想用 GPU 跑 ASR**

```
SKIM_WHISPER_DEVICE=cuda
SKIM_WHISPER_COMPUTE_TYPE=float16
```

**Q10. 视频没有音轨**

`ingest` 会明确报错。请通过命令行提供字幕：`uv run skim ingest <视频> --subtitle <文件>`（支持 `.srt` / `.vtt` / `.json`）。

**Q11. 链接下载很慢 / 卡在某个百分比不动**

两个已知原因，都已在引擎里处理：

1. **并发数**：Bilibili 把每条 DASH 流当作单个 m4s 文件，走本地代理时并行 range 请求会间歇性挂死
   （实测 4 并发时音轨卡在 0 字节不动，改成 1 并发后稳定在 ~90 KiB/s），因此 `concurrent_fragment_downloads` 固定为 1。
2. **CDN 镜像**：每次解析都会分配不同镜像，个别镜像走代理会卡死。`socket_timeout=30` 放弃死连接，外层重新解析换镜像最多 3 次，
   `continuedl` 让断点续传生效。

若整体带宽确实不足（实测走代理约 0.1–0.5 MB/s），可以 `--max-height 480` 降低体积，或用 `SKIM_WHISPER_MODEL=base` 缩短 ASR 时间。
下载与转写都有缓存，`data/downloads/` 里同一链接只会下一次。

**Q12. 为什么默认只下 720p、为什么不用 AV1**

720p 是「板书 / 公式还能看清」与「体积翻倍」之间的折中，抽帧后还要缩到 768px 交给 VLM，1080p 的额外信息大多被浪费。
格式选择器刻意**优先 H.264（avc1）而不是 AV1**：本引擎要做上百次单时间点 seek，AV1 软解又慢又有兼容风险，而各清晰度档位都提供 H.264。

**Q13. 想要更高清晰度 / 需要登录才能看的视频**

用 `--max-height 1080`。需要大会员或登录的视频，可导出 cookies 后设置 `yt-dlp` 的 `cookiefile`
（在 `media/download.py` 的 `_build_ydl_options` 里加一项即可）。

**Q14. 想控制成本**

- 降低 `--max-steps`（默认 8）与 `--max-frames`（默认 4）；
- 提高 `--min-confidence` 让 Agent 更早停止；
- 用 `SKIM_MAX_VLM_FRAMES` 限制单次抽帧上限；
- 关注 `report.md` 里的 `visual_tokens` 与 `coverage_ratio`。

**Q15. 章节没有摘要 / 只有 `Section 1..N`**

这是 LLM 生成章节失败后触发的**均匀兜底**（按时间等分，天然没有摘要）。该类降级结果不会写入缓存，
网络恢复后重新执行一次 `Download` 或 `uv run skim chapters <SOURCE> --force` 即可重新生成带标题与摘要的目录。

---

## 🤝 贡献与开发

欢迎提交 Issue 与 Pull Request。本地开发流程：

```bash
uv sync --all-extras          # 安装依赖
uv run skim sample            # 生成零素材示例，验证全链路
uv run skim ui                # 本地起 UI，人工核验交互
uv run python scripts/verify_asr.py   # 单独验证 ASR 链路
```

约定与注意事项：

- **代码风格**：源码统一 `from __future__ import annotations`，模块与函数保留简短 docstring；
- **别把密钥带进仓库**：`.env`、`*.key`、`secrets.json` 已在 `.gitignore` 中，提交前请自查；
- **业务适配器不要污染通用模块**：新增能力优先收敛在各自子包内，避免把特化逻辑塞进 `agent/` 与 `llm/`；
- **改动评测相关逻辑后请重跑基准**：`skim evaluate` 的口径变化会直接影响 `report.md` 中的结论。
- **内部测试套件**：项目自带离线单元测试与联网端到端测试（`tests/`，`uv run pytest`），该目录不随本仓库分发，仅在维护者本地保留。

---

## 📄 许可证

本项目以 **MIT License** 发布，许可证全文见 [LICENSE](LICENSE)（`pyproject.toml` 中亦声明为 MIT）。

你可以自由用于商业或非商业用途，只需保留原始版权声明与许可声明。

---

<div align="center">

**Skim** —— 长视频 Agent 的关键不是「能不能看懂一帧」，而是「**为了回答这个问题，到底该看哪里、看多少**」。

</div>