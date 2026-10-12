<div align="center">

# Skim

**面向长课程视频的「按需观看」问答 Agent**

给一段课程链接（Bilibili / YouTube）或本地文件直接提问，Agent 自己判断**该看哪里、看多少、什么时候已经看够了**。

[![Python](https://img.shields.io/badge/python-%E2%89%A5%203.11-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-2EA043.svg)](#-许可证)
[![uv](https://img.shields.io/badge/packaged%20with-uv-5C5CFF.svg)](https://docs.astral.sh/uv/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-F97316?logo=gradio\&logoColor=white)](https://gradio.app/)

</div>

***

## 🚀 快速开始

```bash
cd skim
uv sync --all-extras          # 1) 安装依赖（首次需联网）

uv run skim sample            # 2) 生成一段示例课程视频（零素材，无需下载视频）

uv run skim ask "data/assets/sample_lecture.mp4" "GRPO 的目标函数里，优势 A_i 具体是怎么算出来的？" `
    --subtitle "data/assets/sample_lecture.transcript.json"   # 3) 跑一次问答

uv run skim ui                # 4) 启动 Web UI → 打开 http://127.0.0.1:7860
```

> 使用前请先配置 LLM 密钥（见下文「安装与配置 → 配置密钥」），否则问答与章节生成无法运行。
> 也可以直接传链接，跳过手动准备视频：
>
> ```bash
> uv run skim ask "https://www.bilibili.com/video/BV1CAxaeHEeH/?p=106" "定积分的换元法和分部积分法分别在什么时候用？"
> ```

***

## 🛠️ 安装与配置

### 环境要求

| 项目                               | 要求                         |
| -------------------------------- | -------------------------- |
| 操作系统                             | Windows / Linux / macOS    |
| Python                           | ≥ 3.11（由 `uv` 自动管理，无需手动安装） |
| [uv](https://docs.astral.sh/uv/) | **必需**，用于环境与依赖管理           |

### 1. 安装依赖

```bash
cd skim
uv sync --all-extras
```

### 2. 配置密钥

**方式 A（推荐）：直接导出环境变量**

```powershell
$env:DEEPSEEK_API_KEY  = "sk-你的密钥"
$env:DEEPSEEK_BASE_URL = "https://api.deepseek.com"
```

**方式 B（本地开发）：复制模板并填写**

```bash
cp .env.example .env
# 然后编辑 .env，填入 DEEPSEEK_API_KEY
```

任何兼容 OpenAI 接口的服务都可替换 `DEEPSEEK_BASE_URL` 与模型名，不限于 DeepSeek。

***

## 📦 使用指南

### 命令行参考

`<SOURCE>` 处填**链接或本地路径**均可。

| 命令                                                                      | 作用                             |
| ----------------------------------------------------------------------- | ------------------------------ |
| `uv run skim sample [--out DIR]`                                        | 生成合成课程视频 + 评测数据集               |
| `uv run skim ingest SOURCE [--subtitle SRT] [--max-height N] [--force]` | 下载（如为链接）/ 转写 / 建索引并缓存          |
| `uv run skim ask SOURCE "问题" [选项]`                                      | 跑一次完整问答                        |
| `uv run skim chapters SOURCE [--force]`                                 | 生成并打印自动章节目录                    |
| `uv run skim ui [--host H] [--port P] [--share]`                        | 启动 Web UI（默认 `127.0.0.1:7860`） |
| `uv run skim info`                                                      | 显示解析后的配置                       |
| `uv run skim cache`                                                     | 列出已入库的视频                       |
| `uv run skim evaluate DATASET.json [--methods ...] [--out DIR]`         | 运行基准与消融实验（进阶）                  |

`skim ask` 常用选项：

```text
--max-steps 8            Agent 步数上限
--max-frames 4           单次 inspect_video 抽帧上限
--min-confidence 0.7     早停置信度阈值
--json-out PATH          导出完整运行记录（含决策轨迹）
--no-adaptive / --no-memory / --no-uncertainty    消融开关
```

### 典型用法

**① 直接给链接**

把链接交给引擎即可，无需提前下载到本地。

```bash
# 入库：下载链接（默认 720p H.264）→ 抽音频 → faster-whisper 转写
uv run skim ingest "https://www.bilibili.com/video/BV1CAxaeHEeH/?p=106"

# 提问（同一链接再传一次即可，第二次直接命中缓存）
uv run skim ask "https://www.bilibili.com/video/BV1CAxaeHEeH/?p=106" `
    "定积分的换元法和分部积分法分别在什么时候用？"

# 觉得慢可以先降低清晰度
uv run skim ingest "https://www.bilibili.com/video/BV1CAxaeHEeH/?p=106" --max-height 480
```

> **耗时预期（1.5 小时课程，CPU）**：下载约 5–15 分钟，ASR 是主要瓶颈，`small` 模型通常需 20–40 分钟。
> 两者都有缓存，只需付一次成本。想快速体验请用 `skim sample`，或设置 `SKIM_WHISPER_MODEL=base`。

**② 使用本地文件（可自带字幕）**

```bash
# 已有字幕可直接给出，跳过 ASR（最快）
uv run skim ingest "D:/videos/lecture.mp4" --subtitle "D:/videos/lecture.srt"

uv run skim ask "D:/videos/lecture.mp4" "老师在哪一段讲了 GRPO 和 PPO 的区别？"
```

### 🖥️ Web UI

```bash
uv run skim ui
# 浏览器打开 http://127.0.0.1:7860
```

界面分两步：

1. **1 · Load a lecture** —— 粘贴课程链接或填写本地路径，点 **Download**。
   引擎会下载、转写并生成章节目录，并给出状态摘要、可下载的 ASR 字幕（`.srt`）、视频预览与章节表。
2. **2 · Ask** —— 输入问题，点 **Ask**，即可看到答案 + 置信度 + 停止原因，以及证据表与成本指标。

***

## 📁 项目结构

```text
skim/
├── pyproject.toml        uv 项目与依赖定义
├── .env.example          配置模板（复制为 .env 使用）
├── src/skim/
│   ├── config.py         配置 / 密钥 / 路径布局
│   ├── cli.py            命令行入口（skim）
│   ├── llm/              OpenAI 兼容客户端
│   ├── media/            下载 / ffmpeg / 抽帧 / ASR / 入库编排
│   ├── indexing/         分词、BM25 检索、LLM 章节化、粗定位
│   ├── memory/           Temporal Memory
│   ├── agent/            VLM 观察者、工具定义与执行、Agent 主循环
│   ├── eval/             指标、基线、基准与消融运行器
│   └── ui/app.py         Gradio Web UI
└── data/                 运行时目录（git-ignored：缓存、模型、评测报告）
```

***

## 📄 许可证

本项目以 **MIT License** 发布，许可证全文见 [LICENSE](LICENSE)（`pyproject.toml` 中亦声明为 MIT）。

欢迎提交 Issue 与 Pull Request。
