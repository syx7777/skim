"""Configuration, secret handling and on-disk layout.

Security notes
--------------
* The API key is only ever read from the process environment or a git-ignored
  ``.env`` file. It is never hard-coded, never logged and never persisted into
  any run artefact (see :mod:`skim.logging_utils` for redaction).
* ``SecretStr`` is used so that an accidental ``repr(settings)`` or
  ``print(settings)`` emits ``**********`` instead of the key.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_env_file() -> Path | None:
    """Locate a ``.env`` file without depending on the current directory."""
    explicit = os.environ.get("SKIM_ENV_FILE")
    if explicit and Path(explicit).is_file():
        return Path(explicit)

    candidates: list[Path] = [Path.cwd() / ".env"]
    # .../src/skim/config.py -> project root is three levels up.
    here = Path(__file__).resolve()
    for parent in list(here.parents)[:4]:
        candidates.append(parent / ".env")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


class Settings(BaseSettings):
    """Application settings, resolved from environment variables."""

    model_config = SettingsConfigDict(
        env_file=_find_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- credentials -------------------------------------------------------
    api_key: SecretStr = Field(
        ...,
        validation_alias=AliasChoices("DEEPSEEK_API_KEY", "SKIM_API_KEY"),
        description="OpenAI-compatible API key. Required.",
    )
    base_url: str = Field(
        "https://api.deepseek.com",
        validation_alias=AliasChoices("DEEPSEEK_BASE_URL", "SKIM_BASE_URL"),
    )

    # --- models ------------------------------------------------------------
    llm_model: str = Field("deepseek-flash", validation_alias="SKIM_LLM_MODEL")
    vlm_model: str = Field("deepseek-flash", validation_alias="SKIM_VLM_MODEL")

    # --- network -----------------------------------------------------------
    proxy: str = Field("", validation_alias="SKIM_PROXY")

    # --- storage -----------------------------------------------------------
    data_dir: Path = Field(Path("data"), validation_alias="SKIM_DATA_DIR")

    # --- remote video sources ----------------------------------------------
    #: Height cap when downloading a lecture from a URL. 720p keeps blackboard
    #: handwriting and formulas legible; 1080p roughly doubles the download.
    video_max_height: int = Field(720, validation_alias="SKIM_VIDEO_MAX_HEIGHT")

    # --- ASR ---------------------------------------------------------------
    whisper_model: str = Field("small", validation_alias="SKIM_WHISPER_MODEL")
    whisper_device: str = Field("cpu", validation_alias="SKIM_WHISPER_DEVICE")
    whisper_compute_type: str = Field("int8", validation_alias="SKIM_WHISPER_COMPUTE_TYPE")

    # --- observability -----------------------------------------------------
    log_level: str = Field("INFO", validation_alias="SKIM_LOG_LEVEL")

    # --- budget guards -----------------------------------------------------
    max_agent_steps: int = Field(8, validation_alias="SKIM_MAX_AGENT_STEPS")
    max_vlm_frames_per_call: int = Field(4, validation_alias="SKIM_MAX_VLM_FRAMES")
    request_timeout_s: float = Field(180.0, validation_alias="SKIM_REQUEST_TIMEOUT")

    @property
    def masked_key(self) -> str:
        """A safe representation of the key, e.g. ``sk-76b4...c4c6``."""
        raw = self.api_key.get_secret_value()
        if len(raw) <= 12:
            return "***"
        return f"{raw[:7]}...{raw[-4:]}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


def apply_proxy(settings: Settings | None = None) -> None:
    """Export the configured proxy to the whole process.

    Libraries that read the environment (``huggingface_hub`` downloading ASR
    weights, ``yt-dlp``) only see ``SKIM_PROXY`` through these variables, so this
    must be an *assignment*, not a ``setdefault``: a machine can easily already
    have ``HTTP_PROXY`` pointing at a different, non-working port, and silently
    keeping that value breaks model downloads in a very confusing way.
    """
    settings = settings or get_settings()
    if not settings.proxy:
        return
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ[var] = settings.proxy
    # Keep loopback traffic (the Gradio server, local files) off the proxy.
    for var in ("NO_PROXY", "no_proxy"):
        os.environ.setdefault(var, "localhost,127.0.0.1,::1")


def enable_fast_hf_downloads() -> bool:
    """Ask Hugging Face to use its high-performance (Xet) transfer path.

    ``huggingface_hub`` 1.x moved to Xet for chunked, parallel downloads and
    deprecated ``HF_HUB_ENABLE_HF_TRANSFER``. That parallelisation matters a lot
    on a throttled link: this project's proxy caps a *single* connection at
    roughly 110 KB/s, so a ~480 MB whisper model would take over an hour
    single-stream (measured: 4 parallel range requests reach ~460 KB/s).

    An explicit user setting always wins. Returns whether high-performance
    transfer is active.
    """
    explicit = os.environ.get("HF_XET_HIGH_PERFORMANCE")
    if explicit is not None:
        return explicit.strip().lower() not in ("", "0", "false", "no")
    try:
        import hf_xet  # noqa: F401
    except ImportError:  # pragma: no cover - older huggingface_hub
        return False
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    return True


class PathLayout:
    """Resolved directory layout under ``data/``."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.uploads = self.root / "uploads"
        self.downloads = self.root / "downloads"
        self.cache = self.root / "cache"
        self.runs = self.root / "runs"
        self.assets = self.root / "assets"

    def create(self) -> "PathLayout":
        for directory in (
            self.root,
            self.uploads,
            self.downloads,
            self.cache,
            self.runs,
            self.assets,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def video_cache(self, video_id: str) -> Path:
        """Per-video cache directory (transcript, chapters, frames)."""
        path = self.cache / video_id
        path.mkdir(parents=True, exist_ok=True)
        return path


def get_layout(settings: Settings | None = None) -> PathLayout:
    settings = settings or get_settings()
    return PathLayout(settings.data_dir).create()