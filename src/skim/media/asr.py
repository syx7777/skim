"""Speech recognition backends producing a :class:`Transcript`."""

from __future__ import annotations

import time
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ..config import Settings, apply_proxy, enable_fast_hf_downloads, get_settings
from ..logging_utils import get_logger
from . import ffmpeg
from .transcript import Transcript, TranscriptSegment, merge_segments

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

logger = get_logger("asr")

SAMPLE_RATE = 16000


class ASRError(RuntimeError):
    pass


def load_pcm16_wav(path: Path) -> "np.ndarray | None":
    """Read a 16 kHz mono 16-bit PCM WAV into a float32 array in [-1, 1].

    Decoding this ourselves (rather than letting faster-whisper call PyAV) keeps
    the whole media pipeline on the bundled ffmpeg and avoids depending on PyAV's
    container API, which changes between major versions.
    """
    import numpy as np

    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                return None
            if handle.getframerate() != SAMPLE_RATE:
                return None
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError, OSError):
        return None
    if not frames:
        return None
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


class ASRBackend(Protocol):
    name: str

    def transcribe(self, audio_path: Path, *, duration: float = 0.0) -> Transcript: ...


class FasterWhisperASR:
    """Local Whisper inference through CTranslate2 (CPU-friendly)."""

    name = "faster-whisper"

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
        download_root: Path | None = None,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.download_root = Path(download_root) if download_root else None
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover
            raise ASRError(
                "faster-whisper is not installed. Run `uv sync` in the project root."
            ) from exc

        # Model weights come from the Hugging Face hub, which reads the standard
        # proxy environment variables -- make sure they are exported first.
        apply_proxy()
        if enable_fast_hf_downloads():
            logger.info("Hugging Face Xet high-performance transfer enabled")

        logger.info(
            "loading whisper model '%s' (device=%s, compute_type=%s)",
            self.model_size,
            self.device,
            self.compute_type,
        )

        # The first run downloads the weights, and this proxy occasionally drops
        # a TLS handshake mid-transfer. Losing that after a 10-minute video
        # download would be painful, so retry rather than abort the ingest.
        attempts = 3
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            try:
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=str(self.download_root) if self.download_root else None,
                )
            except Exception as exc:  # noqa: BLE001 - hub/network errors vary
                last_error = exc
                if attempt >= attempts:
                    break
                wait = 3 * attempt
                logger.warning(
                    "whisper model load failed (%s: %s); retrying in %ds [%d/%d]",
                    type(exc).__name__,
                    str(exc).splitlines()[0][:120],
                    wait,
                    attempt,
                    attempts,
                )
                time.sleep(wait)
                continue

            logger.info("whisper model ready in %.1fs", time.perf_counter() - started)
            return self._model

        raise ASRError(
            f"could not load whisper model '{self.model_size}' after {attempts} attempts: "
            f"{last_error}. Check network access to huggingface.co (SKIM_PROXY) or "
            f"pre-download the model into {self.download_root}."
        ) from last_error

    def _decode(self, audio_path: Path) -> "np.ndarray":
        """Decode any audio file to a 16 kHz mono float32 array via ffmpeg."""
        samples = load_pcm16_wav(audio_path)
        if samples is not None:
            return samples
        # Anything that is not already a 16 kHz mono PCM WAV gets normalised by
        # the bundled ffmpeg first, so PyAV is never involved.
        converted = audio_path.with_name(audio_path.stem + ".16k.wav")
        ffmpeg.extract_audio(audio_path, converted, sample_rate=SAMPLE_RATE)
        samples = load_pcm16_wav(converted)
        if samples is None:
            raise ASRError(f"could not decode audio: {audio_path}")
        return samples

    def transcribe(self, audio_path: Path, *, duration: float = 0.0) -> Transcript:
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise ASRError(f"audio file not found: {audio_path}")

        model = self._load()
        samples = self._decode(audio_path)

        started = time.perf_counter()
        raw_segments, info = model.transcribe(
            samples,
            language=self.language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
        )

        segments: list[TranscriptSegment] = []
        for index, segment in enumerate(raw_segments, start=1):
            text = (segment.text or "").strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(start=float(segment.start), end=float(segment.end), text=text)
            )
            if index % 200 == 0:
                logger.info("  ... transcribed %d segments (t=%.0fs)", index, segment.end)

        elapsed = time.perf_counter() - started
        logger.info(
            "transcribed %d raw segments in %.1fs (%.1fx realtime)",
            len(segments),
            elapsed,
            (duration / elapsed) if elapsed > 0 and duration else 0.0,
        )

        merged = merge_segments(segments)
        language = getattr(info, "language", None) or self.language or "unknown"
        return Transcript(
            segments=merged,
            language=language,
            source="asr",
            duration=max(duration, float(getattr(info, "duration", 0.0) or 0.0)),
        )


class TranscriptOnlyASR:
    """Placeholder backend used when a subtitle file is supplied instead of ASR."""

    name = "subtitle"

    def transcribe(self, audio_path: Path, *, duration: float = 0.0) -> Transcript:  # pragma: no cover
        raise ASRError("no ASR backend configured; supply a subtitle file instead")


def build_backend(settings: Settings | None = None) -> ASRBackend:
    settings = settings or get_settings()
    return FasterWhisperASR(
        model_size=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        download_root=Path(settings.data_dir) / "models",
    )