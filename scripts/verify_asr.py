"""Verify the speech-recognition path without needing a real lecture.

Downloads a short public-domain speech clip (the Whisper test sample), muxes it
into an mp4 so the video has an audio track, then runs the exact production path:
`ffmpeg.probe` -> `ffmpeg.extract_audio` -> `FasterWhisperASR.transcribe`.

Usage
-----
    uv run python scripts/verify_asr.py                 # tiny model (fast)
    SKIM_WHISPER_MODEL=small uv run python scripts/verify_asr.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROXY = os.environ.get("SKIM_PROXY", "http://127.0.0.1:7892")
CLIP_URL = "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac"
EXPECTED = "ask not what your country can do for you"

WORK = Path("data/assets/asr_check")


def main() -> int:
    for var in ("HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.setdefault(var, PROXY)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from skim.config import get_settings
    from skim.media import ffmpeg
    from skim.media.asr import FasterWhisperASR

    WORK.mkdir(parents=True, exist_ok=True)
    clip = WORK / "speech.flac"
    if not clip.is_file() or clip.stat().st_size < 1000:
        print(f"downloading speech clip via {PROXY} ...")
        subprocess.run(
            ["curl.exe" if os.name == "nt" else "curl",
             "-sL", "-x", PROXY, "-o", str(clip), CLIP_URL],
            check=True,
        )
    print(f"speech clip: {clip} ({clip.stat().st_size} bytes)")

    # Give the clip a video track so we exercise the full video -> audio path.
    slide = Path("data/assets/slides/slide_01.png")
    if not slide.is_file():
        print("generating sample slides first (skim sample) ...")
        from skim.demo import build_sample

        build_sample(Path("data/assets"))

    video = WORK / "speech_with_video.mp4"
    if not video.is_file():
        ffmpeg.run_ffmpeg([
            "-y", "-loglevel", "error",
            "-loop", "1", "-i", str(slide),
            "-i", str(clip),
            "-c:v", "libx264", "-tune", "stillimage", "-r", "2",
            "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p", str(video),
        ])
    info = ffmpeg.probe(video)
    print(f"probe: {info.duration:.1f}s {info.width}x{info.height} has_audio={info.has_audio}")
    assert info.has_audio, "the generated video should carry an audio track"

    settings = get_settings()
    backend = FasterWhisperASR(
        model_size=os.environ.get("SKIM_WHISPER_MODEL", "tiny"),
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        download_root=Path(settings.data_dir) / "models",
    )

    audio = ffmpeg.extract_audio(video, WORK / "extracted_16k.wav")
    transcript = backend.transcribe(audio, duration=info.duration)

    print(f"\nlanguage={transcript.language}  segments={len(transcript)}")
    for segment in transcript.segments:
        print(f"  [{segment.start:6.2f} - {segment.end:6.2f}] {segment.text}")

    heard = transcript.full_text().lower()
    if EXPECTED in heard:
        print("\nASR PIPELINE OK")
        return 0
    print("\nASR ran, but the transcript did not match the expected phrase.")
    print("Try a larger model: SKIM_WHISPER_MODEL=small")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())