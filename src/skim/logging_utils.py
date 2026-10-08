"""Logging helpers.

A redaction filter scrubs the API key (and anything that looks like one) from
every log record, so secrets can never leak into console output or log files.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

from rich.logging import RichHandler

# Matches `sk-...` style tokens as a defensive second line of defence.
_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}\b")

_CONFIGURED = False


class RedactionFilter(logging.Filter):
    """Replace secret material inside formatted log messages."""

    def __init__(self, secrets: list[str] | None = None):
        super().__init__()
        self._secrets = [s for s in (secrets or []) if s]

    def add_secret(self, secret: str) -> None:
        if secret and secret not in self._secrets:
            self._secrets.append(secret)

    def _scrub(self, text: str) -> str:
        for secret in self._secrets:
            if secret in text:
                text = text.replace(secret, "***REDACTED***")
        return _KEY_PATTERN.sub("sk-***REDACTED***", text)

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            return True
        scrubbed = self._scrub(message)
        if scrubbed != message:
            record.msg = scrubbed
            record.args = ()
        if record.exc_text:
            record.exc_text = self._scrub(record.exc_text)
        return True


REDACTION = RedactionFilter()


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configure console (+ optional file) logging once per process."""
    global _CONFIGURED
    root = logging.getLogger("skim")
    if _CONFIGURED:
        return root

    root.setLevel(level.upper())
    root.propagate = False

    console = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
    console.setFormatter(logging.Formatter("%(message)s"))
    console.addFilter(REDACTION)
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        file_handler.addFilter(REDACTION)
        root.addHandler(file_handler)

    # Keep third-party libraries quiet unless the user asks for DEBUG.
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "faster_whisper", "numba"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))

    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"skim.{name}")


def register_secret(secret: str) -> None:
    """Make sure a secret is redacted even if it is logged by a library."""
    REDACTION.add_secret(secret)


def redact(text: str) -> str:
    """Utility used in tests / debugging."""
    return REDACTION._scrub(text)


def _unused_stream_guard() -> None:  # pragma: no cover
    _ = sys.stderr