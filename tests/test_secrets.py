"""Secret handling: the API key must never leak."""

from __future__ import annotations

import logging

import pytest

from skim.config import Settings, apply_proxy, enable_fast_hf_downloads
from skim.logging_utils import REDACTION, redact, register_secret

FAKE_KEY = "sk-abcdefghijklmnopqrstuvwxyz012345"


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    monkeypatch.setenv("SKIM_PROXY", "http://127.0.0.1:7892")
    monkeypatch.setenv("SKIM_DATA_DIR", str(tmp_path / "data"))
    return Settings(_env_file=None)


def test_masked_key_hides_the_middle(settings):
    masked = settings.masked_key
    assert FAKE_KEY not in masked
    assert masked.startswith("sk-abcd")
    assert masked.endswith("2345")
    assert "..." in masked


def test_repr_never_contains_the_key(settings):
    assert FAKE_KEY not in repr(settings)
    assert FAKE_KEY not in str(settings.api_key)
    assert settings.api_key.get_secret_value() == FAKE_KEY


def test_settings_require_a_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("SKIM_API_KEY", raising=False)
    with pytest.raises(Exception):
        Settings(_env_file=None)


def test_redaction_filter_scrubs_explicit_secret():
    register_secret(FAKE_KEY)
    assert FAKE_KEY not in redact(f"using key {FAKE_KEY} now")


def test_redaction_filter_scrubs_unknown_sk_tokens():
    assert "sk-unknownrandomvalue123" not in redact("token=sk-unknownrandomvalue123")


def test_log_records_are_scrubbed(caplog):
    register_secret(FAKE_KEY)
    logger = logging.getLogger("skim.test")
    logger.addFilter(REDACTION)
    with caplog.at_level(logging.INFO, logger="skim.test"):
        logger.info("calling api with %s", FAKE_KEY)
    assert FAKE_KEY not in caplog.text
    assert "REDACTED" in caplog.text


def test_apply_proxy_exports_environment(settings, monkeypatch):
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    apply_proxy(settings)
    import os

    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7892"


def test_apply_proxy_overrides_a_pre_existing_proxy(settings, monkeypatch):
    """A stale HTTP_PROXY on the machine must not win over SKIM_PROXY.

    Regression: this silently broke Hugging Face model downloads, because the
    host had HTTP_PROXY pointing at a different, non-working port.
    """
    import os

    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    apply_proxy(settings)
    assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:7892"
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7892"


def test_apply_proxy_keeps_loopback_off_the_proxy(settings, monkeypatch):
    import os

    monkeypatch.delenv("NO_PROXY", raising=False)
    apply_proxy(settings)
    assert "127.0.0.1" in os.environ["NO_PROXY"]


def test_apply_proxy_is_a_noop_without_a_proxy(monkeypatch, tmp_path):
    import os

    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    monkeypatch.setenv("SKIM_PROXY", "")
    monkeypatch.setenv("SKIM_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    apply_proxy(Settings(_env_file=None))
    assert "HTTP_PROXY" not in os.environ


def test_proxy_is_optional(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    monkeypatch.setenv("SKIM_PROXY", "")
    monkeypatch.setenv("SKIM_DATA_DIR", str(tmp_path))
    empty = Settings(_env_file=None)
    assert empty.proxy == ""


def test_enable_fast_hf_downloads_turns_it_on_when_available(monkeypatch):
    import os

    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)
    enabled = enable_fast_hf_downloads()
    if enabled:  # hf_xet ships with huggingface_hub 1.x
        assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"


def test_enable_fast_hf_downloads_respects_an_explicit_off(monkeypatch):
    import os

    monkeypatch.setenv("HF_XET_HIGH_PERFORMANCE", "0")
    assert enable_fast_hf_downloads() is False
    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "0"


def test_enable_fast_hf_downloads_respects_an_explicit_on(monkeypatch):
    monkeypatch.setenv("HF_XET_HIGH_PERFORMANCE", "1")
    assert enable_fast_hf_downloads() is True