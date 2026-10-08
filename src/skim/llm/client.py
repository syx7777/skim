"""OpenAI-compatible chat client with retries, usage accounting and JSON helpers.

Verified against the DeepSeek gateway used by this project:

* ``deepseek-flash`` accepts **text + image** content, so the same model serves
  both the reasoning LLM and the VLM observer.
* Native function/tool calling works.
* ``response_format={"type": "json_object"}`` works.
* Reasoning can be switched off with ``thinking={"type": "disabled"}`` (fastest
  and cheapest) or dialled with ``reasoning_effort`` -- important because
  reasoning tokens otherwise eat into ``max_tokens`` and can yield empty text.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Iterable, Sequence

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import Settings, apply_proxy, get_settings
from ..logging_utils import get_logger, register_secret
from .types import ImageAttachment, LLMResult, ToolCall, Usage

logger = get_logger("llm")

#: Maps a friendly reasoning level onto the provider's request body.
REASONING_BODIES: dict[str, dict[str, Any] | None] = {
    "off": {"thinking": {"type": "disabled"}},
    "low": {"reasoning_effort": "low"},
    "medium": {"reasoning_effort": "medium"},
    "high": {"reasoning_effort": "high"},
    "max": {"reasoning_effort": "max"},
    "default": None,
}


class UsageTracker:
    """Accumulates token/call statistics across a whole agent run."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.usage = Usage()
        self.llm_usage = Usage()
        self.vlm_usage = Usage()
        self.llm_calls = 0
        self.vlm_calls = 0
        self.frames_sent = 0

    def add(self, result: LLMResult) -> None:
        self.usage = self.usage + result.usage
        if result.kind == "vlm":
            self.vlm_calls += 1
            self.vlm_usage = self.vlm_usage + result.usage
            self.frames_sent += result.usage.images
        else:
            self.llm_calls += 1
            self.llm_usage = self.llm_usage + result.usage
        self.records.append(
            {
                "kind": result.kind,
                "model": result.model,
                "latency_s": round(result.latency_s, 3),
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "images": result.usage.images,
            }
        )

    @property
    def total_calls(self) -> int:
        return self.llm_calls + self.vlm_calls

    @property
    def latency_s(self) -> float:
        return sum(r["latency_s"] for r in self.records)

    @property
    def vlm_latency_s(self) -> float:
        return sum(r["latency_s"] for r in self.records if r["kind"] == "vlm")

    def summary(self) -> dict[str, Any]:
        """Real, billable numbers straight from the API responses.

        ``visual_tokens`` is the prompt-token count of the vision calls. Images
        dominate those prompts, so it is the closest usable proxy for what the
        provider actually charges for the visual modality.
        """
        return {
            "llm_calls": self.llm_calls,
            "vlm_calls": self.vlm_calls,
            "total_calls": self.total_calls,
            "frames_sent": self.frames_sent,
            "api_latency_s": round(self.latency_s, 2),
            "visual_tokens": self.vlm_usage.prompt_tokens,
            "llm_prompt_tokens": self.llm_usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "reasoning_tokens": self.usage.reasoning_tokens,
            "total_tokens": self.usage.total_tokens,
        }


_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, httpx.TimeoutException)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction from a model response."""
    if not text:
        raise ValueError("empty model response")
    candidate = text.strip()

    fenced = _FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fall back to scanning for the first balanced {...} or [...] block.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(candidate)):
            char = candidate[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(candidate[start : idx + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"could not parse JSON from response: {text[:400]!r}")


class LLMClient:
    """Thin, resilient wrapper around the chat-completions endpoint."""

    def __init__(self, settings: Settings | None = None, usage: UsageTracker | None = None):
        self.settings = settings or get_settings()
        apply_proxy(self.settings)
        register_secret(self.settings.api_key.get_secret_value())

        http_client = httpx.Client(
            proxy=self.settings.proxy or None,
            timeout=httpx.Timeout(self.settings.request_timeout_s, connect=30.0),
        )
        self._client = OpenAI(
            api_key=self.settings.api_key.get_secret_value(),
            base_url=self.settings.base_url,
            http_client=http_client,
            max_retries=0,  # tenacity owns retries so we can log them
        )
        self.usage = usage or UsageTracker()

    # ------------------------------------------------------------------ core
    @retry(
        retry=retry_if_exception_type(_RETRYABLE + (APIStatusError,)),
        wait=wait_exponential(multiplier=1.5, min=2, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _create(self, **kwargs: Any) -> Any:
        return self._client.chat.completions.create(**kwargs)

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        json_mode: bool = False,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        reasoning: str = "off",
        kind: str = "llm",
        images: int = 0,
    ) -> LLMResult:
        """Single chat completion with retries and usage accounting."""
        model = model or (self.settings.vlm_model if kind == "vlm" else self.settings.llm_model)
        body = REASONING_BODIES.get(reasoning, None)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if body:
            kwargs["extra_body"] = body

        started = time.perf_counter()
        try:
            response = self._create(**kwargs)
        except Exception as exc:  # pragma: no cover - network path
            logger.error("LLM call failed after retries: %s", type(exc).__name__)
            raise

        latency = time.perf_counter() - started
        choice = response.choices[0]
        message = choice.message

        calls: list[ToolCall] = []
        for raw in getattr(message, "tool_calls", None) or []:
            try:
                args = json.loads(raw.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": raw.function.arguments}
            calls.append(ToolCall(id=raw.id, name=raw.function.name, arguments=args))

        usage = Usage()
        if response.usage is not None:
            details = getattr(response.usage, "completion_tokens_details", None)
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                reasoning_tokens=getattr(details, "reasoning_tokens", 0) or 0,
                total_tokens=response.usage.total_tokens or 0,
                images=images,
            )

        result = LLMResult(
            content=message.content or "",
            tool_calls=calls,
            usage=usage,
            latency_s=latency,
            model=model,
            finish_reason=getattr(choice, "finish_reason", "") or "",
            kind=kind,
        )
        self.usage.add(result)
        return result

    def chat_json(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 4096,
        reasoning: str = "off",
        model: str | None = None,
        kind: str = "llm",
        retries: int = 1,
    ) -> Any:
        """Chat completion that must return parseable JSON."""
        attempt = 0
        last_error: Exception | None = None
        while attempt <= retries:
            result = self.chat(
                messages,
                json_mode=True,
                max_tokens=max_tokens,
                reasoning=reasoning,
                model=model,
                kind=kind,
            )
            try:
                return extract_json(result.content)
            except ValueError as exc:
                last_error = exc
                logger.warning("JSON parse failed (attempt %d): %s", attempt + 1, exc)
                attempt += 1
                if attempt <= retries:
                    messages = messages + [
                        {"role": "assistant", "content": result.content},
                        {
                            "role": "user",
                            "content": "Your previous answer was not valid JSON. "
                            "Reply with a single valid JSON object and nothing else.",
                        },
                    ]
        raise ValueError(f"model did not return valid JSON: {last_error}")

    # ---------------------------------------------------------------- vision
    def vision(
        self,
        prompt: str,
        images: Sequence[ImageAttachment],
        *,
        system: str | None = None,
        json_mode: bool = True,
        max_tokens: int = 3072,
        reasoning: str = "off",
    ) -> LLMResult:
        """Send one prompt plus N frames to the VLM."""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for index, image in enumerate(images):
            caption = image.label or (
                f"frame {index + 1} @ {image.timestamp:.1f}s" if image.timestamp is not None else f"frame {index + 1}"
            )
            content.append({"type": "text", "text": f"[{caption}]"})
            content.append({"type": "image_url", "image_url": {"url": image.data_url}})

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        result = self.chat(
            messages,
            json_mode=json_mode,
            max_tokens=max_tokens,
            reasoning=reasoning,
            kind="vlm",
            images=len(images),
        )
        return result

    def vision_json(
        self,
        prompt: str,
        images: Sequence[ImageAttachment],
        *,
        system: str | None = None,
        max_tokens: int = 3072,
        reasoning: str = "off",
    ) -> Any:
        result = self.vision(
            prompt, images, system=system, json_mode=True, max_tokens=max_tokens, reasoning=reasoning
        )
        try:
            return extract_json(result.content)
        except ValueError as exc:
            logger.warning("VLM returned non-JSON payload: %s", exc)
            raise

    # -------------------------------------------------------------- counters
    def embed_messages(self, messages: Iterable[dict[str, Any]]) -> int:
        """Rough character count, used for prompt-budget logging."""
        return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)