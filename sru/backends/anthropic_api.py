"""Claude models through the Anthropic Messages API."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from ._retry import with_retries
from .base import Backend, Messages


class AnthropicBackend(Backend):
    name = "anthropic"
    supports_loglik = False

    def __init__(self, model_id: str, api_key: str | None = None,
                 base_url: str | None = None, concurrency: int = 4,
                 temperature: float = 0.0, max_tokens: int | None = None,
                 **_ignored):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - install-time guidance
            raise ImportError(
                "the anthropic package is required: pip install 'sru[api]'") from exc

        self.model_id = model_id
        self.concurrency = max(1, concurrency)
        self.temperature = temperature
        self.max_tokens_override = max_tokens
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        url = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        if not key and not url:
            # The SDK defers this to the first request, which would surface it
            # from inside a worker thread half way through a run.
            raise ValueError("no Anthropic credentials")
        # A self-hosted endpoint usually ignores the key but the SDK still
        # insists on one.
        kwargs = {"api_key": key or "not-needed"}
        if url:
            kwargs["base_url"] = url
        self.client = anthropic.Anthropic(**kwargs)

    def _one(self, messages: Messages, max_new_tokens: int) -> str:
        # The Messages API needs max_tokens above a small floor, and counts
        # any thinking tokens against it.
        budget = max(self.max_tokens_override or max_new_tokens, 64)

        def call() -> str:
            response = self.client.messages.create(
                model=self.model_id, max_tokens=budget,
                temperature=self.temperature, messages=messages)
            parts = [b.text for b in response.content
                     if getattr(b, "type", None) == "text"]
            return "".join(parts).strip()

        return with_retries(call)

    def generate(self, conversations: list[Messages],
                 max_new_tokens: int) -> list[str]:
        if self.concurrency == 1:
            return [self._one(c, max_new_tokens) for c in conversations]
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(lambda c: self._one(c, max_new_tokens),
                                 conversations))
