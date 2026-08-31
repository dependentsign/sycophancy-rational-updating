"""Any OpenAI-compatible chat endpoint.

This covers the OpenAI API itself and every server that mirrors it, including
vLLM, SGLang, Ollama, LM Studio, Together, Fireworks, DeepSeek, and OpenRouter.
Point `--base-url` at the server and the rest of the tool is unchanged.

Chat APIs return log-probabilities only for tokens they generated, never for a
continuation you supply, so this backend cannot do the paper's log-likelihood
scoring. TruthfulQA falls back to letter scoring automatically.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from ._retry import with_retries
from .base import Backend, Messages


class OpenAIBackend(Backend):
    name = "openai"
    supports_loglik = False

    def __init__(self, model_id: str, base_url: str | None = None,
                 api_key: str | None = None, concurrency: int = 8,
                 temperature: float = 0.0, max_tokens: int | None = None,
                 **_ignored):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - install-time guidance
            raise ImportError(
                "the openai package is required: pip install 'sru[api]'") from exc

        self.model_id = model_id
        self.concurrency = max(1, concurrency)
        self.temperature = temperature
        self.max_tokens_override = max_tokens
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )
        # Learned on the first rejection and reused afterwards, so a model that
        # refuses these parameters costs one wasted call rather than one per item.
        self._omit_temperature = False
        self._use_completion_tokens = False

    def _one(self, messages: Messages, max_new_tokens: int) -> str:
        budget = self.max_tokens_override or max_new_tokens

        def call() -> str:
            kwargs: dict = {"model": self.model_id, "messages": messages}
            if not self._omit_temperature:
                kwargs["temperature"] = self.temperature
            kwargs["max_completion_tokens" if self._use_completion_tokens
                   else "max_tokens"] = budget
            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                text = str(exc).lower()
                retry = False
                if "temperature" in text and not self._omit_temperature:
                    self._omit_temperature, retry = True, True
                if "max_tokens" in text and not self._use_completion_tokens:
                    self._use_completion_tokens, retry = True, True
                if not retry:
                    raise
                return call()
            return (response.choices[0].message.content or "").strip()

        return with_retries(call)

    def generate(self, conversations: list[Messages],
                 max_new_tokens: int) -> list[str]:
        if self.concurrency == 1:
            return [self._one(c, max_new_tokens) for c in conversations]
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(lambda c: self._one(c, max_new_tokens),
                                 conversations))
