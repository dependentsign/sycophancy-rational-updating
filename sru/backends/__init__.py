"""Backend construction."""
from __future__ import annotations

from .base import Backend, Messages

BACKENDS = ("hf", "openai", "anthropic")


def resolve_backend(model_id: str, backend: str = "auto") -> str:
    """Pick a backend from the model id when the user did not name one."""
    if backend != "auto":
        if backend not in BACKENDS:
            raise KeyError(f"unknown backend {backend!r}; choose from {BACKENDS}")
        return backend
    lowered = model_id.lower()
    if lowered.startswith("claude"):
        return "anthropic"
    if "/" in model_id or lowered.endswith(".gguf"):
        # Looks like a Hugging Face repo id or a local path.
        return "hf"
    if lowered.startswith(("gpt-", "o1", "o3", "o4", "chatgpt", "deepseek")):
        return "openai"
    return "hf"


def build_backend(model_id: str, backend: str = "auto", **kwargs) -> Backend:
    """Instantiate a backend. Imports lazily so that using an API model does
    not require torch to be installed, and vice versa."""
    name = resolve_backend(model_id, backend)
    if name == "hf":
        from .hf import HFBackend
        return HFBackend(model_id, **kwargs)
    if name == "openai":
        from .openai_api import OpenAIBackend
        return OpenAIBackend(model_id, **kwargs)
    from .anthropic_api import AnthropicBackend
    return AnthropicBackend(model_id, **kwargs)


__all__ = ["Backend", "Messages", "BACKENDS", "build_backend", "resolve_backend"]
