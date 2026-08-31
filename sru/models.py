"""Pinned revisions for the backbones the paper reports.

Freezing the data is only half of a reproducible number. A Hugging Face repo can
change under a name: weights get re-uploaded, and chat templates are edited more
often than people expect. A template edit silently rewrites every prompt the
model sees, which moves rates without touching a line of this code.

So when a run names one of the four paper backbones, it gets the revision those
rates were measured on. `--revision` overrides it, and `--revision main` opts out
and takes whatever the Hub serves today.
"""
from __future__ import annotations

import hashlib

#: model id -> the revision the published rates were measured on, plus the
#: SHA-256 of the chat template at that revision.
PAPER_MODELS: dict[str, dict[str, str]] = {
    "meta-llama/Llama-3.1-8B-Instruct": {
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "chat_template_sha256": "e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65",
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
        "chat_template_sha256": "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4",
    },
    "google/gemma-3-4b-it": {
        "revision": "093f9f388b31de276ce2de164bdc2081324b9767",
        "chat_template_sha256": "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4",
    },
    "Qwen/Qwen3-8B": {
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
    },
}

#: Short names, so `--model llama3.1-8b` works as well as the full repo id.
ALIASES: dict[str, str] = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma3-4b": "google/gemma-3-4b-it",
    "gemma3-4b-it": "google/gemma-3-4b-it",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-8b-instruct": "Qwen/Qwen3-8B",
}


def resolve(model: str) -> str:
    """Expand a short alias to its full repo id, leaving anything else alone."""
    return ALIASES.get(model.strip().lower(), model)


def canonical_id(model: str) -> str | None:
    """The paper backbone this model refers to, if any.

    Accepts the repo id, an alias, and a local checkpoint directory whose name
    ends in the repo's own path, which is how mirrored weights usually land.
    """
    resolved = resolve(model)
    if resolved in PAPER_MODELS:
        return resolved
    trimmed = resolved.rstrip("/").lower()
    for known in PAPER_MODELS:
        if trimmed.endswith(known.lower()) or trimmed.endswith(
                known.split("/")[-1].lower()):
            return known
    return None


def pinned_revision(model: str) -> str | None:
    """The revision to load for this model, or None to take the default."""
    known = canonical_id(model)
    return PAPER_MODELS[known]["revision"] if known else None


def expected_template_hash(model: str) -> str | None:
    known = canonical_id(model)
    return PAPER_MODELS[known]["chat_template_sha256"] if known else None


def template_hash(tokenizer) -> str | None:
    """SHA-256 of the tokenizer's effective chat template.

    The template is the part of the setup most likely to change without anyone
    noticing, so it is worth checking on its own rather than trusting a version
    number.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return None
    if not isinstance(template, str):  # some tokenizers carry a dict of templates
        try:
            template = template.get("default") or next(iter(template.values()))
        except Exception:
            return None
    return hashlib.sha256(template.encode("utf-8")).hexdigest()
