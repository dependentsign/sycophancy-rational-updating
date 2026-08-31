"""Backend interface.

A backend is anything that can (a) continue a chat and (b) optionally score a
fixed set of candidate continuations. Local weights do both; a chat API does
only the first, which is why TruthfulQA has two scoring paths.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

Messages = list[dict[str, str]]


class Backend(ABC):
    """Greedy chat completion, plus optional candidate scoring."""

    #: Short backend name, recorded in the run config.
    name: str = "backend"
    #: The model identifier as the user typed it.
    model_id: str = ""
    #: Whether score_choices() is available. False for chat APIs, which do not
    #: expose the log-probability of an arbitrary continuation.
    supports_loglik: bool = False

    @abstractmethod
    def generate(self, conversations: list[Messages],
                 max_new_tokens: int) -> list[str]:
        """Continue each conversation greedily. Returns one string each."""

    def score_choices(self, conversation: Messages,
                      choices: list[str]) -> list[float]:
        """Length-normalised log-likelihood of each choice as the next
        assistant message."""
        raise NotImplementedError(
            f"{self.name} backend cannot score candidate continuations; "
            "use --tqa-scoring letter")

    def close(self) -> None:
        """Release resources. Safe to call more than once."""
