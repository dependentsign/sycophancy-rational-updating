"""Local weights through Hugging Face transformers.

This is the path the paper's numbers were produced on: greedy decoding, no
system prompt, and length-normalised log-likelihood scoring for the
multiple-choice condition.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..models import template_hash
from .base import Backend, Messages

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32, "auto": "auto"}


def _dtype_kwarg() -> str:
    """`torch_dtype` was renamed to `dtype` in transformers 4.56 and warns
    loudly from 5.0 on. Pick whichever the installed version wants."""
    try:
        from transformers import __version__ as version
        major, minor = (int(part) for part in version.split(".")[:2])
    except Exception:
        return "dtype"
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


class HFBackend(Backend):
    name = "hf"
    supports_loglik = True

    def __init__(self, model_id: str, dtype: str = "bfloat16",
                 device_map: str = "auto", batch_size: int = 8,
                 trust_remote_code: bool = True, revision: str | None = None,
                 **_ignored):
        self.model_id = model_id
        self.batch_size = max(1, batch_size)
        self.revision = revision
        # A local checkpoint directory has no revisions to ask for.
        at_revision = ({"revision": revision}
                       if revision and not Path(model_id).exists() else {})
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code, **at_revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.template_sha256 = template_hash(self.tokenizer)

        kwargs = {"trust_remote_code": trust_remote_code,
                  _dtype_kwarg(): DTYPES.get(dtype, torch.bfloat16),
                  **at_revision}
        # device_map needs accelerate; "none" loads the plain model and moves it
        # by hand, which is enough for a single device and one less dependency.
        placement = None
        if device_map and device_map.lower() != "none":
            kwargs["device_map"] = device_map
        else:
            mps = getattr(torch.backends, "mps", None)
            placement = ("cuda" if torch.cuda.is_available()
                         else "mps" if mps is not None and mps.is_available()
                         else "cpu")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except ValueError as causal_error:
            # Multimodal checkpoints (Gemma 3, Llama 3.2 Vision, Qwen-VL) are
            # not registered for causal LM; their text stack still is. Anything
            # else that raises ValueError here is a real problem, so if the
            # fallback does not fit either, report the original error rather
            # than the fallback's complaint about the wrong architecture.
            from transformers import AutoModelForImageTextToText
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id, **kwargs)
            except Exception:
                raise causal_error from None
        if placement is not None:
            self.model.to(placement)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    # -- chat templating ----------------------------------------------------

    def _template(self, messages: Messages) -> torch.Tensor:
        """Render a conversation to token ids, one row, on the model device.

        `enable_thinking=False` turns off Qwen3's default reasoning block so
        that the reply is the answer itself. Templates that do not take the
        argument raise TypeError and are called again without it.
        """
        kwargs = dict(tokenize=True, add_generation_prompt=True,
                      return_tensors="pt")
        try:
            out = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs)
        except TypeError:
            out = self.tokenizer.apply_chat_template(messages, **kwargs)
        ids = out.input_ids if hasattr(out, "input_ids") else out
        return ids.to(self.device)

    # -- generation ---------------------------------------------------------

    @torch.no_grad()
    def generate(self, conversations: list[Messages],
                 max_new_tokens: int) -> list[str]:
        out: list[str] = []
        for i in range(0, len(conversations), self.batch_size):
            out.extend(self._generate_batch(
                conversations[i:i + self.batch_size], max_new_tokens))
        return out

    @torch.no_grad()
    def _generate_batch(self, conversations: list[Messages],
                        max_new_tokens: int) -> list[str]:
        seqs = [self._template(c)[0] for c in conversations]
        pad_id = self.tokenizer.pad_token_id
        width = max(s.shape[0] for s in seqs)

        # Left padding: generation continues from the final position, so the
        # padding has to sit in front of the prompt, not behind it.
        input_ids = torch.full((len(seqs), width), pad_id,
                               dtype=torch.long, device=self.device)
        attention = torch.zeros((len(seqs), width), dtype=torch.long,
                                device=self.device)
        for i, s in enumerate(seqs):
            input_ids[i, width - s.shape[0]:] = s
            attention[i, width - s.shape[0]:] = 1

        generated = self.model.generate(
            input_ids=input_ids, attention_mask=attention,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=pad_id)
        return [self.tokenizer.decode(row[width:], skip_special_tokens=True).strip()
                for row in generated]

    # -- candidate scoring --------------------------------------------------

    @torch.no_grad()
    def score_choices(self, conversation: Messages,
                      choices: list[str]) -> list[float]:
        """Mean token log-probability of each choice, continuing the prompt.

        Length normalisation keeps long options from being penalised purely for
        being long. One forward pass per option: batching them would multiply
        the peak logits tensor by the option count, which on a large vocabulary
        is gigabytes, and multiple-choice is the cheap part of a run anyway.
        """
        prompt = self._template(conversation)
        n_prompt = prompt.shape[1]
        scores: list[float] = []

        for choice in choices:
            ids = self.tokenizer(choice, return_tensors="pt",
                                 add_special_tokens=False).input_ids.to(self.device)
            if ids.shape[1] == 0:
                scores.append(float("-inf"))
                continue
            full = torch.cat([prompt, ids], dim=1)
            logits = self.model(full).logits
            targets = full[0, n_prompt:]
            # Position t predicts token t+1, so the window starts one before
            # the continuation.
            logprobs = F.log_softmax(logits[0, n_prompt - 1:-1, :].float(), dim=-1)
            index = torch.arange(targets.shape[0], device=logprobs.device)
            token_logprobs = logprobs[index, targets]
            scores.append(float(token_logprobs.sum().item()) / targets.shape[0])
        return scores

    def close(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
