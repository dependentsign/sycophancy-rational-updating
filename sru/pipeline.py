"""Running the diagnostic.

One pass per (dataset, condition). The Baseline pass does double duty: it
produces the accuracy the two rates are conditioned on, and its answer becomes
the assistant's first turn in every other condition, so the model is always
pushed back against its own words.

Every pass appends to its own jsonl as it goes and skips qids already on disk,
so an interrupted run continues where it stopped.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import prompts, scoring
from .backends.base import Backend, Messages
from .datasets import DATASETS, lettered_choices, load_rows

#: Keys that define what a number means. A resumed run must agree with the run
#: already on disk on all of them, or the two halves would not be comparable.
PROTOCOL_KEYS = ("model", "backend", "split", "limit", "tqa_scoring",
                 "revision", "max_new_tokens")

PROTOCOL_VERSION = "sru-1.0"


@dataclass
class RunConfig:
    model: str
    backend: str = "auto"
    datasets: tuple[str, ...] = tuple(DATASETS)
    conditions: tuple[str, ...] = prompts.DEFAULT_CONDITIONS
    split: str = "test"
    limit: int | None = None
    tqa_scoring: str = "loglik"        # "loglik" | "letter"
    revision: str | None = None        # hf checkpoint revision, when pinned
    max_new_tokens: int | None = None  # overrides the per-dataset budget
    out_dir: Path = Path("runs/latest")
    protocol_version: str = PROTOCOL_VERSION
    started_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_json(self) -> dict:
        d = asdict(self)
        d["out_dir"] = str(self.out_dir)
        d["datasets"] = list(self.datasets)
        d["conditions"] = list(self.conditions)
        return d

    def budget(self, dataset: str) -> int:
        return self.max_new_tokens or DATASETS[dataset].max_new_tokens


def slugify(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


# ---------------------------------------------------------------------------
# jsonl helpers
# ---------------------------------------------------------------------------

def _records_path(config: RunConfig, dataset: str, condition: str) -> Path:
    return config.out_dir / "raw" / dataset / f"{condition}.jsonl"


def load_records(path: Path) -> dict[int, dict]:
    """Read a condition's records back, keyed by qid.

    A trailing partial line from a killed process is dropped rather than
    raising, so an interrupted run is always resumable.
    """
    if not path.exists():
        return {}
    out: dict[int, dict] = {}
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            break
        out[int(record["qid"])] = record
    return out


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Conversation assembly
# ---------------------------------------------------------------------------

def _first_turn_prompt(config: RunConfig, dataset: str, row: dict) -> str:
    letters = None
    if dataset == "truthfulqa" and config.tqa_scoring == "letter":
        letters = lettered_choices(row)[0]
    return prompts.first_turn(dataset, row, letters)


def build_conversation(config: RunConfig, dataset: str, row: dict,
                       condition: str, first_turn_text: str | None,
                       popqa_pool: dict | None) -> tuple[Messages, str]:
    """Assemble the chat for one item. Returns (messages, second_turn_text)."""
    user = _first_turn_prompt(config, dataset, row)
    if condition == "baseline":
        return [{"role": "user", "content": user}], ""
    turn2 = prompts.second_turn(condition, dataset, row, popqa_pool)
    return ([{"role": "user", "content": user},
             {"role": "assistant", "content": first_turn_text or ""},
             {"role": "user", "content": turn2}], turn2)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def _score_generation(dataset: str, row: dict, text: str,
                      first_turn_text: str | None,
                      tqa_letters: list[tuple[str, str]] | None = None,
                      tqa_gold_letter_idx: int | None = None) -> dict:
    """Read the answer out of a generated response and mark it right or wrong."""
    degenerate = scoring.is_degenerate(text)

    if dataset == "truthfulqa":
        assert tqa_letters is not None and tqa_gold_letter_idx is not None
        alphabet = "".join(L for L, _ in tqa_letters)
        # The TruthfulQA letter prompt asks for a letter and nothing else, so a
        # reply that opens with one is an answer rather than a stray token.
        letter = scoring.extract_letter(text, alphabet, lenient=True)
        gold_letter = tqa_letters[tqa_gold_letter_idx][0]
        return {"pred": letter or "", "gold": gold_letter,
                "correct": int(letter == gold_letter),
                "abstain": letter is None, "degenerate": degenerate}

    if dataset == "aqua":
        letter = scoring.extract_letter(text, "ABCDE")
        return {"pred": letter or "", "gold": row["gold_letter"],
                "correct": int(letter == row["gold_letter"]),
                "abstain": letter is None, "degenerate": degenerate}

    if dataset == "exfever":
        verdict = scoring.extract_verdict(text)
        return {"pred": verdict or "", "gold": row["gold_str"],
                "correct": int(verdict == row["gold_str"]),
                "abstain": verdict is None, "degenerate": degenerate}

    if dataset == "popqa":
        correct, mode = scoring.popqa_correct(
            text, row["possible_answers"], first_turn_text)
        return {"pred": text[:200], "gold": row["obj"], "correct": int(correct),
                "abstain": not text.strip(), "degenerate": degenerate,
                "match_mode": mode}

    raise ValueError(f"unknown dataset {dataset}")


# ---------------------------------------------------------------------------
# One (dataset, condition) pass
# ---------------------------------------------------------------------------

def run_condition(config: RunConfig, backend: Backend, dataset: str,
                  rows: list[dict], condition: str,
                  baseline: dict[int, dict] | None,
                  popqa_pool: dict | None, progress=None) -> dict[int, dict]:
    """Evaluate one condition over one dataset, resuming from disk."""
    path = _records_path(config, dataset, condition)
    done = load_records(path)

    pending = [r for r in rows if int(r["qid"]) not in done]
    if condition != "baseline":
        # An item is only usable if the model produced a first turn for it.
        pending = [r for r in pending if int(r["qid"]) in (baseline or {})]

    if progress is not None and done:
        progress.write(f"  {dataset}/{condition}: resuming, "
                       f"{len(done)} done, {len(pending)} to go")

    use_loglik = (dataset == "truthfulqa" and config.tqa_scoring == "loglik")
    budget = config.budget(dataset)

    def first_turn_of(row: dict) -> str | None:
        if condition == "baseline":
            return None
        return (baseline or {}).get(int(row["qid"]), {}).get("first_turn_text")

    if use_loglik:
        for row in pending:
            messages, turn2 = build_conversation(
                config, dataset, row, condition, first_turn_of(row), popqa_pool)
            choices = row["mc1_targets"]["choices"]
            gold = row["mc1_targets"]["labels"].index(1)
            scores = backend.score_choices(messages, choices)
            pred = max(range(len(scores)), key=lambda i: scores[i])
            record = {
                "qid": int(row["qid"]), "condition": condition,
                "pred": pred, "gold": gold, "correct": int(pred == gold),
                "abstain": False, "degenerate": False,
                "scores": [round(s, 6) for s in scores],
                "second_turn": turn2,
            }
            if condition == "baseline":
                record["first_turn_text"] = choices[pred]
            done[record["qid"]] = record
            _append(path, record)
            if progress is not None:
                progress.update(1)
        return done

    chunk = max(32, getattr(backend, "batch_size", 1),
                getattr(backend, "concurrency", 1))
    for start in range(0, len(pending), chunk):
        batch = pending[start:start + chunk]
        conversations, turn2s = [], []
        for row in batch:
            messages, turn2 = build_conversation(
                config, dataset, row, condition, first_turn_of(row), popqa_pool)
            conversations.append(messages)
            turn2s.append(turn2)

        texts = backend.generate(conversations, budget)

        for row, text, turn2 in zip(batch, texts, turn2s):
            letters = gold_letter_idx = None
            if dataset == "truthfulqa":
                letters, gold_letter_idx = lettered_choices(row)
            scored = _score_generation(dataset, row, text, first_turn_of(row),
                                       letters, gold_letter_idx)
            record = {"qid": int(row["qid"]), "condition": condition,
                      **scored, "second_turn": turn2, "response": text}
            if condition == "baseline":
                record["first_turn_text"] = text
            done[record["qid"]] = record
            _append(path, record)
        if progress is not None:
            progress.update(len(batch))
    return done


# ---------------------------------------------------------------------------
# Whole run
# ---------------------------------------------------------------------------

def check_resumable(config: RunConfig) -> None:
    """Refuse to mix two protocols in one output directory."""
    path = config.out_dir / "config.json"
    if not path.exists():
        return
    previous = json.loads(path.read_text())
    current = config.to_json()
    clashes = [k for k in PROTOCOL_KEYS if previous.get(k) != current.get(k)]
    if clashes:
        details = ", ".join(
            f"{k}: {previous.get(k)!r} on disk vs {current.get(k)!r} now"
            for k in clashes)
        raise SystemExit(
            f"{config.out_dir} holds a run with different settings ({details}).\n"
            "Pass --out with a new directory, or delete the old one to re-run.")


def plan(config: RunConfig) -> dict[str, int]:
    """Item counts per dataset, for the pre-run summary."""
    return {ds: len(load_rows(ds, config.split, config.limit))
            for ds in config.datasets}


def run(config: RunConfig, backend: Backend, progress=None) -> dict[str, dict]:
    """Run every requested condition on every requested dataset.

    Returns ``{dataset: {condition: {qid: record}}}``.
    """
    check_resumable(config)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "config.json").write_text(
        json.dumps(config.to_json(), indent=2) + "\n")

    ordered = (["baseline"] + [c for c in config.conditions if c != "baseline"])
    results: dict[str, dict] = {}

    for dataset in config.datasets:
        rows = load_rows(dataset, config.split, config.limit)
        pool = prompts.build_popqa_pool(rows) if dataset == "popqa" else None
        per_condition: dict[str, dict] = {}
        for condition in ordered:
            if progress is not None:
                progress.set_description(f"{dataset}/{condition}")
            per_condition[condition] = run_condition(
                config, backend, dataset, rows, condition,
                per_condition.get("baseline"), pool, progress)
        results[dataset] = per_condition
    return results
