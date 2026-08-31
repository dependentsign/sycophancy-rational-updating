"""Dataset registry, loading, and split filtering.

Each shipped file is the calibration + held-out test subset used in the paper.
Every row carries a ``qid``, a task field, a gold answer, and an ``evidence``
string that supports the gold answer. That evidence is what the Evidence and
User-Evidence conditions hand back to the model.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

def _find_data_dir() -> Path:
    """Where the evaluation data lives.

    Three ways the tool gets used, in priority order: SRU_DATA_DIR for anyone
    keeping the data elsewhere, the repository's own `data/` for a clone or an
    editable install, and the installed `sru_data` package for a plain
    `pip install`, where the wheel carries the data with it.
    """
    override = os.environ.get("SRU_DATA_DIR")
    if override:
        return Path(override)
    beside_repo = Path(__file__).resolve().parent.parent / "data"
    if (beside_repo / "splits").is_dir():
        return beside_repo
    try:
        import sru_data
        packaged = Path(sru_data.__file__).resolve().parent
        if (packaged / "splits").is_dir():
            return packaged
    except Exception:
        pass
    return beside_repo


DATA_DIR = _find_data_dir()


def require_data_dir() -> Path:
    if not (DATA_DIR / "splits").is_dir():
        raise SystemExit(
            f"no evaluation data at {DATA_DIR}.\n"
            "Run from a clone of the repository, or point SRU_DATA_DIR at its "
            "data/ directory.")
    return DATA_DIR


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task: str
    #: How the second-turn answer is read back out of the model.
    scoring: str          # "choice" | "freeform"
    #: Generation budget for the free-form path.
    max_new_tokens: int
    evidence_source: str
    citation: str


DATASETS: dict[str, DatasetSpec] = {
    "truthfulqa": DatasetSpec(
        "truthfulqa", "multiple-choice (MC1), 2-13 options", "choice", 256,
        "quote-backed reference notes written from the cited Wikipedia pages",
        "Lin et al., 2022"),
    "popqa": DatasetSpec(
        "popqa", "open-ended long-tail entity QA", "freeform", 128,
        "lead paragraph of the subject entity's Wikipedia page",
        "Mallen et al., 2023"),
    "exfever": DatasetSpec(
        "exfever", "fact verification (True/False)", "freeform", 256,
        "the dataset's gold supporting explanation",
        "Ma et al., 2024"),
    "aqua": DatasetSpec(
        "aqua", "multi-step numerical reasoning (5 options)", "freeform", 1024,
        "the annotator-written rationale",
        "Ling et al., 2017"),
}

ALL_DATASETS = tuple(DATASETS)
SPLITS = ("test", "cal", "both")


def data_path(dataset: str) -> Path:
    """Path to the shipped jsonl for a dataset."""
    return DATA_DIR / f"{dataset}.jsonl"


def load_rows(dataset: str, split: str = "test",
              limit: int | None = None) -> list[dict]:
    """Load one dataset, restricted to a split and optionally truncated.

    ``limit`` keeps the first N rows in qid order, so a truncated run is a
    prefix of the full run and two truncated runs are comparable.
    """
    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; choose from {ALL_DATASETS}")
    if split not in SPLITS:
        raise KeyError(f"unknown split {split!r}; choose from {SPLITS}")

    require_data_dir()
    path = data_path(dataset)
    if not path.exists():
        raise SystemExit(f"missing data file {path}")
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

    keep = split_qids(dataset, split)
    rows = [r for r in rows if int(r["qid"]) in keep]
    rows.sort(key=lambda r: int(r["qid"]))
    if limit is not None:
        rows = rows[:limit]
    return rows


def split_qids(dataset: str, split: str) -> set[int]:
    spec = json.loads((DATA_DIR / "splits" / f"{dataset}.json").read_text())
    if split == "both":
        return set(spec["cal"]) | set(spec["test"])
    return set(spec[split])


# ---------------------------------------------------------------------------
# TruthfulQA option lettering (letter scoring only)
# ---------------------------------------------------------------------------

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def lettered_choices(row: dict) -> tuple[list[tuple[str, str]], int]:
    """Shuffle a TruthfulQA row's options and letter them.

    TruthfulQA stores the correct option first in every row, so the options
    must be permuted before they are shown to the model. The permutation is
    seeded by qid, so it is identical for every model and every run.

    Returns ``([(letter, text), ...], gold_letter_index)``.
    """
    choices = row["mc1_targets"]["choices"]
    gold = row["mc1_targets"]["labels"].index(1)
    order = list(range(len(choices)))
    random.Random(1000 + int(row["qid"])).shuffle(order)
    lettered = [(LETTERS[i], choices[j]) for i, j in enumerate(order)]
    return lettered, order.index(gold)
