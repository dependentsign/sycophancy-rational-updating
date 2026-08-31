"""Collecting finished runs into one comparison across models and datasets.

A single run answers "how does this model behave". Put several side by side and
the question becomes "which model handles pushback and evidence better", which
is what the diagnostic was built to ask. This module reads the `metrics.json`
of any number of run directories and lays them out together.
"""
from __future__ import annotations

import json
from pathlib import Path

from .datasets import ALL_DATASETS

HEADLINE_KEYS = ("acc", "r_uy", "r_ru_evidence", "r_ru_user_evidence")


def _short(model: str) -> str:
    """A readable name for a model, so a table of local paths stays legible."""
    from .models import canonical_id

    return canonical_id(model) or model.rstrip("/").rsplit("/", 1)[-1]


def find_runs(paths: list[Path]) -> list[Path]:
    """Every run directory at or under the given paths, in a stable order.

    A path may be a run directory itself, or a parent holding several, so that
    both `sru compare runs/a runs/b` and `sru compare runs` work.
    """
    found: set[Path] = set()
    for path in paths:
        if (path / "metrics.json").is_file():
            found.add(path)
            continue
        found.update(p.parent for p in sorted(path.rglob("metrics.json")))
    return sorted(found)


def load_run(run_dir: Path) -> dict | None:
    """Read one run, or None if its metrics are missing or unreadable."""
    try:
        summary = json.loads((run_dir / "metrics.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if "datasets" not in summary:
        return None
    summary["run_dir"] = str(run_dir)
    summary["started_at"] = _started_at(run_dir)
    return summary


def _started_at(run_dir: Path) -> str:
    """When the run began, for choosing between two runs of the same dataset."""
    try:
        stamp = json.loads((run_dir / "config.json").read_text()).get("started_at")
        if stamp:
            return str(stamp)
    except (OSError, json.JSONDecodeError):
        pass
    try:  # no config.json: fall back to the metrics file's own timestamp
        return str((run_dir / "metrics.json").stat().st_mtime)
    except OSError:
        return ""


def selectivity(headline: dict) -> float | None:
    """`R_RU` under Evidence minus `R_UY`.

    A reading aid, not a metric from the paper: it goes up both by resisting
    empty pushback and by taking real evidence, which is the pair of behaviors
    the diagnostic separates. It says nothing about which of the two moved.
    """
    r_ru = headline.get("r_ru_evidence")
    r_uy = headline.get("r_uy")
    if r_ru is None or r_uy is None:
        return None
    return round(r_ru - r_uy, 2)


def collect(run_dirs: list[Path]) -> dict:
    """Group runs by model, merging per-dataset results across run directories.

    Evaluating one model over several runs (one dataset at a time, say) is the
    normal way to use a shared GPU, so those runs are stitched back together
    here rather than reported as unrelated models.
    """
    by_model: dict[str, dict] = {}
    skipped: list[str] = []
    conflicts: list[str] = []

    loaded = []
    for run_dir in run_dirs:
        summary = load_run(run_dir)
        if summary is None:
            skipped.append(str(run_dir))
        else:
            loaded.append(summary)
    # Oldest first, so a later run of the same dataset replaces an earlier one.
    loaded.sort(key=lambda s: s["started_at"])

    for summary in loaded:
        model = summary["model"]
        entry = by_model.setdefault(model, {
            "model": model,
            "backend": summary.get("backend"),
            "split": summary.get("split"),
            "limit": summary.get("limit"),
            "tqa_scoring": summary.get("tqa_scoring"),
            "datasets": {},
            "runs": [],
        })
        entry["runs"].append(summary["run_dir"])
        for key in ("split", "limit", "tqa_scoring"):
            if entry[key] != summary.get(key):
                conflicts.append(
                    f"{_short(model)}: {key} differs between runs "
                    f"({entry[key]!r} and {summary.get(key)!r})")
                entry["mixed_scope"] = True
        for name, detail in summary["datasets"].items():
            if "headline" not in detail:
                continue
            if name in entry["datasets"]:
                conflicts.append(
                    f"{_short(model)}: {name} was run more than once, showing "
                    "the most recent")
            entry["datasets"][name] = {
                **{k: detail["headline"].get(k) for k in HEADLINE_KEYS},
                "n_items": detail.get("n_items"),
            }

    for entry in by_model.values():
        entry["average"] = _mean_over_datasets(entry["datasets"])
        entry["selectivity"] = selectivity(entry["average"])

    order = sorted(by_model.values(),
                   key=lambda e: (e["selectivity"] is None, -(e["selectivity"] or 0)))
    return {"models": order, "skipped": skipped, "conflicts": conflicts}


def _mean_over_datasets(datasets: dict) -> dict:
    """Average each rate over the datasets that reported it."""
    out: dict[str, float | None] = {}
    for key in HEADLINE_KEYS:
        values = [d[key] for d in datasets.values() if d.get(key) is not None]
        out[key] = round(sum(values) / len(values), 2) if values else None
    return out


def _fmt(value) -> str:
    return "--" if value is None else f"{value:.1f}"


def _table(rows: list[list[str]], header: tuple[str, ...]) -> str:
    widths = [max(len(str(r[i])) for r in [list(header)] + rows)
              for i in range(len(header))]
    line = lambda cells: ("| " + " | ".join(
        str(c).ljust(w) for c, w in zip(cells, widths)) + " |")
    return "\n".join([line(header),
                      "|" + "|".join("-" * (w + 2) for w in widths) + "|",
                      *(line(r) for r in rows)])


def render_markdown(collected: dict) -> str:
    models = collected["models"]
    if not models:
        return "No finished runs found.\n"

    scopes = {(m["split"], m["limit"], m["tqa_scoring"]) for m in models}
    mixed = len(scopes) > 1 or any(m.get("mixed_scope") for m in models)
    lines = ["# Model comparison", ""]
    if mixed:
        lines += ["These runs do not all share a split, an item limit, and a "
                  "scoring mode, so the rows below are not measuring the same "
                  "thing. What differs is listed at the end.", ""]
    else:
        split, limit, scoring = next(iter(scopes))
        scope = f"**split** `{split}`"
        if limit:
            scope += f" · **limit** {limit} items/dataset"
        lines += [scope + f" · **TruthfulQA scoring** `{scoring}`", ""]

    lines += [
        "All values are percentages, averaged over the datasets each model was "
        "run on. `R_UY` is lower-is-better and `R_RU` is higher-is-better, so "
        "the last column subtracts one from the other as a reading aid; it is "
        "not a metric from the paper, and a model can raise it either by "
        "resisting pushback or by taking evidence.",
        "",
    ]
    rows = [[_short(m["model"]),
             ", ".join(sorted(m["datasets"])) or "--",
             _fmt(m["average"]["acc"]),
             _fmt(m["average"]["r_uy"]),
             _fmt(m["average"]["r_ru_evidence"]),
             _fmt(m["average"]["r_ru_user_evidence"]),
             _fmt(m["selectivity"])]
            for m in models]
    lines += [_table(rows, ("Model", "Datasets", "Acc", "R_UY", "R_RU^E",
                            "R_RU^UE", "R_RU^E - R_UY")), ""]

    present = [d for d in ALL_DATASETS
               if any(d in m["datasets"] for m in models)]
    for dataset in present:
        lines += [f"## {dataset}", ""]
        rows = []
        for m in models:
            detail = m["datasets"].get(dataset)
            if detail is None:
                continue
            rows.append([_short(m["model"]), _fmt(detail["acc"]),
                         _fmt(detail["r_uy"]),
                         _fmt(detail["r_ru_evidence"]),
                         _fmt(detail["r_ru_user_evidence"]),
                         str(detail["n_items"])])
        lines += [_table(rows, ("Model", "Acc", "R_UY", "R_RU^E", "R_RU^UE",
                                "Items")), ""]

    if collected["conflicts"]:
        lines += ["## Worth knowing", ""]
        lines += [f"- {c}" for c in dict.fromkeys(collected["conflicts"])]
        lines += [""]
    if collected["skipped"]:
        lines += [f"Skipped {len(collected['skipped'])} directory(ies) without "
                  "readable metrics.", ""]
    return "\n".join(lines)


def write_csv(collected: dict, path: Path) -> None:
    import csv

    fields = ["model", "dataset", "acc", "r_uy", "r_ru_evidence",
              "r_ru_user_evidence", "n_items"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in collected["models"]:
            for dataset, detail in sorted(model["datasets"].items()):
                writer.writerow({"model": _short(model["model"]), "dataset": dataset,
                                 **{k: detail.get(k) for k in HEADLINE_KEYS},
                                 "n_items": detail.get("n_items")})
            writer.writerow({"model": _short(model["model"]), "dataset": "average",
                             **model["average"], "n_items": None})
