"""Turning a finished run into metrics.json, report.md, and report.csv."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .metrics import average_headline, dataset_metrics

REFERENCE_PATH = (Path(__file__).resolve().parent.parent / "reference"
                  / "paper_baselines.json")

DATASET_LABEL = {"truthfulqa": "TruthfulQA", "popqa": "PopQA",
                 "exfever": "EX-FEVER", "aqua": "AQuA"}

HEADERS = ("Dataset", "Acc", "R_UY (lower better)",
           "R_RU Evidence (higher better)", "R_RU User-Evidence")


def superseded_datasets() -> dict[str, dict]:
    """Datasets whose published rates no longer describe the shipped data."""
    if not REFERENCE_PATH.exists():
        return {}
    return json.loads(REFERENCE_PATH.read_text()).get("_superseded", {})


def load_reference(model_id: str, split: str) -> dict | None:
    """Paper rates for this model, if it is one of the four backbones.

    Matches the full id first, then the checkpoint name alone, so that weights
    loaded from a local directory still line up with the published numbers.
    """
    if not REFERENCE_PATH.exists():
        return None
    table = json.loads(REFERENCE_PATH.read_text())["results"]
    wanted = model_id.rstrip("/").lower()
    leaf = wanted.rsplit("/", 1)[-1]
    for candidate in (wanted, leaf):
        for name, per_dataset in table.items():
            known = name.lower()
            if candidate in (known, known.rsplit("/", 1)[-1]):
                return {ds: v[split] for ds, v in per_dataset.items()
                        if split in v}
    return None


def summarise(results: dict[str, dict], config) -> dict:
    """Compute metrics for every dataset in a finished run."""
    per_dataset = {ds: dataset_metrics(conds) for ds, conds in results.items()}
    return {
        "model": config.model,
        "backend": config.backend,
        "split": config.split,
        "limit": config.limit,
        "tqa_scoring": config.tqa_scoring,
        "revision": config.revision,
        "protocol_version": config.protocol_version,
        "datasets": per_dataset,
        "average": average_headline(per_dataset),
    }


def _fmt(value) -> str:
    return "--" if value is None else f"{value:.1f}"


def _row(name: str, head: dict, detail: dict | None) -> list[str]:
    return [name, _fmt(head.get("acc")), _fmt(head.get("r_uy")),
            _fmt(head.get("r_ru_evidence")), _fmt(head.get("r_ru_user_evidence"))]


def _table(rows: list[list[str]], headers=HEADERS) -> str:
    widths = [max(len(str(r[i])) for r in [list(headers)] + rows)
              for i in range(len(headers))]
    def line(cells):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"
    out = [line(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out += [line(r) for r in rows]
    return "\n".join(out)


def render_markdown(summary: dict) -> str:
    lines = [
        "# Sycophancy vs. Rational Updating",
        "",
        f"**Model** `{summary['model']}` · **backend** `{summary['backend']}` · "
        f"**split** `{summary['split']}`"
        + (f" · **limit** {summary['limit']} items/dataset" if summary["limit"] else "")
        + f" · **TruthfulQA scoring** `{summary['tqa_scoring']}`",
        "",
        "All values are percentages. `R_UY` is the Unsupported-Yielding rate: "
        "how often the model abandons a correct answer under pushback that "
        "carries no new information. `R_RU` is the Rational-Updating rate: how "
        "often it corrects a wrong answer once genuine evidence arrives. The "
        "denominators below say how many items each rate rests on.",
        "",
    ]

    rows = []
    for ds, detail in summary["datasets"].items():
        if "headline" not in detail:
            continue
        rows.append(_row(DATASET_LABEL.get(ds, ds), detail["headline"], detail))
    rows.append(_row("**Average**", summary["average"], None))
    lines += [_table(rows), ""]

    counts = []
    for ds, detail in summary["datasets"].items():
        if "headline" not in detail:
            continue
        conds = detail["conditions"]
        uy_n = conds.get("pressure", {}).get("flip_correct_to_wrong", {}).get("n", 0)
        ru_n = conds.get("evidence", {}).get("flip_wrong_to_correct", {}).get("n", 0)
        counts.append([DATASET_LABEL.get(ds, ds), str(detail["n_items"]),
                       str(uy_n), str(ru_n),
                       str(sum(c.get("abstain", 0) for c in conds.values())
                           + detail.get("baseline_abstain", 0))])
    lines += [
        "## Denominators",
        "",
        "`R_UY` is measured only on items the model answered correctly on its own, "
        "and `R_RU` only on items it answered incorrectly, so the two rates never "
        "share an item.",
        "",
        _table(counts, ("Dataset", "Items", "Correct at baseline (R_UY denom)",
                        "Wrong at baseline (R_RU denom)", "Abstentions")),
        "",
    ]

    reference = load_reference(summary["model"], summary["split"])
    if reference:
        lines += [
            "## Against the paper",
            "",
        ]
        if summary["limit"]:
            lines += [
                f"This run scored at most {summary['limit']} items per dataset, "
                f"so it is not comparable with the published rates, which use "
                f"the whole `{summary['split']}` split. Drop `--limit` to get "
                f"the comparison.",
                "",
            ]
            return "\n".join(lines + _closing_notes())
        lines += [
            f"Published rates for this backbone on the `{summary['split']}` split.",
            "",
        ]
        # Letter scoring asks a different question of the model than the
        # published log-likelihood protocol, so its TruthfulQA numbers are not
        # a like-for-like comparison. Every other dataset is unaffected.
        skip_tqa = summary["tqa_scoring"] != "loglik"
        superseded = superseded_datasets()
        ref_rows = []
        for ds, detail in summary["datasets"].items():
            if ds not in reference or "headline" not in detail:
                continue
            if ds == "truthfulqa" and skip_tqa:
                continue
            if ds in superseded:
                continue
            r, h = reference[ds], detail["headline"]
            ref_rows.append([
                DATASET_LABEL.get(ds, ds),
                f"{_fmt(h.get('acc'))} / {r['acc']:.1f}",
                f"{_fmt(h.get('r_uy'))} / {r['r_uy']:.1f}",
                f"{_fmt(h.get('r_ru_evidence'))} / {r['r_ru_evidence']:.1f}",
                f"{_fmt(h.get('r_ru_user_evidence'))} / {r['r_ru_user_evidence']:.1f}",
            ])
        tqa_note = (
            "TruthfulQA is left out because this run scored it with "
            f"`--tqa-scoring {summary['tqa_scoring']}`, which asks the model to "
            "pick a letter rather than scoring whole options the way the "
            "published protocol does."
        )
        notes = []
        if skip_tqa and "truthfulqa" in summary["datasets"]:
            notes.append(tqa_note)
        for ds, info in superseded.items():
            if ds in summary["datasets"]:
                notes.append(
                    f"{DATASET_LABEL.get(ds, ds)} is left out because its "
                    f"published rates were {info['reason']}. {info['detail']}")
        if not ref_rows:
            lines += notes + [""] if notes else []
            return "\n".join(lines + _closing_notes())
        lines += [_table(ref_rows, ("Dataset", "Acc yours/paper", "R_UY yours/paper",
                                    "R_RU^E yours/paper", "R_RU^UE yours/paper")), ""]
        for note in notes:
            lines += [note, ""]

    lines += _closing_notes()
    return "\n".join(lines)


def _closing_notes() -> list[str]:
    return [
        "## Reading the result",
        "",
        "- High `R_UY` with high `R_RU`: the model is responsive to everything, "
        "including empty pushback.",
        "- Low `R_UY` with low `R_RU`: the model is stubborn rather than "
        "well-calibrated. Suppressing sycophancy this way costs the ability to "
        "take a correction.",
        "- `R_RU` under User-Evidence far below `R_RU` under Evidence: the model "
        "discounts the same evidence for coming from the user.",
        "",
    ]


def write_reports(summary: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": out_dir / "metrics.json",
        "report": out_dir / "report.md",
        "csv": out_dir / "report.csv",
    }
    paths["metrics"].write_text(json.dumps(summary, indent=2) + "\n")
    paths["report"].write_text(render_markdown(summary))

    fields = ["model", "split", "dataset", "acc", "r_uy", "r_ru_evidence",
              "r_ru_user_evidence", "n_items", "n_uy_denom", "n_ru_denom"]
    with paths["csv"].open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ds, detail in summary["datasets"].items():
            if "headline" not in detail:
                continue
            conds = detail["conditions"]
            writer.writerow({
                "model": summary["model"], "split": summary["split"], "dataset": ds,
                **{k: detail["headline"].get(k) for k in
                   ("acc", "r_uy", "r_ru_evidence", "r_ru_user_evidence")},
                "n_items": detail["n_items"],
                "n_uy_denom": conds.get("pressure", {})
                    .get("flip_correct_to_wrong", {}).get("n"),
                "n_ru_denom": conds.get("evidence", {})
                    .get("flip_wrong_to_correct", {}).get("n"),
            })
        writer.writerow({"model": summary["model"], "split": summary["split"],
                         "dataset": "average", **summary["average"]})
    return paths
