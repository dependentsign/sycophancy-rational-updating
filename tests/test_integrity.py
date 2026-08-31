"""The shipped data must keep matching the manifest that describes it."""
from __future__ import annotations

import json

import pytest

from sru.integrity import check_files, check_rows, load_manifest, row_problems


def test_every_manifest_file_is_present_and_unchanged():
    checks = check_files()
    assert checks, "manifest listed no files"
    missing = [c.path.name for c in checks if c.missing]
    changed = [c.path.name for c in checks if not c.missing and not c.ok]
    assert not missing, f"manifest names files that are not shipped: {missing}"
    assert not changed, f"shipped data no longer matches the manifest: {changed}"


def test_row_counts_and_splits_agree_with_the_manifest():
    rows = check_rows()
    assert {r["dataset"] for r in rows} == {"truthfulqa", "popqa", "exfever", "aqua"}
    for row in rows:
        assert not row_problems(row), f"{row['dataset']}: {row_problems(row)}"


def test_calibration_and_test_never_share_an_item():
    for row in check_rows():
        assert row["overlap"] == 0, f"{row['dataset']} leaks items between splits"


def test_manifest_records_upstream_provenance():
    manifest = load_manifest()
    for name, spec in manifest["datasets"].items():
        assert spec["upstream"].startswith("http"), f"{name} has no upstream URL"
        assert spec["upstream_revision"], f"{name} has no pinned upstream revision"


@pytest.mark.parametrize("dataset", ["truthfulqa", "popqa", "exfever", "aqua"])
def test_every_row_carries_evidence(dataset):
    from sru.datasets import data_path

    empty = [json.loads(line)["qid"]
             for line in data_path(dataset).open(encoding="utf-8")
             if line.strip() and not (json.loads(line).get("evidence") or "").strip()]
    assert not empty, f"{dataset} rows without evidence: {empty[:5]}"


def test_popqa_evidence_actually_contains_the_gold_answer():
    """The repaired PopQA is the one where Evidence can do its job.

    An item whose evidence never mentions the gold answer cannot measure
    rational updating, so the release keeps that share near zero.
    """
    import re

    from sru.datasets import data_path

    def norm(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (text or "").lower())).strip()

    misses = 0
    total = 0
    for line in data_path("popqa").open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        total += 1
        evidence = norm(row.get("evidence"))
        answers = [row.get("obj"), *(row.get("possible_answers") or [])]
        if not any(len(norm(a)) >= 3 and re.search(rf"\b{re.escape(norm(a))}\b", evidence)
                   for a in answers if a):
            misses += 1
    assert misses / total < 0.01, (
        f"{misses}/{total} PopQA rows have evidence that never states the gold "
        "answer; the repaired snapshot should be under 1%")


def test_paper_backbones_are_pinned_to_a_revision():
    from sru.models import PAPER_MODELS, pinned_revision

    for model in PAPER_MODELS:
        revision = pinned_revision(model)
        assert revision and len(revision) == 40, f"{model} has no pinned revision"


def test_aliases_and_local_paths_resolve_to_the_same_backbone():
    from sru.models import canonical_id

    assert canonical_id("llama3.2-3b") == "meta-llama/Llama-3.2-3B-Instruct"
    assert canonical_id("Qwen3-8B") == "Qwen/Qwen3-8B"
    assert canonical_id("/data/models/google/gemma-3-4b-it") == "google/gemma-3-4b-it"
    assert canonical_id("mistralai/Mistral-7B-Instruct-v0.3") is None


def test_every_pinned_backbone_records_its_chat_template():
    """The template shapes every prompt, so a run can check it did not change."""
    from sru.models import PAPER_MODELS

    for model, spec in PAPER_MODELS.items():
        digest = spec["chat_template_sha256"]
        assert digest and len(digest) == 64, f"{model} has no chat template hash"


def test_a_revision_change_is_treated_as_a_protocol_change():
    from sru.pipeline import PROTOCOL_KEYS

    assert "revision" in PROTOCOL_KEYS, (
        "resuming across a revision change would mix two model versions")
