"""Verify the shipped evaluation data against its manifest.

Every data and split file has a SHA-256 recorded in ``data/manifest.json``,
together with the upstream repository and revision it was built from. Checking
those hashes is what makes a number comparable across machines: two people who
both pass this check scored the same bytes, the same qids, and the same
calibration/test assignment.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .datasets import DATA_DIR, require_data_dir

MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class FileCheck:
    dataset: str
    role: str          # "data" | "split" | an additional_files key
    path: Path
    expected: str
    actual: str | None  # None when the file is missing

    @property
    def ok(self) -> bool:
        return self.actual == self.expected

    @property
    def missing(self) -> bool:
        return self.actual is None


def manifest_path(data_dir: Path | None = None) -> Path:
    return (data_dir or DATA_DIR) / MANIFEST_NAME


def load_manifest(data_dir: Path | None = None) -> dict:
    path = manifest_path(data_dir)
    if not path.exists():
        raise SystemExit(
            f"no manifest at {path}. Run from a clone of the repository, or "
            "point SRU_DATA_DIR at its data/ directory.")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_files(datasets: tuple[str, ...] | None = None,
                data_dir: Path | None = None) -> list[FileCheck]:
    """Hash every file the manifest names and compare with the recorded value."""
    root = data_dir or require_data_dir()
    manifest = load_manifest(root)
    checks: list[FileCheck] = []
    for name, spec in manifest["datasets"].items():
        if datasets and name not in datasets:
            continue
        entries = [("data", spec["data_file"], spec["data_sha256"]),
                   ("split", spec["split_file"], spec["split_sha256"])]
        for role, extra in (spec.get("additional_files") or {}).items():
            entries.append((role, extra["file"], extra["sha256"]))
        for role, relative, expected in entries:
            path = root / relative
            checks.append(FileCheck(
                dataset=name, role=role, path=path, expected=expected,
                actual=sha256(path) if path.exists() else None))
    return checks


def check_rows(datasets: tuple[str, ...] | None = None,
               data_dir: Path | None = None) -> list[dict]:
    """Confirm row counts and split sizes agree with the manifest."""
    root = data_dir or require_data_dir()
    manifest = load_manifest(root)
    out: list[dict] = []
    for name, spec in manifest["datasets"].items():
        if datasets and name not in datasets:
            continue
        data_file = root / spec["data_file"]
        split_file = root / spec["split_file"]
        if not data_file.exists() or not split_file.exists():
            continue
        qids = [int(json.loads(line)["qid"])
                for line in data_file.open(encoding="utf-8") if line.strip()]
        split = json.loads(split_file.read_text(encoding="utf-8"))
        cal, test = set(split["cal"]), set(split["test"])
        out.append({
            "dataset": name,
            "rows": len(qids),
            "rows_expected": spec["rows"],
            "unique_qids": len(set(qids)),
            "cal": len(cal),
            "cal_expected": spec["calibration"],
            "test": len(test),
            "test_expected": spec["test"],
            "overlap": len(cal & test),
            "unknown_qids": len((cal | test) - set(qids)),
        })
    return out


def row_problems(row: dict) -> list[str]:
    """Human-readable problems with one check_rows entry, empty when clean."""
    problems = []
    if row["rows"] != row["rows_expected"]:
        problems.append(f"{row['rows']} rows, manifest says {row['rows_expected']}")
    if row["unique_qids"] != row["rows"]:
        problems.append(f"{row['rows'] - row['unique_qids']} duplicate qids")
    if row["cal"] != row["cal_expected"]:
        problems.append(f"cal {row['cal']}, manifest says {row['cal_expected']}")
    if row["test"] != row["test_expected"]:
        problems.append(f"test {row['test']}, manifest says {row['test_expected']}")
    if row["overlap"]:
        problems.append(f"{row['overlap']} qids in both cal and test")
    if row["unknown_qids"]:
        problems.append(f"{row['unknown_qids']} split qids absent from the data")
    return problems
