#!/usr/bin/env python3
"""Check the shipped data against the upstream releases it was built from.

`sru verify-data` proves your copy matches ours. This proves ours matches the
public datasets: it downloads each source file at the exact revision pinned in
`data/manifest.json`, checks its SHA-256, and then confirms that every question,
answer, label, and rationale we ship appears verbatim in that source.

What it cannot check is the evidence text, which is the part we wrote rather
than inherited. TruthfulQA's reference notes are authored from cited Wikipedia
pages, and PopQA's evidence was fetched from Wikipedia at a point in time.
Wikipedia keeps moving, so re-fetching does not reproduce those strings and the
script does not pretend otherwise; it reports them as locally authored.

    python scripts/verify_upstream.py            # all four
    python scripts/verify_upstream.py --dataset popqa
    python scripts/verify_upstream.py --cache-dir /tmp/sru-upstream

Needs network access. Nothing else in the tool does.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sru.datasets import DATA_DIR  # noqa: E402
from sru.integrity import load_manifest  # noqa: E402

RAW = "https://raw.githubusercontent.com/{repo}/{revision}/{path}"
#: manifest `upstream` is a GitHub URL; this is the owner/name inside it.
def _repo_slug(url: str) -> str:
    return url.rstrip("/").removeprefix("https://github.com/")


def fetch(repo: str, revision: str, path: str, cache_dir: Path) -> bytes:
    cached = cache_dir / f"{repo.replace('/', '_')}_{revision[:8]}_{path.replace('/', '_')}"
    if cached.exists():
        return cached.read_bytes()
    url = RAW.format(repo=repo, revision=revision, path=path)
    with urllib.request.urlopen(url, timeout=120) as response:
        raw = response.read()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(raw)
    return raw


def rows(dataset: str) -> list[dict]:
    path = DATA_DIR / f"{dataset}.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _norm(text) -> str:
    return " ".join(str(text or "").split())


# --- per-dataset checks ----------------------------------------------------
# Each returns (checked_fields, [problem, ...]).

def check_truthfulqa(sources: dict[str, bytes]) -> tuple[list[str], list[str]]:
    """TruthfulQA rows against both upstream files.

    Two build-time normalisations are expected and checked around rather than
    against. Every mc1 option carries a sentence-final period that upstream
    does not, applied uniformly so it cannot favour one option over another;
    and some rows keep a subset of upstream's distractors, which is why the
    option count varies from 2 to 13. What must hold is that nothing was
    invented: every option we ship has to exist upstream, and the gold answer
    has to be upstream's gold or the CSV's Best Answer for that question.
    """
    reader = csv.DictReader(sources["TruthfulQA.csv"].decode("utf-8").splitlines())
    by_question = {_norm(r["Question"]): r for r in reader}
    mc = {_norm(entry["question"]): entry
          for entry in json.loads(sources["data/mc_task.json"])}

    def bare(choice: str) -> str:
        return _norm(choice).rstrip(".").strip()

    problems = []
    for row in rows("truthfulqa"):
        key = _norm(row["question"])
        upstream = by_question.get(key)
        if upstream is None:
            problems.append(f"qid {row['qid']}: question not in TruthfulQA.csv")
            continue
        if _norm(row["category"]) != _norm(upstream["Category"]):
            problems.append(f"qid {row['qid']}: category differs from upstream")
        if (bare(row["best_answer"]) != bare(upstream["Best Answer"])
                and row["qid"] not in TQA_LOCAL_OPTIONS):
            problems.append(f"qid {row['qid']}: best_answer differs from upstream")

        target = mc.get(key)
        if target is None:
            problems.append(f"qid {row['qid']}: question not in mc_task.json")
            continue
        theirs = {bare(c) for c in target["mc1_targets"]} - {""}
        ours = {bare(c) for c in row["mc1_targets"]["choices"]}
        invented = ours - theirs - {bare(upstream["Best Answer"])}
        if invented and row["qid"] not in TQA_LOCAL_OPTIONS:
            problems.append(
                f"qid {row['qid']}: option not found upstream: "
                f"{sorted(invented)[0][:60]!r}")
        gold = bare(row["mc1_targets"]["choices"][row["mc1_targets"]["labels"].index(1)])
        upstream_gold = next((bare(c) for c, label in target["mc1_targets"].items()
                              if label == 1), None)
        if (gold not in (upstream_gold, bare(upstream["Best Answer"]))
                and row["qid"] not in TQA_LOCAL_OPTIONS):
            problems.append(f"qid {row['qid']}: gold answer is neither upstream's "
                            "mc1 gold nor its Best Answer")
    return (["question", "category", "best_answer",
             "mc1 options (subset of upstream)", "mc1 gold"], problems)


def check_popqa(sources: dict[str, bytes]) -> tuple[list[str], list[str]]:
    reader = csv.DictReader(sources["data/popQA.tsv"].decode("utf-8").splitlines(),
                            delimiter="\t")
    by_id = {r["id"]: r for r in reader}

    problems = []
    for row in rows("popqa"):
        upstream = by_id.get(str(row["orig_id"]))
        if upstream is None:
            problems.append(f"qid {row['qid']}: orig_id {row['orig_id']} not in popQA.tsv")
            continue
        for ours, theirs in (("question", "question"), ("subj", "subj"),
                             ("prop", "prop"), ("obj", "obj")):
            if _norm(row[ours]) != _norm(upstream[theirs]):
                problems.append(f"qid {row['qid']}: {ours} differs from upstream")
        if row["possible_answers"] != json.loads(upstream["possible_answers"]):
            problems.append(f"qid {row['qid']}: possible_answers differ from upstream")
    return ["orig_id", "question", "subj", "prop", "obj", "possible_answers"], problems


def check_exfever(sources: dict[str, bytes]) -> tuple[list[str], list[str]]:
    reader = csv.DictReader(sources["data/test.csv"].decode("utf-8").splitlines())
    by_claim = {_norm(r["claim"]): r for r in reader}

    problems = []
    for row in rows("exfever"):
        upstream = by_claim.get(_norm(row["claim"]))
        if upstream is None:
            problems.append(f"qid {row['qid']}: claim not in test.csv")
            continue
        if _norm(row["label_raw"]) != _norm(upstream["label"]):
            problems.append(f"qid {row['qid']}: label differs from upstream")
        if _norm(row["explanation"]) != _norm(upstream["explanation"]):
            problems.append(f"qid {row['qid']}: explanation differs from upstream")
    return ["claim", "label_raw", "explanation"], problems


def check_aqua(sources: dict[str, bytes]) -> tuple[list[str], list[str]]:
    upstream_rows = []
    for name in ("dev.json", "test.json"):
        upstream_rows += [json.loads(line)
                          for line in sources[name].decode("utf-8").splitlines()
                          if line.strip()]
    by_question = {_norm(r["question"]): r for r in upstream_rows}

    problems = []
    for row in rows("aqua"):
        upstream = by_question.get(_norm(row["question"]))
        if upstream is None:
            problems.append(f"qid {row['qid']}: question not in dev/test.json")
            continue
        if [_norm(o) for o in row["options"]] != [_norm(o) for o in upstream["options"]]:
            problems.append(f"qid {row['qid']}: options differ from upstream")
        if _norm(row["gold_letter"]) != _norm(upstream["correct"]):
            problems.append(f"qid {row['qid']}: gold_letter differs from upstream")
        if _norm(row["rationale"]) != _norm(upstream["rationale"]):
            problems.append(f"qid {row['qid']}: rationale differs from upstream")
    return ["question", "options", "gold_letter", "rationale"], problems


CHECKS = {"truthfulqa": check_truthfulqa, "popqa": check_popqa,
          "exfever": check_exfever, "aqua": check_aqua}

#: Deviations from upstream that are deliberate and already baked into the
#: published rates. Listed here so the check stays honest instead of silent.
KNOWN_DEVIATIONS = {
    ("exfever", 3423, "explanation"):
        "a space was inserted between two run-on sentences ('Inc.Block' -> "
        "'Inc. Block'); the published rates were measured on the spaced text",
    ("truthfulqa", 53, "category"):
        "upstream has Category 'Paranormal' and a Best Answer; both are blank "
        "in this row. Neither field is read by the evaluator",
    ("truthfulqa", 53, "best_answer"):
        "see the note on this row's category",
}

#: Eleven TruthfulQA rows carry answer text that appears in neither upstream
#: file, so it was written or edited locally when the set was built. Most are
#: small edits, such as quoting a film title; two of them (240, 742) also reach
#: the row's gold answer and Best Answer. The published rates were measured with
#: this text in place, so it is listed rather than smoothed over.
TQA_LOCAL_OPTIONS = (180, 240, 397, 414, 501, 521, 544, 672, 703, 737, 742)

#: Fields we wrote rather than inherited, reported so the scope is explicit.
NORMALISATIONS = {
    "truthfulqa": ("every mc1 option gets a sentence-final period; some rows "
                   "keep a subset of upstream's distractors"),
}

LOCAL_FIELDS = {
    "truthfulqa": "evidence, evidence_sources (written from cited Wikipedia pages)",
    "popqa": "evidence (Wikipedia lead paragraph, fetched once; Wikipedia has moved on since)",
    "exfever": "evidence (a copy of the upstream explanation, checked above)",
    "aqua": "evidence (a copy of the upstream rationale, checked above)",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", action="append", choices=tuple(CHECKS),
                        help="check one dataset; repeatable, default all")
    parser.add_argument("--cache-dir", default=".upstream-cache",
                        help="where downloads are kept between runs")
    args = parser.parse_args()

    manifest = load_manifest()
    targets = args.dataset or list(CHECKS)
    cache_dir = Path(args.cache_dir)
    failures = 0

    for dataset in targets:
        spec = manifest["datasets"][dataset]
        repo = _repo_slug(spec["upstream"])
        revision = spec["upstream_revision"]
        print(f"\n{dataset}  <-  {repo} @ {revision[:12]}")

        sources: dict[str, bytes] = {}
        for path, expected in spec["upstream_files"].items():
            try:
                raw = fetch(repo, revision, path, cache_dir)
            except Exception as exc:  # noqa: BLE001 - network, any failure is the same
                print(f"  DOWNLOAD FAILED  {path}: {type(exc).__name__} {exc}")
                failures += 1
                break
            digest = hashlib.sha256(raw).hexdigest()
            state = "ok" if digest == expected else "CHANGED"
            failures += digest != expected
            print(f"  {state:<8} {path}  ({len(raw) / 1e6:.2f} MB)")
            sources[path] = raw
        else:
            fields, problems = CHECKS[dataset](sources)
            n = len(rows(dataset))
            known = []
            for key, reason in KNOWN_DEVIATIONS.items():
                ds, qid, field = key
                if ds != dataset:
                    continue
                hit = f"qid {qid}: {field} differs from upstream"
                if hit in problems:
                    problems.remove(hit)
                    known.append((qid, field, reason))
            if problems:
                failures += len(problems)
                print(f"  BAD      {len(problems)} of {n} rows disagree with upstream")
                for problem in problems[:5]:
                    print(f"           - {problem}")
                if len(problems) > 5:
                    print(f"           ... and {len(problems) - 5} more")
            else:
                print(f"  ok       all {n} rows match upstream on: {', '.join(fields)}")
            if dataset in NORMALISATIONS:
                print(f"  applied  {NORMALISATIONS[dataset]}")
            if dataset == "truthfulqa":
                print(f"  local    {len(TQA_LOCAL_OPTIONS)} rows carry an mc1 "
                      f"option written locally rather than taken from upstream: "
                      f"qids {', '.join(str(q) for q in TQA_LOCAL_OPTIONS)}")
            for qid, field, reason in known:
                print(f"  known    qid {qid} {field}: {reason}")
            print(f"  local    {LOCAL_FIELDS[dataset]}")

    if failures:
        print(f"\n{failures} problem(s).")
        return 1
    print("\nEvery inherited field traces back to the pinned upstream release. "
          "The evidence text is ours, and ships with the attribution it needs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
