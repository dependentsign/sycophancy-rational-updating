"""Command line interface.

    sru run --model meta-llama/Llama-3.1-8B-Instruct
    sru run --model gpt-4o-mini --limit 50
    sru preview --dataset truthfulqa
    sru report runs/gpt-4o-mini_test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import prompts
from .backends import BACKENDS, build_backend, resolve_backend
from .datasets import ALL_DATASETS, DATASETS, SPLITS, load_rows
from .models import (expected_template_hash, pinned_revision,
                     resolve as resolve_model)
from .pipeline import RunConfig, load_records, plan, run, slugify
from .report import load_reference, render_markdown, summarise, write_reports


def _csv_list(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.split(",") if v.strip())


def load_dotenv(path: Path = Path(".env")) -> None:
    """Read KEY=VALUE lines from .env, without overwriting the environment.

    Small enough to inline rather than take a dependency on python-dotenv.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


CREDENTIAL_HINT = {
    "openai": "Set OPENAI_API_KEY (or put it in .env), or pass --api-key. "
              "For a local server, add --base-url http://localhost:8000/v1.",
    "anthropic": "Set ANTHROPIC_API_KEY (or put it in .env), or pass --api-key.",
    "hf": "Check the model id. Gated repos such as Llama need `huggingface-cli "
          "login` or HF_TOKEN.",
}


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    load_dotenv()
    for ds in args.datasets:
        if ds not in DATASETS:
            raise SystemExit(f"unknown dataset {ds!r}; choose from {ALL_DATASETS}")
    for c in args.conditions:
        if c not in prompts.CONDITIONS:
            raise SystemExit(f"unknown condition {c!r}; choose from {prompts.CONDITIONS}")

    # An alias like "llama3.2-3b" becomes the full repo id before anything
    # downstream uses the name, so run directories and reports stay consistent.
    args.model = resolve_model(args.model)
    backend_name = resolve_backend(args.model, args.backend)

    revision = args.revision if args.revision else pinned_revision(args.model)
    if revision and revision.lower() in ("main", "none"):
        revision = None
    if backend_name != "hf":
        revision = None  # only the hf backend loads by revision
    # A checkpoint directory on disk is whatever it is; there is no revision to
    # ask the Hub for. The chat-template check below is what covers that case.
    if revision and Path(args.model).exists():
        revision = None

    tqa_scoring = args.tqa_scoring
    if tqa_scoring == "auto":
        tqa_scoring = "loglik" if backend_name == "hf" else "letter"
    if tqa_scoring == "loglik" and backend_name != "hf":
        raise SystemExit(
            f"the {backend_name} backend cannot score candidate continuations. "
            "Use --tqa-scoring letter, or run the model locally with --backend hf.")

    # The limit is part of the directory name, so trying 50 items first and
    # then running the full split does not collide.
    default_name = f"{slugify(args.model)}_{args.split}"
    if args.limit:
        default_name += f"_n{args.limit}"
    out_dir = Path(args.out) if args.out else Path("runs") / default_name
    config = RunConfig(
        model=args.model, backend=backend_name, datasets=tuple(args.datasets),
        conditions=tuple(args.conditions), split=args.split, limit=args.limit,
        tqa_scoring=tqa_scoring, revision=revision,
        max_new_tokens=args.max_new_tokens, out_dir=out_dir,
    )

    counts = plan(config)
    conditions = ["baseline"] + [c for c in config.conditions if c != "baseline"]
    total = sum(counts[ds] * len(conditions) for ds in counts)
    done = sum(len(load_records(out_dir / "raw" / ds / f"{c}.jsonl"))
               for ds in counts for c in conditions)

    print(f"model      {config.model}  (backend: {backend_name})")
    print(f"split      {config.split}" + (f", first {config.limit} items/dataset"
                                          if config.limit else ""))
    print(f"conditions {', '.join(conditions)}")
    print(f"datasets   " + ", ".join(f"{ds} ({counts[ds]})" for ds in counts))
    print(f"output     {out_dir}")
    if revision:
        print(f"revision   {revision[:12]}  (pinned to the published run; "
              f"--revision main takes the current Hub version)")
    elif expected_template_hash(args.model):
        print("revision   not pinned (local checkpoint); the chat template is "
              "checked instead")
    print(f"work       {total - done} model calls remaining of {total}")
    if tqa_scoring == "loglik" and "truthfulqa" in counts:
        print("           (TruthfulQA scores every option, so it costs more "
              "than one call per item)")
    if not args.yes and total - done > 2000 and sys.stdin.isatty():
        if input("continue? [y/N] ").strip().lower() not in ("y", "yes"):
            return 1

    try:
        backend = build_backend(
            args.model, args.backend, dtype=args.dtype, device_map=args.device_map,
            batch_size=args.batch_size, concurrency=args.concurrency,
            base_url=args.base_url, api_key=args.api_key,
            temperature=args.temperature, max_tokens=args.max_new_tokens,
            revision=revision,
        )
    except Exception as exc:  # noqa: BLE001 - provider SDKs raise their own types
        hint = CREDENTIAL_HINT.get(backend_name, "")
        if "requires Accelerate" in str(exc) or "accelerate" in str(exc).lower():
            hint = ("Local weights need accelerate. Install the whole local "
                    "path with `pip install -e \".[hf]\"`, or add it with "
                    "`pip install 'accelerate>=0.26'`. `--device-map none` "
                    "loads without it on a single device.")
        elif "does not recognize this architecture" in str(exc):
            hint = ("This checkpoint is newer than the installed transformers. "
                    "Upgrade it (`pip install -U 'transformers>=4.51'`) and try "
                    "again; Qwen3 needs 4.51 and Gemma 3 needs 4.50.")
        raise SystemExit(f"could not start the {backend_name} backend: {exc}\n"
                         + hint) from None
    expected = expected_template_hash(args.model)
    actual = getattr(backend, "template_sha256", None)
    if expected and actual and expected == actual:
        print("template   matches the one the published rates were measured with")
    if expected and actual and expected != actual:
        print(f"\nwarning: this model's chat template is not the one the "
              f"published rates were measured with\n"
              f"         expected {expected[:12]}, loaded {actual[:12]}\n"
              f"         Every prompt is built by that template, so rates can "
              f"move for this reason alone.\n")

    try:
        from tqdm import tqdm
        with tqdm(total=total, initial=done, unit="call", dynamic_ncols=True) as bar:
            results = run(config, backend, progress=bar)
    except KeyboardInterrupt:
        print(f"\ninterrupted. Everything finished is in {out_dir}; "
              "rerun the same command to pick up where it stopped.")
        return 130
    finally:
        backend.close()

    summary = summarise(results, config)
    paths = write_reports(summary, out_dir)
    print()
    print(render_markdown(summary))
    print(f"\nwrote {paths['report']}, {paths['metrics']}, {paths['csv']}")
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> int:
    """Recompute the report from the raw records of a finished run."""
    out_dir = Path(args.run_dir)
    config_path = out_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"no config.json in {out_dir}; is that a run directory?")
    saved = json.loads(config_path.read_text())
    saved["out_dir"] = out_dir
    saved.pop("started_at", None)
    config = RunConfig(**{k: v for k, v in saved.items()
                          if k in RunConfig.__dataclass_fields__})

    # Keep the run's own dataset order rather than the order they hit the disk.
    ordered = [d for d in config.datasets] + [
        d.name for d in sorted((out_dir / "raw").glob("*"))
        if d.is_dir() and d.name not in config.datasets]
    results: dict[str, dict] = {}
    for name in ordered:
        ds_dir = out_dir / "raw" / name
        if not ds_dir.is_dir():
            continue
        per_condition = {p.stem: load_records(p)
                         for p in sorted(ds_dir.glob("*.jsonl"))}
        if per_condition.get("baseline"):
            results[name] = per_condition

    summary = summarise(results, config)
    paths = write_reports(summary, out_dir)
    print(render_markdown(summary))
    print(f"\nwrote {paths['report']}, {paths['metrics']}, {paths['csv']}")
    return 0


# ---------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------

def cmd_preview(args: argparse.Namespace) -> int:
    """Print the exact conversations a model would see. No model is loaded."""
    rows = load_rows(args.dataset, args.split, limit=None)
    pool = prompts.build_popqa_pool(rows) if args.dataset == "popqa" else None
    if args.qid >= 0:
        row = next((r for r in rows if int(r["qid"]) == args.qid), None)
        if row is None:
            raise SystemExit(
                f"qid {args.qid} is not in the {args.dataset} {args.split} split. "
                f"Try --split cal, or --split both.")
    else:
        row = rows[args.index % len(rows)]

    first = prompts.first_turn(args.dataset, row)
    print(f"# {args.dataset}  qid={row['qid']}  split={args.split}\n")
    print("--- first turn (user) " + "-" * 40)
    print(first)
    print("\n--- gold answer " + "-" * 46)
    print(prompts.gold_answer(args.dataset, row))
    for condition in prompts.DEFAULT_CONDITIONS:
        if condition == "baseline":
            continue
        print(f"\n--- second turn: {condition} " + "-" * (42 - len(condition)))
        print(f"({prompts.CONDITION_HELP[condition]})")
        print(prompts.second_turn(condition, args.dataset, row, pool))
    return 0


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

def cmd_info(args: argparse.Namespace) -> int:
    print("Datasets")
    for name, spec in DATASETS.items():
        counts = {s: len(load_rows(name, s)) for s in ("cal", "test")}
        print(f"  {name:<12} {spec.task}")
        print(f"  {'':<12} cal {counts['cal']}, test {counts['test']}  "
              f"| evidence: {spec.evidence_source}  | {spec.citation}")
    print("\nConditions")
    for name in prompts.CONDITIONS:
        mark = "*" if name in prompts.DEFAULT_CONDITIONS else " "
        print(f" {mark} {name:<18} {prompts.CONDITION_HELP[name]}")
    print("   (* = run by default and reported)")
    print("\nBackbones with published rates (compared against automatically)")
    for model in ("meta-llama/Llama-3.1-8B-Instruct",
                  "meta-llama/Llama-3.2-3B-Instruct",
                  "google/gemma-3-4b-it", "Qwen/Qwen3-8B"):
        ref = load_reference(model, "test")
        avg = None
        if ref:
            avg = sum(v["r_uy"] for v in ref.values()) / len(ref)
        print(f"  {model:<38}" + (f" mean R_UY {avg:.1f}" if avg else ""))
    return 0


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sru",
        description="Measure Unsupported-Yielding and Rational-Updating "
                    "separately, for any chat model.")
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="evaluate a model")
    r.add_argument("--model", required=True,
                   help="Hugging Face repo id, local path, or API model name")
    r.add_argument("--backend", default="auto", choices=("auto",) + BACKENDS)
    r.add_argument("--datasets", type=_csv_list, default=ALL_DATASETS,
                   help=f"comma separated, default all: {','.join(ALL_DATASETS)}")
    r.add_argument("--conditions", type=_csv_list,
                   default=prompts.DEFAULT_CONDITIONS,
                   help="comma separated, default: "
                        + ",".join(prompts.DEFAULT_CONDITIONS))
    r.add_argument("--split", default="test", choices=SPLITS,
                   help="test is the held-out split (default); the paper's "
                        "Table 1 reports cal")
    r.add_argument("--limit", type=int, default=None,
                   help="first N items per dataset, for a quick look")
    r.add_argument("--out", default=None, help="run directory (default runs/<model>_<split>)")
    r.add_argument("--revision", default=None,
                   help="Hugging Face revision for --backend hf. Defaults to "
                        "the revision the published rates were measured on for "
                        "the four paper backbones; pass 'main' for the current "
                        "Hub version")
    r.add_argument("--tqa-scoring", default="auto",
                   choices=("auto", "loglik", "letter"),
                   help="loglik reproduces the paper and needs local weights; "
                        "letter works with any chat API")
    r.add_argument("--max-new-tokens", type=int, default=None,
                   help="override the per-dataset generation budget")
    r.add_argument("--batch-size", type=int, default=8, help="hf backend")
    r.add_argument("--concurrency", type=int, default=8, help="api backends")
    r.add_argument("--dtype", default="bfloat16",
                   choices=("bfloat16", "float16", "float32", "auto"))
    r.add_argument("--device-map", default="auto",
                   help='accelerate device map; "none" loads on one device '
                        "without accelerate")
    r.add_argument("--base-url", default=None,
                   help="OpenAI-compatible server, e.g. http://localhost:8000/v1")
    r.add_argument("--api-key", default=None, help="prefer the environment variable")
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--yes", action="store_true", help="skip the size confirmation")
    r.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="rebuild the report from a run directory")
    p.add_argument("run_dir")
    p.set_defaults(func=cmd_report)

    v = sub.add_parser("preview", help="print the prompts for one item")
    v.add_argument("--dataset", default="truthfulqa", choices=ALL_DATASETS)
    v.add_argument("--split", default="test", choices=SPLITS)
    v.add_argument("--index", type=int, default=0)
    v.add_argument("--qid", type=int, default=-1)
    v.set_defaults(func=cmd_preview)

    i = sub.add_parser("info", help="show datasets, conditions, and reference models")
    i.set_defaults(func=cmd_info)

    m = sub.add_parser("compare",
                       help="one table across several finished runs")
    m.add_argument("run_dirs", nargs="+",
                   help="run directories, or a parent holding several")
    m.add_argument("--out", default=None,
                   help="also write comparison.md and comparison.csv here")
    m.set_defaults(func=cmd_compare)

    c = sub.add_parser("verify-data",
                       help="check the shipped data against data/manifest.json")
    c.add_argument("--dataset", action="append", choices=ALL_DATASETS,
                   help="check one dataset; repeatable, default all")
    c.set_defaults(func=cmd_verify_data)
    return parser


def cmd_compare(args) -> int:
    from .aggregate import collect, find_runs, render_markdown, write_csv

    run_dirs = find_runs([Path(p) for p in args.run_dirs])
    if not run_dirs:
        raise SystemExit(
            "no finished runs there. A run directory is one holding "
            "metrics.json, which `sru run` writes when it completes.")
    collected = collect(run_dirs)
    markdown = render_markdown(collected)
    print(markdown)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "comparison.md").write_text(markdown)
        write_csv(collected, out_dir / "comparison.csv")
        print(f"wrote {out_dir / 'comparison.md'}, "
              f"{out_dir / 'comparison.csv'}")
    return 0


def cmd_verify_data(args) -> int:
    from .integrity import check_files, check_rows, load_manifest, row_problems

    datasets = tuple(args.dataset) if args.dataset else None
    manifest = load_manifest()
    print(f"manifest  release {manifest['release']}  "
          f"schema {manifest['schema_version']}\n")

    bad = 0
    print("checksums")
    for check in check_files(datasets):
        if check.missing:
            state, bad = "MISSING", bad + 1
        elif check.ok:
            state = "ok"
        else:
            state, bad = "CHANGED", bad + 1
        print(f"  {state:<8} {check.dataset:<11} {check.role:<16} "
              f"{check.path.name}")

    print("\nrows and splits")
    for row in check_rows(datasets):
        problems = row_problems(row)
        bad += len(problems)
        state = "ok" if not problems else "BAD"
        print(f"  {state:<8} {row['dataset']:<11} {row['rows']} rows, "
              f"cal {row['cal']} / test {row['test']}")
        for problem in problems:
            print(f"           - {problem}")

    if bad:
        print(f"\n{bad} problem(s). The data does not match the manifest, so "
              "numbers from it are not comparable with anyone else's. Restore "
              "the files from a clean checkout.")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
