"""Putting several finished runs into one table."""
from __future__ import annotations

import json

from sru.aggregate import collect, find_runs, render_markdown, selectivity


def _write_run(tmp_path, name, model, datasets, *, split="test", limit=None,
               tqa_scoring="loglik", started_at="2026-01-01T00:00:00"):
    """A run directory holding just enough for the aggregator to read it."""
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True)
    summary = {
        "model": model, "backend": "hf", "split": split, "limit": limit,
        "tqa_scoring": tqa_scoring,
        "datasets": {
            ds: {"n_items": 100, "headline": head}
            for ds, head in datasets.items()
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary))
    (run_dir / "config.json").write_text(json.dumps({"started_at": started_at}))
    return run_dir


HEAD_A = {"acc": 40.0, "r_uy": 30.0, "r_ru_evidence": 60.0,
          "r_ru_user_evidence": 50.0}
HEAD_B = {"acc": 20.0, "r_uy": 80.0, "r_ru_evidence": 10.0,
          "r_ru_user_evidence": 10.0}


def test_finds_runs_under_a_parent_directory(tmp_path):
    _write_run(tmp_path, "one", "m1", {"truthfulqa": HEAD_A})
    _write_run(tmp_path, "two", "m2", {"truthfulqa": HEAD_B})
    assert len(find_runs([tmp_path])) == 2


def test_a_run_directory_itself_is_accepted(tmp_path):
    run = _write_run(tmp_path, "one", "m1", {"truthfulqa": HEAD_A})
    assert find_runs([run]) == [run]


def test_datasets_of_one_model_are_merged_across_runs(tmp_path):
    _write_run(tmp_path, "tqa", "m1", {"truthfulqa": HEAD_A})
    _write_run(tmp_path, "pop", "m1", {"popqa": HEAD_B})
    collected = collect(find_runs([tmp_path]))
    assert len(collected["models"]) == 1
    assert set(collected["models"][0]["datasets"]) == {"truthfulqa", "popqa"}


def test_the_average_spans_the_datasets_that_reported(tmp_path):
    _write_run(tmp_path, "both", "m1",
               {"truthfulqa": HEAD_A, "popqa": HEAD_B})
    entry = collect(find_runs([tmp_path]))["models"][0]
    assert entry["average"]["acc"] == 30.0          # (40 + 20) / 2
    assert entry["average"]["r_uy"] == 55.0         # (30 + 80) / 2


def test_models_are_ordered_by_the_gap_between_the_two_rates(tmp_path):
    _write_run(tmp_path, "good", "good-model", {"truthfulqa": HEAD_A})
    _write_run(tmp_path, "bad", "bad-model", {"truthfulqa": HEAD_B})
    models = collect(find_runs([tmp_path]))["models"]
    assert [m["model"] for m in models] == ["good-model", "bad-model"]
    assert models[0]["selectivity"] == 30.0         # 60 - 30
    assert models[1]["selectivity"] == -70.0        # 10 - 80


def test_the_most_recent_run_of_a_dataset_wins(tmp_path):
    _write_run(tmp_path, "old", "m1", {"truthfulqa": HEAD_A},
               started_at="2026-01-01T00:00:00")
    _write_run(tmp_path, "new", "m1", {"truthfulqa": HEAD_B},
               started_at="2026-06-01T00:00:00")
    collected = collect(find_runs([tmp_path]))
    assert collected["models"][0]["datasets"]["truthfulqa"]["acc"] == 20.0
    assert any("more than once" in c for c in collected["conflicts"])


def test_a_mixed_split_is_called_out_rather_than_averaged_silently(tmp_path):
    _write_run(tmp_path, "test-run", "m1", {"truthfulqa": HEAD_A}, split="test")
    _write_run(tmp_path, "cal-run", "m1", {"popqa": HEAD_B}, split="cal")
    collected = collect(find_runs([tmp_path]))
    assert any("split differs" in c for c in collected["conflicts"])
    assert "not measuring the same thing" in render_markdown(collected)


def test_a_uniform_scope_is_stated_plainly(tmp_path):
    _write_run(tmp_path, "one", "m1", {"truthfulqa": HEAD_A})
    _write_run(tmp_path, "two", "m2", {"truthfulqa": HEAD_B})
    rendered = render_markdown(collect(find_runs([tmp_path])))
    assert "**split** `test`" in rendered
    assert "not measuring the same thing" not in rendered


def test_an_unreadable_run_is_skipped_not_fatal(tmp_path):
    _write_run(tmp_path, "fine", "m1", {"truthfulqa": HEAD_A})
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "metrics.json").write_text("{not json")
    collected = collect(find_runs([tmp_path]))
    assert len(collected["models"]) == 1
    assert len(collected["skipped"]) == 1


def test_local_checkpoint_paths_are_shown_by_name(tmp_path):
    _write_run(tmp_path, "one", "/data/models/meta-llama/Llama-3.2-3B-Instruct",
               {"truthfulqa": HEAD_A})
    rendered = render_markdown(collect(find_runs([tmp_path])))
    assert "meta-llama/Llama-3.2-3B-Instruct" in rendered
    assert "/data/models/" not in rendered


def test_selectivity_needs_both_rates():
    assert selectivity({"r_ru_evidence": 50.0, "r_uy": 20.0}) == 30.0
    assert selectivity({"r_ru_evidence": None, "r_uy": 20.0}) is None
