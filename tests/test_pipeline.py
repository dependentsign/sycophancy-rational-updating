"""End-to-end pipeline behaviour, driven by a stand-in backend."""
import hashlib
import json

import pytest

from sru import report
from sru.backends.base import Backend
from sru.datasets import load_rows
from sru.pipeline import RunConfig, load_records, run


def _hash(text: str, n: int) -> int:
    return int(hashlib.sha1(text.encode()).hexdigest(), 16) % n


class FakeBackend(Backend):
    """Deterministic answers derived from the prompt, plus a call counter."""

    name = "fake"
    model_id = "fake/model"
    supports_loglik = True
    batch_size = 4

    def __init__(self):
        self.calls = 0

    def generate(self, conversations, max_new_tokens):
        out = []
        for messages in conversations:
            self.calls += 1
            prompt = messages[0]["content"]
            if "Answer: True" in prompt:            # EX-FEVER cue
                out.append("Answer: " + ("True" if _hash(prompt, 2) else "False"))
            elif "Answer: <letter>" in prompt:      # AQuA cue
                out.append("Answer: " + "ABCDE"[_hash(prompt, 5)])
            else:                                   # PopQA
                out.append(f"entity number {_hash(prompt, 7)}")
        return out

    def score_choices(self, conversation, choices):
        self.calls += 1
        seed = _hash(conversation[0]["content"], len(choices))
        return [1.0 if i == seed else 0.0 for i in range(len(choices))]


@pytest.fixture
def config(tmp_path):
    return RunConfig(model="fake/model", backend="fake", split="test", limit=4,
                     out_dir=tmp_path / "run")


def test_run_covers_every_dataset_and_condition(config):
    results = run(config, FakeBackend())
    assert set(results) == set(config.datasets)
    for dataset, conditions in results.items():
        assert set(conditions) == {"baseline", "pressure", "evidence", "user_evidence"}
        for records in conditions.values():
            assert len(records) == 4


def test_records_land_on_disk_and_carry_the_second_turn(config):
    run(config, FakeBackend())
    path = config.out_dir / "raw" / "popqa" / "pressure.jsonl"
    records = load_records(path)
    assert len(records) == 4
    for record in records.values():
        assert record["second_turn"].startswith("I think the answer is")
        assert "correct" in record and "response" in record


def test_the_first_turn_is_the_models_own_answer(config):
    run(config, FakeBackend())
    baseline = load_records(config.out_dir / "raw" / "exfever" / "baseline.jsonl")
    assert all(r["first_turn_text"] == r["response"] for r in baseline.values())


def test_a_second_run_resumes_instead_of_recomputing(config):
    first = FakeBackend()
    run(config, first)
    second = FakeBackend()
    run(config, second)
    assert first.calls > 0
    assert second.calls == 0


def test_changing_the_protocol_refuses_to_reuse_the_directory(config):
    run(config, FakeBackend())
    clashing = RunConfig(model="fake/model", backend="fake", split="cal",
                         limit=4, out_dir=config.out_dir)
    with pytest.raises(SystemExit, match="different settings"):
        run(clashing, FakeBackend())


def test_a_truncated_line_does_not_break_resume(config):
    run(config, FakeBackend())
    path = config.out_dir / "raw" / "aqua" / "baseline.jsonl"
    with path.open("a") as f:
        f.write('{"qid": 999, "cond')
    assert len(load_records(path)) == 4


def test_report_renders_and_writes_all_three_files(config):
    results = run(config, FakeBackend())
    summary = report.summarise(results, config)
    paths = report.write_reports(summary, config.out_dir)
    assert all(p.exists() for p in paths.values())
    text = paths["report"].read_text()
    assert "R_UY" in text and "Average" in text
    metrics = json.loads(paths["metrics"].read_text())
    assert set(metrics["datasets"]) == set(config.datasets)
    assert metrics["average"]["acc"] is not None


def test_limit_takes_a_stable_prefix():
    a = [r["qid"] for r in load_rows("popqa", "test", limit=5)]
    b = [r["qid"] for r in load_rows("popqa", "test", limit=10)][:5]
    assert a == b == sorted(a)


def test_truthfulqa_options_are_shuffled_before_being_shown():
    from sru.datasets import lettered_choices
    rows = load_rows("truthfulqa", "test", limit=40)
    gold_positions = {lettered_choices(r)[1] for r in rows}
    # Gold sits first in every raw row; the letters must not inherit that.
    assert len(gold_positions) > 1
    for row in rows:
        lettered, gold_idx = lettered_choices(row)
        assert lettered[gold_idx][1] == row["mc1_targets"]["choices"][0]
