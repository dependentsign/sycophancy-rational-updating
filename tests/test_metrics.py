import pytest

from sru.metrics import average_headline, dataset_metrics


def records(flags):
    return {i: {"correct": int(f)} for i, f in enumerate(flags)}


def test_rates_use_disjoint_denominators():
    m = dataset_metrics({
        "baseline":      records([1, 1, 0, 0]),
        "pressure":      records([1, 0, 0, 1]),
        "evidence":      records([0, 0, 1, 0]),
    })
    assert m["acc"]["rate"] == 50.0
    # Of the two items correct at baseline, one flipped away.
    assert m["headline"]["r_uy"] == 50.0
    # Of the two items wrong at baseline, one flipped to correct.
    assert m["headline"]["r_ru_evidence"] == 50.0
    assert m["conditions"]["pressure"]["flip_correct_to_wrong"]["n"] == 2
    assert m["conditions"]["evidence"]["flip_wrong_to_correct"]["n"] == 2


def test_missing_qids_shrink_the_denominator_only():
    m = dataset_metrics({
        "baseline": records([1, 1, 1, 1]),
        "pressure": {0: {"correct": 0}},
    })
    assert m["conditions"]["pressure"]["flip_correct_to_wrong"] == {
        "rate": 100.0, "n": 1, "k": 1}


def test_a_perfect_model_has_no_yielding():
    m = dataset_metrics({"baseline": records([1, 1]), "pressure": records([1, 1])})
    assert m["headline"]["r_uy"] == 0.0
    assert m["conditions"]["pressure"]["flip_wrong_to_correct"]["rate"] is None


def test_abstentions_are_counted():
    m = dataset_metrics({
        "baseline": {0: {"correct": 1}, 1: {"correct": 0, "abstain": True}},
        "pressure": {0: {"correct": 0, "abstain": True, "degenerate": True}},
    })
    assert m["baseline_abstain"] == 1
    assert m["conditions"]["pressure"]["abstain"] == 1
    assert m["conditions"]["pressure"]["degenerate"] == 1


def test_average_skips_missing_cells():
    avg = average_headline({
        "a": {"headline": {"acc": 40.0, "r_uy": 20.0}},
        "b": {"headline": {"acc": 60.0, "r_uy": None}},
    })
    assert avg["acc"] == 50.0 and avg["r_uy"] == 20.0
    assert avg["r_ru_evidence"] is None
