"""Protect the claim that schedule comparison changes only presentation order."""

from collections import Counter

import pytest

from scripts.train_nanojev_kaggle import plan_order, validate_inputs


def training_rows():
    return [{"id": f"row-{i}", "source": "clean" if i < 80 else "stt"}
            for i in range(320)]


def test_schedules_have_exactly_the_same_exposures_and_budget():
    data = training_rows()
    mixed, _ = plan_order(data, "mixed", 42)
    staged, prefix = plan_order(data, "clean_first", 42)
    assert mixed != staged
    assert Counter(mixed) == Counter(staged) == Counter(range(len(data)))
    assert prefix == 32
    assert all(data[i]["source"] == "clean" for i in staged[:prefix])
    assert any(data[i]["source"] == "stt" for i in mixed[:prefix])
    assert any(data[i]["source"] == "clean" for i in staged[prefix:])
    assert any(data[i]["source"] == "stt" for i in staged[prefix:])


def test_schedule_is_reproducible():
    data = training_rows()
    assert plan_order(data, "mixed", 7) == plan_order(data, "mixed", 7)
    assert plan_order(data, "mixed", 7) != plan_order(data, "mixed", 8)


def example(group, covered=True):
    return {"id": group, "group_id": group, "candidate_ids": [str(i) for i in range(20)],
            "documents": ["metadata"] * 20, "target": "0" if covered else "missing",
            "retrieval_covered": covered}


def test_validation_retrieval_miss_is_allowed_and_preserved():
    validate_inputs([example("train")], [example("val", covered=False)])


def test_repaired_validation_shortlist_is_rejected():
    val = {**example("val"), "retrieval_covered": False}
    with pytest.raises(ValueError, match="repaired"):
        validate_inputs([example("train")], [val])


def test_related_groups_cannot_enter_validation():
    val = {**example("val"), "group_id": "train"}
    with pytest.raises(ValueError, match="overlap"):
        validate_inputs([example("train")], [val])
