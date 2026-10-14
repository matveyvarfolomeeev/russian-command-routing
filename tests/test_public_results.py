"""Reject tampered routing evidence rather than trusting reported correctness."""

import pytest

from scripts.verify_public_results import validate_prediction


def prediction():
    ids = [f"object-{i}" for i in range(20)]
    return {"target": ids[0], "group_id": "group", "candidate_ids": ids,
            "retrieved_ids": ids.copy(), "predicted": ids[0], "correct": True,
            "retrieval_covered": True, "scores": [1.0] + [0.0] * 19}


@pytest.mark.parametrize("change,reason", [
    ({"target": "object-1"}, "target differs"),
    ({"predicted": "object-1"}, "correctness flag"),
    ({"scores": [0.0, 1.0] + [0.0] * 18}, "probability argmax"),
    ({"candidate_ids": ["foreign"] + [f"object-{i}" for i in range(1, 20)]}, "Unknown catalog"),
])
def test_tampered_evidence_is_rejected(change, reason):
    row = prediction()
    expected = {"expected_object_id": row["target"], "group_id": row["group_id"]}
    with pytest.raises(ValueError, match=reason):
        validate_prediction(row | change, expected, set(row["candidate_ids"]))
