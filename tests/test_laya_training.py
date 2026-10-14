"""Validate stopping and group weighting independently of Torch inference."""

import math

import pytest

from scripts.train_laya_kaggle import ValidationStopper, selection_loss


def prediction(profile, group, probability, covered=True):
    return {"profile": profile, "group_id": group, "target": "a", "candidate_ids": ["a", "b"],
            "scores": [probability, 1 - probability], "retrieval_covered": covered}


def test_related_variants_do_not_multiply_group_weight():
    examples = [prediction("asr", "a", 0.9), prediction("asr", "b", 0.2),
                prediction("basic", "c", 0.6)]
    first = selection_loss(examples)[0]
    duplicated = examples + [examples[0]] * 20
    assert selection_loss(duplicated)[0] == pytest.approx(first)


def test_retrieval_misses_do_not_become_supervised_choice_labels():
    examples = [prediction("asr", "a", 0.5)]
    assert selection_loss(examples + [prediction("asr", "missing", 0, False)])[0] == pytest.approx(math.log(2))


def test_stopping_retains_baseline_if_training_never_improves():
    stopper = ValidationStopper(1, patience=3)
    assert stopper.update(1.1, 142) == (False, False)
    assert stopper.update(1.2, 284) == (False, False)
    assert stopper.update(1.3, 426) == (False, True)
    assert stopper.best_step == 0


def test_improvement_resets_patience_and_small_noise_does_not():
    stopper = ValidationStopper(1, patience=2, min_delta=0.005)
    assert stopper.update(1.1, 142) == (False, False)
    assert stopper.update(0.9, 284) == (True, False)
    assert stopper.update(0.898, 426) == (False, False)
    assert stopper.update(0.897, 568) == (False, True)
    assert stopper.best_step == 284


def test_invalid_validation_cannot_select_a_checkpoint():
    with pytest.raises(ValueError, match="Non-finite"):
        ValidationStopper(1).update(float("nan"), 142)
    with pytest.raises(ValueError, match="reachable"):
        selection_loss([prediction("asr", "missing", 0, False)])
