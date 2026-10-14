import math

import pytest

from scripts.run_launcher_model import choice_scores


def test_sharp_temperature_preserves_order_when_tail_probabilities_underflow():
    probabilities, order = choice_scores([-200.0, -100.0, 0.0], 0.1)
    assert probabilities == [0.0, 0.0, 1.0]
    assert order == [2, 1, 0]


def test_temperature_changes_confidence_without_changing_choice():
    sharp, sharp_order = choice_scores([1.0, 2.0, -5.0], 0.1)
    softer, soft_order = choice_scores([1.0, 2.0, -5.0], 2.49)
    assert sharp_order == soft_order == [1, 0, 2]
    assert max(sharp) > max(softer)
    assert math.fsum(softer) == pytest.approx(1.0)


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_temperature_is_rejected(temperature):
    with pytest.raises(ValueError, match="Температура"):
        choice_scores([0, 1], temperature)
