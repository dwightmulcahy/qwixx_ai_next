"""Phase 10.5.4 confidence sequence and qualification safety tests."""
import math

import pytest

from qwixx_ai.phase105_adaptive import decision_summary
from qwixx_ai.phase105_qualification import qualify


def test_simultaneous_bounds_cover_mean_for_constant_samples():
    result = decision_summary([0.25] * 64, [1] * 64, 0.95, 0.05, looks=4)
    expected = math.sqrt(2 * math.log(2 * 4 / 0.05) / 64)
    assert result["win_ci_low"] == pytest.approx(0.25 - expected)
    assert result["win_ci_high"] == pytest.approx(0.25 + expected)


def test_more_looks_widens_bounds():
    samples = [0.5] * 64
    a = decision_summary(samples, samples, 0.95, 0.05, looks=1)
    b = decision_summary(samples, samples, 0.95, 0.05, looks=4)
    assert b["win_ci_low"] < a["win_ci_low"]


def test_qualification_rejects_too_few_rounds_without_loading_model():
    with pytest.raises(ValueError, match="500"):
        qualify("unused.pt", rounds=499)
