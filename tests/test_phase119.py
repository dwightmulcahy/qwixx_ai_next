"""Tests for Phase 11.9 paired statistics and input checks."""
import pytest

from qwixx_ai.phase119 import interval, paired_actions


def test_interval_constant():
    assert interval([0.25, 0.25, 0.25])["ci_low"] == 0.25
    assert interval([0.25, 0.25, 0.25])["ci_high"] == 0.25


def test_interval_spans_zero():
    result = interval([-1.0, 1.0])
    assert result["ci_low"] < 0 < result["ci_high"]


def test_interval_rejects_one_sample():
    with pytest.raises(ValueError):
        interval([1.0])


def test_paired_actions_requires_futures():
    with pytest.raises(ValueError):
        paired_actions(None, 1, 2, futures=1)


def test_paired_actions_requires_distinct_actions():
    with pytest.raises(ValueError):
        paired_actions(None, 1, 1, futures=2)
