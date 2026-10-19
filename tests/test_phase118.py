"""Phase 11.8 classification and balanced selection tests."""
import pytest

from qwixx_ai.phase118 import (
    action_category, disagreement_category, select_states,
)


def test_action_categories():
    assert action_category(None) == "pass"
    assert action_category((0, 7, False)) == "mark"
    assert action_category((0, 12, True)) == "lock"


def test_disagreement_categories():
    assert disagreement_category(None, (0, 7, False)) == "pass_vs_mark"
    assert disagreement_category((0, 12, True), (1, 8, False)) == "locking"
    assert disagreement_category((0, 7, False), (1, 7, False)) == "row_choice"
    assert disagreement_category((0, 7, False), (0, 8, False)) == "same_row_different_mark"


def test_balanced_selection():
    states = [
        {"stage": "early", "actor": 1, "category": "row_choice", "priority": 3},
        {"stage": "early", "actor": 1, "category": "row_choice", "priority": 1},
        {"stage": "late", "actor": 2, "category": "pass_vs_mark", "priority": 2},
    ]
    result = select_states(states, 2)
    assert len(result) == 2
    assert {r["stage"] for r in result} == {"early", "late"}


def test_invalid_limit():
    with pytest.raises(ValueError):
        select_states([], 0)
