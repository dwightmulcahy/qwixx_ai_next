"""Tests for the row-change-only frozen candidate."""
import pytest

from qwixx_ai.champion import champion
from qwixx_ai.core import Game
from qwixx_ai.row_change_experiment import (
    evaluate, row_change_only, row_change_only_from_choices,
)


def test_only_row_changes_allowed():
    base = (0, 7, False)
    assert row_change_only_from_choices(base, (1, 7, False)) == (1, 7, False)
    assert row_change_only_from_choices(base, (0, 8, False)) == base
    assert row_change_only_from_choices(base, None) == base
    assert row_change_only_from_choices(None, (1, 7, False)) is None
    assert row_change_only_from_choices(base, (1, 12, True)) == base
    assert row_change_only_from_choices((0, 12, True), (1, 7, False)) == (
        0, 12, True
    )


def test_row_candidate_legal():
    game = Game(seed=3)
    incumbent = champion(game)
    assert row_change_only(game, incumbent) in [None, *game.legal()]


def test_reject_old_seed_and_rounds():
    with pytest.raises(ValueError, match="fresh"):
        evaluate(seed=23000000)
    with pytest.raises(ValueError, match="rounds"):
        evaluate(rounds=1)
