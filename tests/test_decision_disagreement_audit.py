"""Disagreement audit invariants."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.decision_disagreement_audit import (
    audit, classify, collect_overrides, model_choice,
)
from qwixx_ai.champion import champion


def test_categories():
    assert classify((0, 6, False), (1, 6, False)) == "row_change"
    assert classify((0, 6, False), (0, 7, False)) == "same_row"
    assert classify(None, (0, 6, False)) == "pass_to_mark"
    assert classify((0, 6, False), None) == "pass"
    assert classify((0, 11, False), (0, 12, True)) == "lock"


def test_choice_legal():
    game = Game(seed=8)
    assert model_choice(game, champion(game)) in [None, *game.legal()]


def test_sampling_reproducible():
    first, counts_a = collect_overrides(games=2, seed=24000000, limit=4)
    second, counts_b = collect_overrides(games=2, seed=24000000, limit=4)
    assert counts_a == counts_b
    assert [(i, g.turn, a, b) for i, g, a, b, _ in first] == [
        (i, g.turn, a, b) for i, g, a, b, _ in second
    ]


def test_invalid_inputs():
    with pytest.raises(ValueError):
        collect_overrides(games=0)
    with pytest.raises(ValueError):
        audit(futures=0)
