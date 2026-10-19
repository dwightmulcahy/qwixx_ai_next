"""Promotion guardrails for the frozen row-change hybrid."""
from unittest.mock import patch

from qwixx_ai.champion import champion, phase91_champion
from qwixx_ai.core import Game
from qwixx_ai.frozen_row_policy import (
    MARGIN, WEIGHTS, row_change_only_from_choices,
)


def test_frozen_settings():
    assert len(WEIGHTS) == 14
    assert MARGIN == 0.02


def test_pass_and_lock_and_same_row_preserved():
    mark = (0, 7, False)
    assert row_change_only_from_choices(None, mark) is None
    assert row_change_only_from_choices(mark, None) == mark
    assert row_change_only_from_choices(mark, (0, 8, False)) == mark
    assert row_change_only_from_choices(mark, (1, 12, True)) == mark
    assert row_change_only_from_choices((0, 12, True), mark) == (
        0, 12, True
    )


def test_different_row_allowed():
    assert row_change_only_from_choices(
        (0, 7, False), (1, 8, False)
    ) == (1, 8, False)


def test_promoted_policy_legal():
    game = Game(n=4, seed=3)
    assert champion(game) in [None, *game.legal()]


def test_non_four_player_fallback_unchanged():
    game = Game(n=3, seed=3)
    assert champion(game) == phase91_champion(game)


def test_champion_uses_original_baseline():
    game = Game(n=4, seed=3)
    baseline = (0, 7, False)
    proposed = (1, 8, False)
    with patch("qwixx_ai.champion.phase91_champion", return_value=baseline):
        with patch(
            "qwixx_ai.frozen_row_policy.model_choice",
            return_value=proposed,
        ):
            assert champion(game) == proposed
