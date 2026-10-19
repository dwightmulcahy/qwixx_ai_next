"""Promoted row-change hybrid, preserving Phase 9.1 as a named baseline."""
from .frozen_row_policy import row_change_only
from .heuristic_v2 import heuristic_v2
from .phase91_search import policy

SETTINGS = dict(
    futures=32, seed=900000, min_gain=0.10, min_score_gain=0.0,
    start_turn=12, end_turn=35,
)
_search = policy(**SETTINGS)


def phase91_champion(game):
    """Original Phase 9.1 baseline, unchanged."""
    return _search(game) if game.n == 4 else heuristic_v2(game)


def champion(game):
    """Promoted frozen row-change-only hybrid for four-player games."""
    incumbent = phase91_champion(game)
    return row_change_only(game, incumbent) if game.n == 4 else incumbent
