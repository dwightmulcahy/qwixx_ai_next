"""Fast, non-recursive rollout policy for Monte Carlo search.

This intentionally avoids calling the search-based champion from inside a search.
It combines heuristic V2 with the frozen row-choice model that was previously
validated as a useful champion override.
"""
from .frozen_row_policy import row_change_only
from .heuristic_v2 import heuristic_v2


def rollout_policy(game):
    """Return a stronger continuation action without nested Monte Carlo search."""
    incumbent=heuristic_v2(game)
    return row_change_only(game,incumbent) if game.n==4 else incumbent
