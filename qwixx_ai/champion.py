"""Frozen Phase 9.1 champion policy with V2 fallback."""
from .heuristic_v2 import heuristic_v2
from .phase91_search import policy

SETTINGS = dict(futures=32, seed=900000, min_gain=0.10, min_score_gain=0.0, start_turn=12, end_turn=35)
_search = policy(**SETTINGS)


def champion(game):
    """Use confirmed four-player search; V2 for other player counts."""
    return _search(game) if game.n == 4 else heuristic_v2(game)
