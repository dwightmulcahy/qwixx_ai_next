"""Champion policies for offline strength validation and bounded live play."""
import os
import random
import time

from .frozen_row_policy import row_change_only
from .heuristic_v2 import heuristic_v2
from .runtime_search import outcome, policy, state_seed

SETTINGS = dict(
    futures=32, seed=900000, min_gain=0.10, min_score_gain=0.0,
    start_turn=12, end_turn=35,
)
_search = policy(**SETTINGS)


def phase91_champion(game):
    """Original Phase 9.1 baseline, unchanged."""
    return _search(game) if game.n == 4 else heuristic_v2(game)


def base_champion(game):
    """Frozen 0.8-era Champion strength reference."""
    incumbent = phase91_champion(game)
    return row_change_only(game, incumbent) if game.n == 4 else incumbent


def _live_incumbent(game):
    """Fast strategic baseline used when live search is skipped or exhausted."""
    incumbent = heuristic_v2(game)
    return row_change_only(game, incumbent) if game.n == 4 else incumbent


def _live_should_search(game, incumbent):
    if game.n != 4 or game.done:
        return False
    legal = game.legal()
    if not legal:
        return False
    if any(action[2] for action in legal):
        return True
    penalty_risk = (
        game.phase == game.n
        and game.actor == game.active
        and game.active_marks == 0
        and bool(legal)
    )
    return penalty_risk or (game.turn >= 10 and len(legal) >= 2 and incumbent is not None)


def live_champion(game, *, budget_ms=None, max_futures=None, stats=None):
    """Latency-bounded live Champion.

    Preserve the 0.8-era strategy shape (V2 + row-change + matched future
    rollouts) without allowing the unbounded 32+ future searches used by
    offline Champion policies. Only complete matched futures are considered.
    """
    incumbent = _live_incumbent(game)
    if not _live_should_search(game, incumbent):
        if stats is not None:
            stats.update(searched=False, futures=0, elapsed_ms=0.0, overridden=False)
        return incumbent

    if budget_ms is None:
        budget_ms = float(os.getenv("QWIXX_LIVE_CHAMPION_BUDGET_MS", "350"))
    if max_futures is None:
        max_futures = int(os.getenv("QWIXX_LIVE_CHAMPION_MAX_FUTURES", "12"))
    budget_ms = max(25.0, min(2000.0, float(budget_ms)))
    max_futures = max(1, min(32, int(max_futures)))

    actions = [None, *game.legal()]
    if incumbent not in actions:
        incumbent = heuristic_v2(game)
    if incumbent not in actions:
        incumbent = actions[0]

    base_index = actions.index(incumbent)
    totals = [0.0] * len(actions)
    score_totals = [0.0] * len(actions)
    completed = 0
    rng = random.Random(state_seed(game, SETTINGS["seed"]))
    started = time.perf_counter()
    deadline = started + budget_ms / 1000.0

    for _ in range(max_futures):
        if completed and time.perf_counter() >= deadline:
            break
        future = rng.randrange(2**64)
        samples = []
        for action in actions:
            samples.append(outcome(game, action, future))
        for i, (share, score) in enumerate(samples):
            totals[i] += share
            score_totals[i] += score
        completed += 1

    chosen = incumbent
    if completed:
        baseline_share = totals[base_index] / completed
        baseline_score = score_totals[base_index] / completed
        viable = [
            i for i in range(len(actions))
            if i != base_index
            and totals[i] / completed >= baseline_share + SETTINGS["min_gain"]
            and score_totals[i] / completed >= baseline_score + SETTINGS["min_score_gain"]
        ]
        if viable:
            best = max(viable, key=lambda i: (
                totals[i] / completed,
                score_totals[i] / completed,
            ))
            chosen = actions[best]

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if stats is not None:
        stats.update(
            searched=True,
            futures=completed,
            elapsed_ms=elapsed_ms,
            overridden=chosen != incumbent,
            budget_ms=budget_ms,
            max_futures=max_futures,
        )
    return chosen


def champion(game):
    """Offline/research Champion with full adaptive search strength."""
    incumbent=base_champion(game)
    if game.n!=4:
        return incumbent
    from .search_adaptive import select
    return select(game,incumbent_policy=base_champion)
