"""Phase 10.1: reproducible, seat-aware diagnostics for a frozen challenger.

Diagnostics are descriptive; promotion remains governed by the paired benchmark.
"""
from collections import defaultdict
from statistics import mean

from .core import Game


def _trial(policy, opponents, seat, seed):
    agents = list(opponents)
    agents.insert(seat, policy)
    game = Game(n=4, seed=seed)
    while not game.done:
        game.step(agents[game.actor](game))
    scores = game.scores()
    best = max(scores)
    winners = [i for i, score in enumerate(scores) if score == best]
    return {
        "score": scores[seat],
        "win_share": 1 / len(winners) if seat in winners else 0.0,
        "outcome": "win" if seat in winners and len(winners) == 1 else
                   "tie" if seat in winners else "loss",
        "penalties": game.boards[seat].penalties,
        "marks": sum(len(row) for row in game.boards[seat].rows),
        "locks": sum(bool(row and row[-1] == (12 if c < 2 else 2))
                     for c, row in enumerate(game.boards[seat].rows)),
        "turns": game.turn + 1,
    }


def _summary(records):
    if not records:
        return {}
    return {
        "games": len(records),
        "wins": sum(r["outcome"] == "win" for r in records),
        "ties": sum(r["outcome"] == "tie" for r in records),
        "losses": sum(r["outcome"] == "loss" for r in records),
        "win_share": mean(r["win_share"] for r in records),
        "mean_score": mean(r["score"] for r in records),
        "mean_penalties": mean(r["penalties"] for r in records),
        "mean_marks": mean(r["marks"] for r in records),
        "mean_personal_locks": mean(r["locks"] for r in records),
        "mean_turns": mean(r["turns"] for r in records),
    }


def compare_diagnostics(challenger, reference, opponents, rounds, seed):
    """Use identical game seeds and seat rotation for both policies."""
    if rounds < 1 or len(opponents) != 3:
        raise ValueError("rounds must be positive and exactly three opponents required")
    samples = defaultdict(list)
    for index in range(rounds):
        for seat in range(4):
            for label, policy in (("challenger", challenger), ("reference", reference)):
                samples[(label, seat)].append(_trial(policy, opponents, seat, seed + index))
    return {
        "rounds": rounds,
        "seed": seed,
        "games_per_policy": rounds * 4,
        "challenger": {
            "overall": _summary([r for seat in range(4) for r in samples["challenger", seat]]),
            "by_seat": {str(seat + 1): _summary(samples["challenger", seat])
                        for seat in range(4)},
        },
        "reference": {
            "overall": _summary([r for seat in range(4) for r in samples["reference", seat]]),
            "by_seat": {str(seat + 1): _summary(samples["reference", seat])
                        for seat in range(4)},
        },
        "note": "Descriptive metrics only; not a promotion criterion.",
    }
