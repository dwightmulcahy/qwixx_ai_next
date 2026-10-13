"""Phase 5.4: stronger league sampling and multi-opponent checkpoint selection.

Training selection uses reusable development seeds. Final validation must use
independent, precommitted seeds through qwixx_ai.phase53.
"""
from .phase53 import controlled_evaluate


def exploration_rate(episode, episodes, start=0.08, finish=0.015):
    """Warm-start epsilon annealing, deliberately lower than Phase 5."""
    if episodes < 1 or not 0 <= episode < episodes:
        raise ValueError("invalid episode range")
    return start + (finish - start) * min(1.0, episode / max(1, episodes * 0.8))


def opponent_pool(permanent_count, snapshot_count):
    """Weighted opponent tickets for each nonlearner seat.

    The fixed champion and historical policies dominate the pool while
    snapshots/current/heuristic preserve some strategic diversity.
    """
    if permanent_count < 3 or snapshot_count < 1:
        raise ValueError("Phase 5.4 requires champion, two historical and a snapshot")
    tickets = ["current", "heuristic"]
    tickets += [0] * 4
    tickets += [("permanent", 0)] * 8
    for index in range(1, permanent_count):
        tickets += [("permanent", index)] * 6
    tickets += list(range(1, snapshot_count))
    return tickets


def select_checkpoint(challenger, champion, historical, rounds=375,
                      seed=900000, alpha=0.05, max_checks=10,
                      lineup=None, incumbent_score=float("-inf")):
    """Rank across all baselines, then independently gate champion promotion.

    Selection requires nonnegative advantage against each historical baseline
    and improvement over the current champion; the conservative league score
    is the minimum paired advantage across all baselines. A selected checkpoint
    is *not* automatically promoted. The promotion gate uses multiplicity-
    adjusted confidence bounds for all comparisons.
    """
    history = list(historical)
    if len(history) < 2:
        raise ValueError("two historical opponents are required")
    if not 0 < alpha < 1 or max_checks < 1:
        raise ValueError("invalid promotion settings")
    fixed = list(lineup) if lineup is not None else [champion, *history[:2]]
    if len(fixed) != 3:
        raise ValueError("fixed lineup must have three policies")
    confidence = 1 - alpha / (max_checks * (1 + len(history)))
    primary = controlled_evaluate(challenger, champion, fixed, rounds, seed, confidence)
    previous = [
        controlled_evaluate(challenger, opponent, fixed, rounds,
                            seed + (index + 1) * 100000, confidence)
        for index, opponent in enumerate(history)
    ]
    comparisons = [primary, *previous]
    score = min(item["paired_advantage"] for item in comparisons)
    selected = (
        primary["paired_advantage"] > 0
        and all(item["paired_advantage"] >= 0 for item in previous)
        and score > incumbent_score
    )
    promoted = all(item["advantage_ci_low"] > 0 for item in comparisons)
    return {
        "promoted": promoted,
        "alpha": alpha,
        "max_checks": max_checks,
        "comparison_confidence": confidence,
        "primary": primary,
        "historical": previous,
        "lineup": "fixed_three",
        "selection": {
            "selected": selected,
            "score": score,
            "incumbent_score": incumbent_score if incumbent_score != float("-inf") else None,
            "metric": "minimum paired advantage across champion and historical opponents",
        },
    }
