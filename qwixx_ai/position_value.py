"""Experimental position-value heuristic with interpretable components.

The frozen V2 and Phase 9.1 champion remain untouched. Compare this policy
against the champion using paired seeds before considering promotion.
"""
import argparse
import json
import math
from pathlib import Path

from .champion import champion
from .controlled_strategy import paired_compare
from .core import ORDER
from .heuristic_v2 import heuristic_v2, probability


def row_future_value(row, color, turn, locks):
    """Expected accessible mark potential, with no claim of exact optimality."""
    if color in locks:
        return 0.0
    order = ORDER[color]
    previous = order.index(row[-1]) if row else -1
    remaining = order[previous + 1:]
    # Probability of at least one appearance in an approximate remaining
    # horizon. This deliberately avoids treating rolls as guaranteed marks.
    horizon = max(1, int((48 - turn) * 0.45))
    reachable = sum(1.0 - (1.0 - probability(number)) ** horizon
                    for number in remaining)
    # More existing marks make later marks more valuable.
    return reachable * (len(row) + 2) * 0.25


def position_value(game, seat):
    board = game.boards[seat]
    current = board.score(game.locks)
    potential = sum(row_future_value(row, color, game.turn, game.locks)
                    for color, row in enumerate(board.rows))
    # Favor relative position without ignoring absolute score.
    other_best = max(game.boards[i].score(game.locks)
                     for i in range(game.n) if i != seat)
    lead = current - other_best
    # Saturated lead signal reduces overreaction to large early leads.
    return current + potential + 3.0 * math.tanh(lead / 18.0)


def choose(game, future_weight=0.35, lock_weight=1.0,
           penalty_weight=1.0):
    if future_weight < 0 or lock_weight < 0 or penalty_weight < 0:
        raise ValueError("weights must be nonnegative")
    legal = game.legal()
    if not legal:
        return None
    seat = game.actor
    baseline = heuristic_v2(game)
    actions = [None, *legal]
    scores = []
    for action in actions:
        trial = game.clone()
        trial.step(action)
        delta = position_value(trial, seat) - position_value(game, seat)
        # On a nonterminal white-dice decision, passing may still be followed
        # by a colored-dice mark; do not assume a penalty immediately.
        if action is None and game.phase == game.n and game.actor == game.active:
            delta -= penalty_weight * (
                trial.boards[seat].penalties - game.boards[seat].penalties
            ) * 5
        if action is not None and action[2]:
            delta *= lock_weight
        scores.append(delta + future_weight * (
            row_future_value(trial.boards[seat].rows[action[0]], action[0],
                             trial.turn, trial.locks)
            if action is not None else 0.0
        ))
    best_index = max(range(len(actions)), key=lambda i: scores[i])
    best = actions[best_index]
    # Small advantage is not enough to discard the established baseline.
    baseline_index = actions.index(baseline)
    return best if scores[best_index] > scores[baseline_index] + 1.0 else baseline


def evaluate(rounds=12, seed=19020000, future_weight=0.35,
             lock_weight=1.0, penalty_weight=1.0):
    if rounds < 2:
        raise ValueError("rounds must be at least two")
    def candidate(game):
        return choose(game, future_weight, lock_weight, penalty_weight)
    result = paired_compare(candidate, champion, [heuristic_v2] * 3,
                            rounds=rounds, seed=seed)
    return {
        "experiment": "position_value",
        "settings": dict(future_weight=future_weight,
                         lock_weight=lock_weight,
                         penalty_weight=penalty_weight),
        "versus_phase91": result,
        "promoted": False,
        "note": "Experimental state evaluation, no Monte Carlo at decision time.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=19020000)
    parser.add_argument("--future-weight", type=float, default=0.35)
    parser.add_argument("--lock-weight", type=float, default=1.0)
    parser.add_argument("--penalty-weight", type=float, default=1.0)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/search/position_value_exploratory.json"))
    args = parser.parse_args(argv)
    report = evaluate(args.rounds, args.seed, args.future_weight,
                      args.lock_weight, args.penalty_weight)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
