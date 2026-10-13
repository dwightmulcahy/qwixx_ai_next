"""Phase 8: audit actual threshold-4 overrides and paired counterfactuals.

Exploratory only: no policy tuning or promotion. The focal player follows the
hybrid during collection; all other seats follow V2. For each override, paired
rollouts compare the neural action with V2 from the same cloned position.
"""
import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .core import COLORS, Game, ORDER
from .heuristic_v2 import heuristic_v2
from .model import choose, load
from .phase71_counterfactual import finish
from .phase72_hybrid import large_jump


def describe(game, action):
    color, number, locking = action
    board = game.boards[game.actor]
    row = board.rows[color]
    previous = ORDER[color].index(row[-1]) if row else -1
    return {
        "turn": game.turn, "phase": game.phase, "color": color,
        "number": number, "locking": locking,
        "jump": ORDER[color].index(number) - previous - 1,
        "row_marks": len(row), "penalties": board.penalties,
        "total_marks": sum(map(len, board.rows)),
    }


def audit(model, games=1000, seed=810000, futures=16,
          max_overrides=300, bootstrap=2000, output=None):
    if min(games, futures, max_overrides, bootstrap) < 1:
        raise ValueError("all sample sizes must be positive")
    randomizer = random.Random(seed)
    eligible = 0
    overrides = 0
    candidates = []
    for game_id in range(games):
        game = Game(seed=seed + game_id)
        # Rotate the focal hybrid seat; V2 fills the other seats.
        focal = game_id % game.n
        while not game.done:
            teacher = heuristic_v2(game)
            if game.actor == focal:
                eligible += 1
                student = choose(model, game)
                override = (
                    student is not None and teacher is not None
                    and student != teacher
                    and large_jump(game, student, threshold=4)
                )
                if override:
                    overrides += 1
                    record = {
                        "game_id": game_id, "actor": game.actor,
                        "student": list(student), "v2": list(teacher),
                        **describe(game, student),
                    }
                    item = (copy.deepcopy(game), student, teacher, record)
                    # Reservoir sample over all actual overrides.
                    if len(candidates) < max_overrides:
                        candidates.append(item)
                    else:
                        position = randomizer.randrange(overrides)
                        if position < max_overrides:
                            candidates[position] = item
                    action = student
                else:
                    action = teacher
            else:
                action = teacher
            game.step(action)
    records = []
    for game, student, teacher, record in candidates:
        differences = []
        for _ in range(futures):
            future_seed = randomizer.randrange(2**32)
            a_win, a_score = finish(game, student, future_seed)
            b_win, b_score = finish(game, teacher, future_seed)
            differences.append((a_win - b_win, a_score - b_score))
        avg = np.mean(differences, axis=0)
        record["win_share_delta"] = float(avg[0])
        record["score_delta"] = float(avg[1])
        records.append(record)
    by_game = defaultdict(list)
    for record in records:
        by_game[record["game_id"]].append(
            (record["win_share_delta"], record["score_delta"])
        )
    values = np.asarray([np.mean(group, axis=0)
                         for group in by_game.values()], dtype=float)
    estimate = [0.0, 0.0]
    intervals = None
    if len(values):
        estimate = values.mean(axis=0).tolist()
        rng = np.random.default_rng(seed)
        draws = np.empty((bootstrap, 2))
        for i in range(bootstrap):
            indices = rng.integers(0, len(values), len(values))
            draws[i] = values[indices].mean(axis=0)
        intervals = np.quantile(draws, [0.025, 0.975], axis=0).tolist()
    breakdown = {}
    for label, predicate in (
        ("early", lambda r: r["turn"] < 15),
        ("middle", lambda r: 15 <= r["turn"] < 30),
        ("late", lambda r: r["turn"] >= 30),
        ("penalties_0", lambda r: r["penalties"] == 0),
        ("penalties_1_plus", lambda r: r["penalties"] >= 1),
        ("jump_4", lambda r: r["jump"] == 4),
        ("jump_5_plus", lambda r: r["jump"] >= 5),
    ):
        selected = [r for r in records if predicate(r)]
        breakdown[label] = {
            "states": len(selected),
            "mean_win_share_delta": (
                sum(r["win_share_delta"] for r in selected) / len(selected)
                if selected else None
            ),
            "mean_score_delta": (
                sum(r["score_delta"] for r in selected) / len(selected)
                if selected else None
            ),
        }
    report = {
        "games": games, "seed": seed, "futures": futures,
        "focal_decisions": eligible, "actual_overrides": overrides,
        "override_rate": overrides / eligible if eligible else 0,
        "sampled_overrides": len(records),
        "sampled_games": len(by_game),
        "mean_student_minus_v2_win_share": estimate[0],
        "mean_student_minus_v2_score": estimate[1],
        "game_bootstrap_ci_95": {
            "win_share": [intervals[0][0], intervals[1][0]],
            "score": [intervals[0][1], intervals[1][1]],
        } if intervals is not None else None,
        "breakdown": breakdown,
        "override_colors": dict(Counter(COLORS[r["color"]]
                                        for r in records)),
        "warning": "Exploratory only. Paired RNG streams can diverge when "
                   "actions change game duration; bootstrap intervals are "
                   "not proof of causation or independent confirmation.",
    }
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        report["records"] = str(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="runs/phase7/targeted.pt")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=810000)
    parser.add_argument("--futures", type=int, default=16)
    parser.add_argument("--max-overrides", type=int, default=300)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", default="runs/phase8/overrides.jsonl")
    args = parser.parse_args()
    print(json.dumps(audit(load(args.model), args.games, args.seed,
                           args.futures, args.max_overrides,
                           args.bootstrap, args.output), indent=2))


if __name__ == "__main__":
    main()
