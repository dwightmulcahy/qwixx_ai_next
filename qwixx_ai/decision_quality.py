"""Decision quality dataset and linear action-value experiment."""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from .champion import phase91_champion
from .controlled_strategy import paired_compare
from .core import Game, ORDER
from .heuristic_v2 import heuristic_v2, probability
from .phase9_search import evaluate_actions


def features(game, action):
    board = game.boards[game.actor]
    scores = game.scores()
    lead = scores[game.actor] - max(
        score for seat, score in enumerate(scores) if seat != game.actor
    )
    base = [game.turn / 60, game.phase / game.n, lead / 50,
            board.penalties / 4, len(game.locks) / 4,
            float(action is None)]
    if action is None:
        return base + [0.0] * 8
    color, number, locking = action
    row = board.rows[color]
    order = ORDER[color]
    index = order.index(number)
    previous = order.index(row[-1]) if row else -1
    return base + [len(row) / 11, (index - previous - 1) / 10,
                   index / 10, probability(number),
                   sum(probability(x) for x in order[index + 1:]),
                   float(locking), float(len(game.locks) == 1 and locking),
                   float(game.actor == game.active)]


def collect(games=30, seed=20000000, max_states=120, futures=128):
    if min(games, max_states, futures) < 1:
        raise ValueError("collection settings must be positive")
    candidates = []
    for game_id in range(games):
        game = Game(n=4, seed=seed + game_id)
        while not game.done:
            if len(game.legal()) >= 2:
                teacher = heuristic_v2(game)
                if teacher != phase91_champion(game) or 12 <= game.turn < 35:
                    candidates.append((game_id, game.clone()))
            game.step(heuristic_v2(game))
    random.Random(seed).shuffle(candidates)
    states = []
    for game_id, game in candidates[:max_states]:
        estimates = evaluate_actions(game, futures, seed=seed + 100000 + game_id)
        states.append({
            "game_id": game_id, "turn": game.turn, "actor": game.actor,
            "actions": [{**item, "features": features(game, item["action"])}
                        for item in estimates],
        })
    return {"settings": dict(games=games, seed=seed, futures=futures),
            "states": states, "label_policy": "V2 rollout, not optimal oracle"}


def fit(dataset, ridge=10.0):
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    xs, ys = [], []
    for state in dataset["states"]:
        actions = state["actions"]
        x = np.asarray([a["features"] for a in actions])
        y = np.asarray([a["win_share"] for a in actions])
        xs.extend(x - x.mean(axis=0))
        ys.extend(y - y.mean())
    if not xs:
        raise ValueError("no states to train")
    x = np.asarray(xs)
    return np.linalg.solve(
        x.T @ x + ridge * np.eye(x.shape[1]), x.T @ np.asarray(ys)
    ).tolist()


def diagnose(dataset, weights):
    regrets = []
    for state in dataset["states"]:
        actions = state["actions"]
        predicted = max(actions, key=lambda a: np.dot(weights, a["features"]))
        regrets.append(max(a["win_share"] for a in actions)
                       - predicted["win_share"])
    return {"states": len(regrets),
            "mean_rollout_regret": sum(regrets) / len(regrets) if regrets else 0}


def policy(weights, margin=0.02):
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    def act(game):
        incumbent = phase91_champion(game)
        actions = [None, *game.legal()]
        values = {a: float(np.dot(weights, features(game, a)))
                  for a in actions}
        best = max(actions, key=lambda a: values[a])
        return best if values[best] > values[incumbent] + margin else incumbent
    return act


def experiment(games=30, states=120, futures=128, rounds=12,
               train_seed=20000000, validation_seed=21000000,
               test_seed=22000000, ridge=10.0, margin=0.02):
    if len({train_seed, validation_seed, test_seed}) != 3 or rounds < 2:
        raise ValueError("invalid evaluation settings")
    train = collect(games, train_seed, states, futures)
    validation = collect(max(4, games // 4), validation_seed,
                         max(12, states // 4), futures)
    weights = fit(train, ridge)
    comparison = paired_compare(
        policy(weights, margin), phase91_champion, [heuristic_v2] * 3,
        rounds=rounds, seed=test_seed)
    return {"weights": weights, "training": diagnose(train, weights),
            "validation": diagnose(validation, weights),
            "versus_phase91": comparison, "promoted": False,
            "note": "Exploratory only. V2 rollout labels are biased; "
                    "use fresh independent seeds before promotion."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--states", type=int, default=120)
    parser.add_argument("--futures", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/decision_quality/experiment.json"))
    args = parser.parse_args(argv)
    result = experiment(args.games, args.states, args.futures, args.rounds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
