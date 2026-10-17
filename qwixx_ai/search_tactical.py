"""Search with tactical endgame rollouts and measured computational cost.

Experimental only. Phase 9.1 stays frozen. Candidate and reference are
compared on identical game seeds; rollout RNG is independent per action.
"""
import argparse
import json
import random
import time
from pathlib import Path

from .champion import champion
from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2
from .phase9_search import state_seed
from .phase91_search import eligible
from .champion import SETTINGS
from .search_selective import choose_from_estimates


def tactical_rollout(game):
    """V2 until endgame; then score immediate outcomes when game can end.

    The endgame evaluation is deliberately one-ply and cheap: choose the
    highest terminal win share, with focal score as a tie-breaker. Otherwise V2.
    """
    baseline = heuristic_v2(game)
    if game.turn < 30:
        return baseline
    seat = game.actor
    choices = [None, *game.legal()]
    best_action = baseline
    best_value = None
    for action in choices:
        trial = game.clone()
        trial.step(action)
        if not trial.done:
            continue
        scores = trial.scores()
        top = max(scores)
        share = (1.0 / scores.count(top)) if scores[seat] == top else 0.0
        value = (share, scores[seat])
        if best_value is None or value > best_value:
            best_action, best_value = action, value
    return best_action


def simulate(game, action, future_seed, rollout):
    trial = game.clone()
    seat = trial.actor
    trial.rng = random.Random(future_seed)
    trial.step(action)
    while not trial.done:
        trial.step(rollout(trial))
    scores = trial.scores()
    best = max(scores)
    share = 1.0 / scores.count(best) if scores[seat] == best else 0.0
    return share, scores[seat]


def select(game, initial=32, maximum=64, ambiguity=0.06,
           seed=16000000, rollout="tactical", stats=None):
    if initial < 1 or maximum < initial or ambiguity < 0:
        raise ValueError("invalid search settings")
    if rollout not in ("v2", "tactical"):
        raise ValueError("rollout must be v2 or tactical")
    teacher = heuristic_v2(game)
    if game.n != 4 or not eligible(game, teacher, SETTINGS["start_turn"], SETTINGS["end_turn"]):
        if stats is not None:
            stats["skipped"] += 1
        return teacher
    actions = [None, *game.legal()]
    wins = [0.0] * len(actions)
    scores = [0.0] * len(actions)
    rng = random.Random(state_seed(game, seed))
    continuation = tactical_rollout if rollout == "tactical" else heuristic_v2
    started = time.perf_counter()

    def sample(count):
        for _ in range(count):
            future = rng.randrange(2**64)
            for i, action in enumerate(actions):
                share, score = simulate(game, action, future, continuation)
                wins[i] += share
                scores[i] += score

    sample(initial)
    baseline_index = actions.index(teacher)
    near = any(
        abs((wins[i] - wins[baseline_index]) / initial - SETTINGS["min_gain"]) <= ambiguity
        and scores[i] >= scores[baseline_index] - ambiguity * initial * 25
        for i in range(len(actions)) if i != baseline_index
    )
    used = initial
    if near and maximum > initial:
        sample(maximum - initial)
        used = maximum
    selected = choose_from_estimates(
        actions, wins, scores, used, teacher,
        SETTINGS["min_gain"], SETTINGS["min_score_gain"])
    if stats is not None:
        stats["searched"] += 1
        stats["futures"] += used
        stats["action_rollouts"] += used * len(actions)
        stats["refined"] += used > initial
        stats["overrides"] += selected != teacher
        stats["search_seconds"] += time.perf_counter() - started
    return selected


def evaluate(rounds=8, seed=16020000, initial=32, maximum=64,
             ambiguity=0.06, rollout="tactical"):
    if rounds < 2:
        raise ValueError("rounds must be at least two")
    stats = dict(skipped=0, searched=0, futures=0, action_rollouts=0,
                 refined=0, overrides=0, search_seconds=0.0)
    def candidate(game):
        return select(game, initial, maximum, ambiguity, rollout=rollout,
                      stats=stats)
    started = time.perf_counter()
    comparison = paired_compare(candidate, champion, [heuristic_v2] * 3,
                                rounds=rounds, seed=seed)
    elapsed = time.perf_counter() - started
    return {
        "experiment": "tactical_rollout_search",
        "settings": dict(initial=initial, maximum=maximum,
                         ambiguity=ambiguity, rollout=rollout),
        "versus_phase91": comparison,
        "search_stats": stats,
        "mean_search_seconds": (stats["search_seconds"] / stats["searched"]
                                if stats["searched"] else 0),
        "benchmark_wall_seconds": elapsed,
        "promoted": False,
        "note": "Exploratory. Wall time includes both policies. Tactical "
                "continuation adds terminal one-ply checks, not full tree search.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=16020000)
    parser.add_argument("--initial", type=int, default=32)
    parser.add_argument("--maximum", type=int, default=64)
    parser.add_argument("--ambiguity", type=float, default=0.06)
    parser.add_argument("--rollout", choices=("v2", "tactical"), default="tactical")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = evaluate(args.rounds, args.seed, args.initial, args.maximum,
                      args.ambiguity, args.rollout)
    destination = args.output or Path(
        f"runs/search/{args.rollout}_rollout_seed{args.seed}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        parser.error(f"refusing to overwrite {destination}")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
