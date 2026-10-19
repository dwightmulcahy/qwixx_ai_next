"""Experimental stochastic UCT search for four-player Qwixx.

Chance is sampled by the game RNG; only the focal player's decision nodes
are expanded. Other seats follow V2. Node keys include the currently observed
dice, so decisions never inspect future rolls. No promotion.
"""
import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from .champion import champion
from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2
from .phase9_search import state_seed
from .phase91_search import eligible
from .champion import SETTINGS


@dataclass
class Node:
    actions: tuple
    visits: int = 0
    counts: dict = field(default_factory=dict)
    rewards: dict = field(default_factory=dict)


def observed_key(game):
    """Observable state only; never include hidden RNG state."""
    return (
        game.actor, game.active, game.phase, game.turn, game.active_marks,
        game.dice, tuple(sorted(game.locks)), tuple(sorted(game.white_locks)),
        tuple((tuple(tuple(row) for row in board.rows), board.penalties)
              for board in game.boards),
    )


def choose_uct(node, exploration, rng):
    unvisited = [a for a in node.actions if node.counts.get(a, 0) == 0]
    if unvisited:
        return rng.choice(unvisited)
    return max(node.actions, key=lambda a: (
        node.rewards[a] / node.counts[a]
        + exploration * math.sqrt(math.log(node.visits + 1) / node.counts[a]),
        rng.random(),
    ))


def simulate(game, seat, tree, rng, exploration, max_nodes):
    trial = game.clone()
    # The current dice remain visible; only future dice are resampled.
    trial.rng = random.Random(rng.randrange(2**64))
    path = []
    while not trial.done:
        if trial.actor == seat:
            key = observed_key(trial)
            node = tree.get(key)
            if node is None and len(tree) < max_nodes:
                node = Node(tuple([None, *trial.legal()]))
                tree[key] = node
            if node is not None:
                action = choose_uct(node, exploration, rng)
                path.append((node, action))
            else:
                action = heuristic_v2(trial)
        else:
            action = heuristic_v2(trial)
        trial.step(action)
    scores = trial.scores()
    top = max(scores)
    reward = 1.0 / scores.count(top) if scores[seat] == top else 0.0
    for node, action in path:
        node.visits += 1
        node.counts[action] = node.counts.get(action, 0) + 1
        node.rewards[action] = node.rewards.get(action, 0.0) + reward
    return reward


def select(game, simulations=128, exploration=1.0, max_nodes=4000,
           seed=17000000, stats=None):
    if simulations < 1 or exploration < 0 or max_nodes < 1:
        raise ValueError("invalid tree search settings")
    teacher = heuristic_v2(game)
    if game.n != 4 or not eligible(
        game, teacher, SETTINGS["start_turn"], SETTINGS["end_turn"]
    ):
        if stats is not None:
            stats["skipped"] += 1
        return teacher
    started = time.perf_counter()
    rng = random.Random(state_seed(game, seed))
    tree = {}
    seat = game.actor
    for _ in range(simulations):
        simulate(game, seat, tree, rng, exploration, max_nodes)
    root = tree[observed_key(game)]
    # Root is sampled with UCT; select by observed mean, not exploration bonus.
    # Unvisited actions are never selected.
    best = max(
        (a for a in root.actions if root.counts.get(a, 0)),
        key=lambda a: (root.rewards[a] / root.counts[a], root.counts[a]),
    )
    if stats is not None:
        stats["searched"] += 1
        stats["simulations"] += simulations
        stats["nodes"] += len(tree)
        stats["overrides"] += best != teacher
        stats["search_seconds"] += time.perf_counter() - started
    return best


def evaluate(rounds=8, seed=17020000, simulations=128,
             exploration=1.0, max_nodes=4000):
    if rounds < 2:
        raise ValueError("rounds must be at least two")
    stats = dict(skipped=0, searched=0, simulations=0, nodes=0,
                 overrides=0, search_seconds=0.0)
    def candidate(game):
        return select(game, simulations, exploration, max_nodes, stats=stats)
    started = time.perf_counter()
    result = paired_compare(candidate, champion, [heuristic_v2] * 3,
                            rounds=rounds, seed=seed)
    return {
        "experiment": "stochastic_uct",
        "settings": dict(simulations=simulations, exploration=exploration,
                         max_nodes=max_nodes),
        "versus_phase91": result,
        "search_stats": stats,
        "mean_search_seconds": stats["search_seconds"] / stats["searched"]
        if stats["searched"] else 0,
        "benchmark_wall_seconds": time.perf_counter() - started,
        "promoted": False,
        "note": "Experimental single-agent UCT with sampled chance and V2 "
                "opponents. Full-information observed-state keys; not an "
                "information-set equilibrium solver.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17020000)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--exploration", type=float, default=1.0)
    parser.add_argument("--max-nodes", type=int, default=4000)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/search/uct_exploratory.json"))
    args = parser.parse_args(argv)
    report = evaluate(args.rounds, args.seed, args.simulations,
                      args.exploration, args.max_nodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
