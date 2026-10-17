"""Constrained UCT: shallow future search with conservative Phase 9.1 fallback.

Root decisions are evaluated on *paired* future dice, independently of UCT
exploration. UCT only informs later decisions; no action is promoted without
meeting win-share and score safeguards against the frozen incumbent.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

from .champion import SETTINGS, champion
from .controlled_strategy import paired_compare
from .heuristic_v2 import heuristic_v2
from .phase9_search import state_seed
from .phase91_search import eligible
from .search_uct import Node, choose_uct, observed_key


def continuation(trial, seat, tree, rng, exploration, max_nodes, depth):
    """UCT for at most depth future focal decisions, then V2 for all."""
    path = []
    expanded = 0
    while not trial.done:
        if trial.actor == seat and expanded < depth:
            key = observed_key(trial)
            node = tree.get(key)
            if node is None and len(tree) < max_nodes:
                node = Node(tuple([None, *trial.legal()]))
                tree[key] = node
            if node is not None:
                action = choose_uct(node, exploration, rng)
                path.append((node, action))
                expanded += 1
            else:
                action = heuristic_v2(trial)
        else:
            action = heuristic_v2(trial)
        trial.step(action)
    scores = trial.scores()
    best = max(scores)
    reward = 1.0 / scores.count(best) if scores[seat] == best else 0.0
    for node, action in path:
        node.visits += 1
        node.counts[action] = node.counts.get(action, 0) + 1
        node.rewards[action] = node.rewards.get(action, 0.0) + reward
    return reward, scores[seat]


def root_estimates(game, actions, simulations, depth, exploration,
                   max_nodes, seed):
    """Common future seeds for all actions; root action gets equal samples."""
    rng = random.Random(state_seed(game, seed))
    wins = [0.0] * len(actions)
    scores = [0.0] * len(actions)
    trees = [{} for _ in actions]
    for _ in range(simulations):
        future_seed = rng.randrange(2**64)
        for i, action in enumerate(actions):
            trial = game.clone()
            seat = trial.actor
            trial.rng = random.Random(future_seed)
            trial.step(action)
            reward, score = continuation(
                trial, seat, trees[i], rng, exploration, max_nodes, depth)
            wins[i] += reward
            scores[i] += score
    return wins, scores, sum(map(len, trees))


def guarded_choice(actions, wins, scores, incumbent, simulations,
                   min_gain=0.10, min_score_gain=0.0):
    """Never switch unless both expected outcome and score clear margins."""
    baseline = actions.index(incumbent)
    eligible_actions = [
        i for i, action in enumerate(actions)
        if action != incumbent
        and wins[i] >= wins[baseline] + simulations * min_gain
        and scores[i] >= scores[baseline] + simulations * min_score_gain
    ]
    return (actions[max(eligible_actions, key=lambda i: (wins[i], scores[i]))]
            if eligible_actions else incumbent)


def select(game, simulations=64, depth=2, exploration=1.0,
           max_nodes=128, seed=18000000, min_gain=0.10,
           min_score_gain=0.0, stats=None):
    if (simulations < 2 or depth < 0 or exploration < 0 or max_nodes < 1
            or min_gain < 0 or min_score_gain < 0):
        raise ValueError("invalid constrained search settings")
    teacher = heuristic_v2(game)
    if game.n != 4 or not eligible(
        game, teacher, SETTINGS["start_turn"], SETTINGS["end_turn"]
    ):
        if stats is not None:
            stats["skipped"] += 1
        return teacher
    started = time.perf_counter()
    incumbent = champion(game)
    actions = [None, *game.legal()]
    wins, scores, nodes = root_estimates(
        game, actions, simulations, depth, exploration, max_nodes, seed)
    result = guarded_choice(
        actions, wins, scores, incumbent, simulations, min_gain, min_score_gain)
    if stats is not None:
        stats["searched"] += 1
        stats["simulations"] += simulations
        stats["action_rollouts"] += simulations * len(actions)
        stats["nodes"] += nodes
        stats["overrides"] += result != incumbent
        stats["search_seconds"] += time.perf_counter() - started
        stats["root_diagnostics"].append({
            "turn": game.turn,
            "incumbent": incumbent,
            "chosen": result,
            "actions": [
                {"action": action, "win_share": wins[i] / simulations,
                 "mean_score": scores[i] / simulations}
                for i, action in enumerate(actions)
            ],
        })
    return result


def evaluate(rounds=8, seed=18020000, simulations=64, depth=2,
             exploration=1.0, max_nodes=128, min_gain=0.10):
    if rounds < 2:
        raise ValueError("rounds must be at least two")
    stats = dict(skipped=0, searched=0, simulations=0, action_rollouts=0,
                 nodes=0, overrides=0, search_seconds=0.0, root_diagnostics=[])
    def candidate(game):
        return select(game, simulations, depth, exploration, max_nodes,
                      min_gain=min_gain, stats=stats)
    start = time.perf_counter()
    comparison = paired_compare(candidate, champion, [heuristic_v2] * 3,
                                rounds=rounds, seed=seed)
    diagnostics = stats.pop("root_diagnostics")
    return {
        "experiment": "constrained_uct",
        "settings": dict(simulations=simulations, depth=depth,
                         exploration=exploration, max_nodes=max_nodes,
                         min_gain=min_gain),
        "versus_phase91": comparison,
        "search_stats": stats,
        "mean_search_seconds": stats["search_seconds"] / stats["searched"]
        if stats["searched"] else 0,
        "benchmark_wall_seconds": time.perf_counter() - start,
        "root_diagnostics": diagnostics,
        "promoted": False,
        "note": "Exploratory paired comparison. Root actions get equal "
                "chance samples; shallow UCT affects continuation decisions.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=18020000)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--exploration", type=float, default=1.0)
    parser.add_argument("--max-nodes", type=int, default=128)
    parser.add_argument("--min-gain", type=float, default=0.10)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/search/constrained_uct_exploratory.json"))
    args = parser.parse_args(argv)
    report = evaluate(args.rounds, args.seed, args.simulations, args.depth,
                      args.exploration, args.max_nodes, args.min_gain)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    summary = dict(report)
    summary["root_diagnostics"] = {
        "count": len(report["root_diagnostics"]),
        "saved_to": str(args.output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
