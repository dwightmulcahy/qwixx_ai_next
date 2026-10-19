"""Phase 11.8: investigate source/champion disagreements using paired rollouts.

This is diagnostic research, not an optimal-action oracle. All continuations
use heuristic V2; common future seeds reduce variance but cannot remove bias.
No training, model promotion, or deployment.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from .champion import champion
from .core import Game
from .env import encode
from .heuristic_v2 import heuristic_v2
from .model import choose, load
from .phase112 import priority, relative_stage
from .phase9_search import evaluate_actions


def action_category(action):
    if action is None:
        return "pass"
    return "lock" if action[2] else "mark"


def disagreement_category(source_action, champion_action):
    a, b = action_category(source_action), action_category(champion_action)
    if a == "pass" or b == "pass":
        return "pass_vs_mark"
    if a == "lock" or b == "lock":
        return "locking"
    if source_action[0] != champion_action[0]:
        return "row_choice"
    return "same_row_different_mark"


def select_states(states, limit):
    """Deterministically balance relative stage, seat and disagreement category."""
    if limit < 1:
        raise ValueError("limit must be positive")
    buckets = {}
    for state in states:
        key = (state["stage"], state["actor"], state["category"])
        buckets.setdefault(key, []).append(state)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: item["priority"])
    selected = []
    keys = sorted(buckets)
    while len(selected) < limit:
        progress = False
        for key in keys:
            if len(selected) == limit:
                break
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                progress = True
        if not progress:
            break
    return selected


def collect_states(checkpoint, games=60, seed=11080001, max_states=120):
    if games < 1 or max_states < 1:
        raise ValueError("games and max_states must be positive")
    model = load(checkpoint)
    candidates = []
    all_counts = Counter()
    for game_id in range(games):
        game = Game(n=4, seed=seed + game_id)
        game_states = []
        while not game.done:
            if game.actor in (1, 2):
                source = choose(model, game, epsilon=0)
                teacher = champion(game)
                if source != teacher:
                    category = disagreement_category(source, teacher)
                    all_counts[category] += 1
                    game_states.append({
                        "game_seed": seed + game_id,
                        "game_id": game_id,
                        "turn": game.turn,
                        "phase": game.phase,
                        "actor": game.actor,
                        "category": category,
                        "source_action": encode(source),
                        "champion_action": encode(teacher),
                        "priority": priority(
                            seed, game_id, game.turn, game.phase, game.actor),
                        "game": game.clone(),
                    })
            game.step(heuristic_v2(game))
        for state in game_states:
            state["stage"] = relative_stage(state["turn"], game.turn)
        candidates.extend(game_states)
    return select_states(candidates, max_states), dict(all_counts)


def investigate(checkpoint, games=60, seed=11080001, max_states=120,
                futures=32, search_seed=11080002):
    if futures < 1:
        raise ValueError("futures must be positive")
    states, categories = collect_states(checkpoint, games, seed, max_states)
    results = []
    for item in states:
        estimates = evaluate_actions(item["game"], futures=futures, seed=search_seed)
        by_action = {encode(entry["action"]): entry for entry in estimates}
        source = by_action[item["source_action"]]
        teacher = by_action[item["champion_action"]]
        win_gap = teacher["win_share"] - source["win_share"]
        results.append({
            key: value for key, value in item.items()
            if key not in ("game", "priority")
        } | {
            "source_win_share": source["win_share"],
            "champion_win_share": teacher["win_share"],
            "champion_minus_source_win_share": win_gap,
            "source_mean_score": source["mean_score"],
            "champion_mean_score": teacher["mean_score"],
            "champion_minus_source_mean_score":
                teacher["mean_score"] - source["mean_score"],
            "rollout_preferred": (
                "champion" if win_gap > 0 else
                "source" if win_gap < 0 else "tie"),
        })
    results.sort(key=lambda r: (-abs(r["champion_minus_source_win_share"]),
                                r["game_seed"], r["turn"], r["phase"]))
    grouped = {}
    for dimension in ("stage", "category"):
        grouped[dimension] = {}
        for value in sorted({r[dimension] for r in results}):
            rows = [r for r in results if r[dimension] == value]
            grouped[dimension][value] = {
                "sampled": len(rows),
                "champion_preferred": sum(
                    r["champion_minus_source_win_share"] > 0 for r in rows),
                "source_preferred": sum(
                    r["champion_minus_source_win_share"] < 0 for r in rows),
                "ties": sum(
                    r["champion_minus_source_win_share"] == 0 for r in rows),
                "mean_champion_minus_source_win_share": sum(
                    r["champion_minus_source_win_share"] for r in rows) / len(rows),
            }
    return {
        "phase": "11.8", "source_checkpoint": str(checkpoint),
        "games": games, "seed": seed, "futures_per_action": futures,
        "search_seed": search_seed, "max_states": max_states,
        "sampled_disagreements": len(results),
        "all_disagreements_by_category": categories,
        "summary": grouped, "decisions": results,
        "promoted": False,
        "note": "Exploratory V2-continuation estimates, not optimal labels. "
                "Selection and rollout noise can bias conclusions. "
                "Do not train or promote from these results without validation.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--games", type=int, default=60)
    p.add_argument("--seed", type=int, default=11080001)
    p.add_argument("--max-states", type=int, default=120)
    p.add_argument("--futures", type=int, default=32)
    p.add_argument("--search-seed", type=int, default=11080002)
    p.add_argument("--output", type=Path, default=Path(
        "runs/phase11/phase118_decision_quality.json"))
    args = p.parse_args(argv)
    if not args.checkpoint.is_file():
        p.error(f"checkpoint does not exist: {args.checkpoint}")
    result = investigate(args.checkpoint, args.games, args.seed,
                         args.max_states, args.futures, args.search_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "decisions"},
                     indent=2))
    print(f"Detailed decisions: {args.output}")


if __name__ == "__main__":
    main()
