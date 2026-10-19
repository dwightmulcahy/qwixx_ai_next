"""Audit frozen-model overrides against Phase 9.1 using paired rollouts.

This is diagnostic analysis, not independent promotion evidence. Outcomes
assume V2 continuation play and do not prove causal game-level improvement.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .champion import phase91_champion
from .core import Game
from .frozen_row_policy import model_choice
from .heuristic_v2 import heuristic_v2
from .phase9_search import evaluate_actions


def classify(before, after):
    if after is None:
        return "pass"
    if after[2]:
        return "lock"
    if before is None:
        return "pass_to_mark"
    if before[0] != after[0]:
        return "row_change"
    return "same_row"


def collect_overrides(games=40, seed=24000000, limit=120):
    if games < 1 or limit < 1:
        raise ValueError("games and limit must be positive")
    candidates = []
    counts = defaultdict(int)
    for game_id in range(games):
        game = Game(n=4, seed=seed + game_id)
        while not game.done:
            # Reproduce a fixed V2-generated state distribution; this is not
            # the distribution of states visited by the learned candidate.
            incumbent = phase91_champion(game)
            chosen = model_choice(game, incumbent)
            if chosen != incumbent:
                category = classify(incumbent, chosen)
                counts[category] += 1
                candidates.append((game_id, game.clone(), incumbent, chosen,
                                   category))
            game.step(heuristic_v2(game))
    rng = random.Random(seed + 701)
    buckets = defaultdict(list)
    for item in candidates:
        game = item[1]
        buckets[(item[4], min(game.turn // 12, 4), game.actor)].append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    keys = sorted(buckets)
    while len(selected) < limit and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop())
    return selected, dict(counts)


def audit(games=40, seed=24000000, limit=120, futures=256,
          rollout_seed=25000000):
    if futures < 1:
        raise ValueError("futures must be positive")
    selected, counts = collect_overrides(games, seed, limit)
    details = []
    for game_id, game, incumbent, chosen, category in selected:
        # Paired chance seeds for the two actions in the same state.
        estimates = evaluate_actions(
            game, futures=futures, seed=rollout_seed + game_id
        )
        by_action = {item["action"]: item for item in estimates}
        reference = by_action[incumbent]
        alternative = by_action[chosen]
        details.append({
            "game_seed": seed + game_id,
            "turn": game.turn, "phase": game.phase, "actor": game.actor,
            "category": category,
            "champion_action": incumbent, "model_action": chosen,
            "champion_win_share": reference["win_share"],
            "model_win_share": alternative["win_share"],
            "win_share_delta": alternative["win_share"] - reference["win_share"],
            "champion_mean_score": reference["mean_score"],
            "model_mean_score": alternative["mean_score"],
            "score_delta": alternative["mean_score"] - reference["mean_score"],
        })
    groups = {}
    for category in sorted({row["category"] for row in details}):
        rows = [row for row in details if row["category"] == category]
        groups[category] = {
            "sampled": len(rows),
            "mean_win_share_delta": sum(r["win_share_delta"] for r in rows) / len(rows),
            "mean_score_delta": sum(r["score_delta"] for r in rows) / len(rows),
            "win_share_better": sum(r["win_share_delta"] > 0 for r in rows),
            "win_share_worse": sum(r["win_share_delta"] < 0 for r in rows),
            "win_share_tied": sum(r["win_share_delta"] == 0 for r in rows),
        }
    details.sort(key=lambda row: (row["win_share_delta"], row["score_delta"]))
    return {
        "experiment": "frozen_model_disagreement_audit",
        "settings": dict(games=games, seed=seed, limit=limit,
                         futures=futures, rollout_seed=rollout_seed),
        "overrides_found": sum(counts.values()),
        "override_categories": counts,
        "sampled": len(details),
        "groups": groups,
        "worst_examples": details[:10],
        "best_examples": details[-10:],
        "decisions": details,
        "promoted": False,
        "note": "Diagnostic V2-continuation paired estimates on V2-generated "
                "states. No training or parameter changes. Selection bias "
                "and rollout bias remain; not causal promotion evidence.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed", type=int, default=24000000)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--futures", type=int, default=256)
    parser.add_argument("--rollout-seed", type=int, default=25000000)
    parser.add_argument("--output", type=Path, default=Path(
        "runs/decision_quality/disagreement_audit.json"))
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    report = audit(args.games, args.seed, args.limit, args.futures,
                   args.rollout_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "decisions"}, indent=2))


if __name__ == "__main__":
    main()
