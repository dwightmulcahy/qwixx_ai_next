"""Phase 11.9: independent confirmation of mid/late row-choice decisions.

Uses fresh game and rollout seeds, common random numbers for paired actions,
and game-cluster confidence intervals. V2 continuation is not an optimal oracle.
No training or promotion.
"""
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist, mean, stdev

from .env import decode
from .phase118 import collect_states
from .phase9_search import outcome, state_seed


def interval(values, confidence=0.95):
    """Normal-approximation CI over independent sampling units."""
    if len(values) < 2 or not 0 < confidence < 1:
        raise ValueError("at least two samples and valid confidence required")
    center = mean(values)
    margin = NormalDist().inv_cdf((1 + confidence) / 2) * (
        stdev(values) / math.sqrt(len(values)))
    return {"mean": center, "ci_low": center - margin,
            "ci_high": center + margin, "units": len(values),
            "confidence": confidence}


def paired_actions(game, source_action, champion_action, futures=128,
                   rollout_seed=11091002):
    """Use identical future seeds for the two actions at a given state."""
    if futures < 2:
        raise ValueError("futures must be at least two")
    if source_action == champion_action:
        raise ValueError("actions must differ")
    legal = {None, *game.legal()}
    a, b = decode(source_action), decode(champion_action)
    if a not in legal or b not in legal:
        raise ValueError("actions must be legal in the supplied state")
    rng = random.Random(state_seed(game, rollout_seed))
    gaps, score_gaps = [], []
    for _ in range(futures):
        future_seed = rng.randrange(2**64)
        source_win, source_score = outcome(game, a, future_seed)
        champion_win, champion_score = outcome(game, b, future_seed)
        gaps.append(champion_win - source_win)
        score_gaps.append(champion_score - source_score)
    return {"win_share": interval(gaps), "score": interval(score_gaps),
            "champion_preferred": interval(gaps)["ci_low"] > 0,
            "source_preferred": interval(gaps)["ci_high"] < 0}


def confirm(checkpoint, games=120, game_seed=11091001, max_states=120,
            futures=128, rollout_seed=11091002):
    if games < 2 or max_states < 1 or futures < 2:
        raise ValueError("games >= 2, max_states >= 1, futures >= 2 required")
    # Collect more states than the requested quota, then filter before sampling.
    # Selection is independent of rollout results, preventing winner's curse.
    states, categories = collect_states(
        checkpoint, games=games, seed=game_seed,
        max_states=games * 200)
    focused = [item for item in states if item["stage"] in ("middle", "late")
               and item["category"] == "row_choice"]
    # Balanced round-robin by stage and seat; priority comes from game state,
    # never from observed rollout advantage.
    buckets = defaultdict(list)
    for item in focused:
        buckets[(item["stage"], item["actor"])].append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda x: x["priority"])
    selected = []
    keys = sorted(buckets)
    while len(selected) < max_states:
        progress = False
        for key in keys:
            if len(selected) >= max_states:
                break
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                progress = True
        if not progress:
            break
    rows = []
    for item in selected:
        paired = paired_actions(
            item["game"], item["source_action"], item["champion_action"],
            futures=futures, rollout_seed=rollout_seed)
        rows.append({
            **{k: v for k, v in item.items() if k not in ("game", "priority")},
            "paired_win_share": paired["win_share"],
            "paired_score": paired["score"],
            "champion_preferred": paired["champion_preferred"],
            "source_preferred": paired["source_preferred"],
        })
    # Treat all states from the same game as one cluster; a naive state-level
    # CI would understate uncertainty from correlated positions.
    per_game = defaultdict(list)
    for row in rows:
        per_game[row["game_seed"]].append(row["paired_win_share"]["mean"])
    game_means = [mean(values) for values in per_game.values()]
    aggregate = interval(game_means) if len(game_means) >= 2 else None
    by_stage = {}
    for stage in ("middle", "late"):
        stage_rows = [row for row in rows if row["stage"] == stage]
        grouped = defaultdict(list)
        for row in stage_rows:
            grouped[row["game_seed"]].append(row["paired_win_share"]["mean"])
        by_stage[stage] = {
            "states": len(stage_rows), "games": len(grouped),
            "champion_preferred_with_95pct_ci": sum(
                row["champion_preferred"] for row in stage_rows),
            "source_preferred_with_95pct_ci": sum(
                row["source_preferred"] for row in stage_rows),
            "game_cluster_ci": (
                interval([mean(v) for v in grouped.values()])
                if len(grouped) >= 2 else None),
        }
    return {
        "phase": "11.9", "checkpoint": str(checkpoint),
        "games": games, "game_seed": game_seed,
        "futures_per_action": futures, "rollout_seed": rollout_seed,
        "focus": "middle/late row_choice",
        "max_states": max_states, "eligible_disagreements": len(focused),
        "sampled_states": len(rows), "sampled_games": len(per_game),
        "all_disagreements_by_category": categories,
        "aggregate_game_cluster_ci": aggregate,
        "by_stage": by_stage, "decisions": rows,
        "promoted": False,
        "note": "Exploratory V2-continuation results; game-cluster normal CI "
                "does not account for all selection or simulator biases. "
                "Do not treat rollout preference as optimal labels.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=120)
    parser.add_argument("--game-seed", type=int, default=11091001)
    parser.add_argument("--max-states", type=int, default=120)
    parser.add_argument("--futures", type=int, default=128)
    parser.add_argument("--rollout-seed", type=int, default=11091002)
    parser.add_argument("--output", type=Path, default=Path(
        "runs/phase11/phase119_confirmation.json"))
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    result = confirm(args.checkpoint, args.games, args.game_seed,
                     args.max_states, args.futures, args.rollout_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "decisions"},
                     indent=2))
    print(f"Detailed decisions: {args.output}")


if __name__ == "__main__":
    main()
