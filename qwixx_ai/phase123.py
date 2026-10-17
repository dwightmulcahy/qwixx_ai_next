"""Phase 12.3: independent, pre-registered skip-cost confirmation.

Fresh deterministic games, paired V2 continuation seeds and game-cluster
bootstrap on the between-group contrast. Diagnostic only; no promotion.
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .phase118 import collect_states
from .phase119 import paired_actions
from .phase121 import action_details, snapshot


def fingerprint(state):
    payload = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def bootstrap_contrast(records, iterations=4000, seed=12030002):
    """Resample whole games, retaining both skip groups within each game."""
    if iterations < 100:
        raise ValueError("iterations must be >= 100")
    by_game = defaultdict(list)
    for row in records:
        by_game[row["game_seed"]].append(row)
    games = sorted(by_game)
    if len(games) < 2:
        return None
    rng = random.Random(seed)
    contrasts = []
    for _ in range(iterations):
        chosen = [by_game[games[rng.randrange(len(games))]] for _ in games]
        values = {"3+": [], "0-2": []}
        for game_rows in chosen:
            for row in game_rows:
                values[row["skip_group"]].append(row["win_gap"])
        if all(values.values()):
            contrasts.append(mean(values["3+"]) - mean(values["0-2"]))
    if len(contrasts) < iterations * 0.95:
        return {"estimable": False, "valid_resamples": len(contrasts)}
    contrasts.sort()
    return {
        "estimable": True,
        "valid_resamples": len(contrasts),
        "ci95": [contrasts[int(0.025 * len(contrasts))],
                 contrasts[min(len(contrasts) - 1, int(0.975 * len(contrasts))) - 1]],
    }


def confirm(checkpoint, games=160, seed=12030001, per_group=60,
            futures=128, rollout_seed=12030003, bootstraps=4000):
    if games < 2 or per_group < 2 or futures < 2:
        raise ValueError("games >= 2, per_group >= 2, futures >= 2 required")
    states, counts = collect_states(
        checkpoint, games=games, seed=seed, max_states=games * 200)
    eligible = []
    for item in states:
        if item["stage"] not in ("middle", "late") or item["category"] != "row_choice":
            continue
        source = action_details(item["game"], item["source_action"])
        group = "3+" if source["skipped_cells"] >= 3 else "0-2"
        eligible.append((item, group))
    buckets = defaultdict(list)
    for item, group in eligible:
        buckets[(item["stage"], item["actor"], group)].append(item)
    for items in buckets.values():
        items.sort(key=lambda x: x["priority"])
    selected = []
    group_counts = {"3+": 0, "0-2": 0}
    while True:
        progress = False
        for key in sorted(buckets):
            group = key[2]
            if buckets[key] and group_counts[group] < per_group:
                selected.append((buckets[key].pop(0), group))
                group_counts[group] += 1
                progress = True
        if not progress:
            break
    rows = []
    for item, group in selected:
        game = item["game"]
        paired = paired_actions(
            game, item["source_action"], item["champion_action"],
            futures=futures, rollout_seed=rollout_seed)
        state = snapshot(game)
        rows.append({
            **{k: v for k, v in item.items() if k not in ("game", "priority")},
            "skip_group": group,
            "state_sha256": fingerprint(state),
            "state": state,
            "source_move": action_details(game, item["source_action"]),
            "champion_move": action_details(game, item["champion_action"]),
            "win_gap": paired["win_share"]["mean"],
            "score_gap": paired["score"]["mean"],
        })
    groups = {}
    for group in ("0-2", "3+"):
        subset = [r for r in rows if r["skip_group"] == group]
        groups[group] = {
            "states": len(subset),
            "games": len({r["game_seed"] for r in subset}),
            "mean_win_gap": mean(r["win_gap"] for r in subset) if subset else None,
        }
    contrast = bootstrap_contrast(rows, bootstraps)
    if contrast and contrast.get("estimable"):
        contrast["observed_difference"] = (
            groups["3+"]["mean_win_gap"] - groups["0-2"]["mean_win_gap"])
        contrast["ci_above_zero"] = contrast["ci95"][0] > 0
    return {
        "phase": "12.3", "checkpoint": str(checkpoint),
        "game_seed": seed, "games": games, "futures": futures,
        "rollout_seed": rollout_seed, "per_group": per_group,
        "selection": "priority-sorted round robin by stage, actor and skip group",
        "all_disagreements_by_category": counts,
        "groups": groups, "contrast": contrast,
        "decisions": rows, "promoted": False,
        "note": "Fresh-seed selected-state V2-continuation association only. No causal claims or optimal labels.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--games", type=int, default=160)
    p.add_argument("--seed", type=int, default=12030001)
    p.add_argument("--per-group", type=int, default=60)
    p.add_argument("--futures", type=int, default=128)
    p.add_argument("--rollout-seed", type=int, default=12030003)
    p.add_argument("--bootstraps", type=int, default=4000)
    p.add_argument("--output", type=Path, default=Path("runs/phase12/phase123_confirmation.json"))
    args = p.parse_args(argv)
    if not args.checkpoint.is_file():
        p.error(f"checkpoint does not exist: {args.checkpoint}")
    result = confirm(args.checkpoint, args.games, args.seed, args.per_group,
                     args.futures, args.rollout_seed, args.bootstraps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "decisions"}, indent=2))
    print(f"Detailed decisions: {args.output}")


if __name__ == "__main__":
    main()
