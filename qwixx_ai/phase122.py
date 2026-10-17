"""Phase 12.2: stratified, game-clustered analysis of Phase 12.1 decisions.

Exploratory associations only. No causal attribution, training or promotion.
"""
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


def group_summary(rows, bootstraps=2000, seed=12020001):
    if not rows:
        return None
    games = defaultdict(list)
    for row in rows:
        games[row["game_seed"]].append(row["win_gap"])
    game_values = [mean(values) for _, values in sorted(games.items())]
    result = {
        "states": len(rows), "games": len(games),
        "state_weighted_gap": mean(r["win_gap"] for r in rows),
        "game_weighted_gap": mean(game_values),
        "mean_score_gap": mean(r["score_gap"] for r in rows),
    }
    if len(games) < 2:
        result["game_bootstrap_ci95"] = None
        return result
    rng = random.Random(seed)
    estimates = []
    for _ in range(bootstraps):
        sample = [game_values[rng.randrange(len(game_values))]
                  for _ in game_values]
        estimates.append(mean(sample))
    estimates.sort()
    result["game_bootstrap_ci95"] = [
        estimates[int(0.025 * bootstraps)],
        estimates[min(bootstraps - 1, math.ceil(0.975 * bootstraps) - 1)],
    ]
    return result


def bins(row):
    source = row["source_move"]
    champion = row["champion_move"]
    skip = source["skipped_cells"]
    progress = source["existing_marks"]
    return {
        "stage": row["stage"],
        "source_color": ("red", "yellow", "green", "blue")[source["color"]],
        "champion_color": ("red", "yellow", "green", "blue")[champion["color"]],
        "source_skips": "3+" if skip >= 3 else str(skip),
        "source_progress": "0-2" if progress <= 2 else "3-5" if progress <= 5 else "6+",
        "source_red_green": "red_green" if source["color"] in (0, 2) else "yellow_blue",
        "skip_3plus": "3+" if skip >= 3 else "0-2",
    }


def analyze(report, bootstraps=2000, seed=12020001):
    if report.get("phase") != "12.1":
        raise ValueError("expected Phase 12.1 diagnostic JSON")
    if bootstraps < 100:
        raise ValueError("bootstraps must be >= 100")
    rows = report["decisions"]
    grouped = defaultdict(list)
    for row in rows:
        dimensions = bins(row)
        for name, value in dimensions.items():
            grouped[(name, value)].append(row)
        grouped[("stage_x_skip", row["stage"] + "/" + dimensions["skip_3plus"])].append(row)
        grouped[("stage_x_progress", row["stage"] + "/" + bins(row)["source_progress"])].append(row)
        grouped[("stage_x_color", row["stage"] + "/" + bins(row)["source_red_green"])].append(row)
    result = defaultdict(dict)
    for (name, value), subset in sorted(grouped.items()):
        result[name][value] = group_summary(subset, bootstraps, seed)
    return {
        "phase": "12.2", "input_states": len(rows),
        "bootstrap_resamples": bootstraps, "seed": seed,
        "overall": group_summary(rows, bootstraps, seed),
        "strata": dict(result), "promoted": False,
        "note": "Exploratory selected-state associations; subgroup CIs are not adjusted for multiple comparisons or selection. Independent fresh-seed confirmation is required.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("runs/phase12/phase121_diagnostics.json"))
    p.add_argument("--output", type=Path, default=Path("runs/phase12/phase122_strata.json"))
    p.add_argument("--bootstraps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=12020001)
    args = p.parse_args(argv)
    result = analyze(json.loads(args.input.read_text()), args.bootstraps, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
