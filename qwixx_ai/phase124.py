"""Phase 12.4: diagnose Phase 12.3 game clustering and confounding.

Observational diagnostics; never trains or promotes a checkpoint.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev


def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def describe(rows):
    if not rows:
        return {"states": 0, "games": 0}
    by_game = defaultdict(list)
    for row in rows:
        by_game[row["game_seed"]].append(row["win_gap"])
    values = [mean(v) for v in by_game.values()]
    return {
        "states": len(rows),
        "games": len(by_game),
        "state_mean_gap": mean(r["win_gap"] for r in rows),
        "game_mean_gap": mean(values),
        "between_game_sd": pstdev(values),
        "states_per_game": dict(sorted(Counter(map(len, by_game.values())).items())),
        "effective_games_kish": sum(map(len, by_game.values())) ** 2
        / sum(len(v) ** 2 for v in by_game.values()),
    }


def paired_games(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["game_seed"]][row["skip_group"]].append(row["win_gap"])
    paired = []
    for seed, values in sorted(groups.items()):
        if values["3+"] and values["0-2"]:
            paired.append({"game_seed": seed, "difference": mean(values["3+"]) - mean(values["0-2"])})
    return paired


def bootstrap_paired(pairs, iterations=4000, seed=12040001):
    if iterations < 100:
        raise ValueError("iterations must be >= 100")
    if len(pairs) < 2:
        return {"estimable": False, "paired_games": len(pairs)}
    values = [p["difference"] for p in pairs]
    rng = random.Random(seed)
    samples = sorted(mean(values[rng.randrange(len(values))] for _ in values)
                     for _ in range(iterations))
    return {
        "estimable": True, "paired_games": len(pairs),
        "mean_difference": mean(values),
        "ci95": [quantile(samples, 0.025), quantile(samples, 0.975)],
        "positive_games": sum(v > 0 for v in values),
        "negative_games": sum(v < 0 for v in values),
        "ties": sum(v == 0 for v in values),
    }


def analyze(report, iterations=4000, seed=12040001):
    if report.get("phase") != "12.3":
        raise ValueError("expected Phase 12.3 report")
    rows = report["decisions"]
    strata = defaultdict(list)
    for row in rows:
        source = row["source_move"]
        stage = row["stage"]
        actor = row["actor"]
        progress = source["existing_marks"]
        progress_bin = "0-2" if progress <= 2 else "3-5" if progress <= 5 else "6+"
        color = ("red", "yellow", "green", "blue")[source["color"]]
        for dimension, value in (
            ("skip", row["skip_group"]),
            ("stage", stage),
            ("actor", str(actor)),
            ("color", color),
            ("progress", progress_bin),
            ("stage_x_skip", f"{stage}/{row['skip_group']}"),
            ("actor_x_skip", f"{actor}/{row['skip_group']}"),
            ("progress_x_skip", f"{progress_bin}/{row['skip_group']}"),
            ("color_x_skip", f"{color}/{row['skip_group']}"),
        ):
            strata[(dimension, value)].append(row)
    output = defaultdict(dict)
    for (dimension, value), subset in sorted(strata.items()):
        output[dimension][value] = describe(subset)
    paired = paired_games(rows)
    return {
        "phase": "12.4", "source_phase": "12.3",
        "states": len(rows), "games": len({r["game_seed"] for r in rows}),
        "overall": describe(rows),
        "strata": dict(output),
        "paired_game_contrast": bootstrap_paired(paired, iterations, seed),
        "paired_game_details": paired,
        "group_overlap": {
            "both_groups": len(paired),
            "only_one_group": len({r["game_seed"] for r in rows}) - len(paired),
        },
        "promoted": False,
        "note": "Exploratory diagnostics; sparse strata and paired-game selection limit inference. No new rollouts.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("runs/phase12/phase123_confirmation.json"))
    p.add_argument("--output", type=Path, default=Path("runs/phase12/phase124_diagnostics.json"))
    p.add_argument("--bootstraps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=12040001)
    args = p.parse_args(argv)
    report = analyze(json.loads(args.input.read_text()), args.bootstraps, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "paired_game_details"}, indent=2))
    print(f"Detailed diagnostics: {args.output}")


if __name__ == "__main__":
    main()
