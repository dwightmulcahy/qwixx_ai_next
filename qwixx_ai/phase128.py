"""Phase 12.8: confidence-gated hybrid, tuned and held-out against V2.

The frozen Phase 9.1 champion is invoked only for small neural top-two
Q margins. No model updates or automatic promotion.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from .champion import champion
from .controlled_strategy import paired_compare
from .env import mask, observe, decode
from .heuristic_v2 import heuristic_v2
from .model import load


def neural_decision(model, game):
    with torch.no_grad():
        values = model(torch.from_numpy(observe(game))).detach().cpu().numpy()
    legal = [i for i, ok in enumerate(mask(game)) if ok]
    ranked = sorted(legal, key=lambda i: (-float(values[i]), i))
    margin = float(values[ranked[0]] - values[ranked[1]]) if len(ranked) > 1 else math.inf
    return decode(ranked[0]), margin


def hybrid(model, threshold, stats=None):
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")

    def act(game):
        action, margin = neural_decision(model, game)
        if stats is not None:
            stats["decisions"] += 1
        if margin <= threshold:
            fallback = champion(game)
            if stats is not None:
                stats["fallbacks"] += 1
                stats["changed"] += fallback != action
            return fallback
        return action
    return act


def paired(model, threshold, rounds, seed):
    stats = {"decisions": 0, "fallbacks": 0, "changed": 0}
    candidate = hybrid(model, threshold, stats)
    # Use a true neural baseline; zero-margin ties must not trigger fallback.
    baseline = lambda game: neural_decision(model, game)[0]
    result = paired_compare(
        candidate, baseline, [heuristic_v2] * 3,
        rounds=rounds, seed=seed,
    )
    return {
        **result,
        "fallback_rate": stats["fallbacks"] / stats["decisions"] if stats["decisions"] else 0.0,
        "changed_rate": stats["changed"] / stats["decisions"] if stats["decisions"] else 0.0,
        "fallback_calls": stats["fallbacks"],
    }


def run(checkpoint, thresholds=(0.0, 0.025, 0.05, 0.1, 0.2),
        tune_rounds=12, eval_rounds=30, tune_seed=12080001,
        eval_seed=12090001):
    thresholds = tuple(float(t) for t in thresholds)
    if len(set(thresholds)) != len(thresholds) or 0.0 not in thresholds or any(t < 0 for t in thresholds):
        raise ValueError("thresholds must be distinct, nonnegative and include 0")
    if tune_rounds < 2 or eval_rounds < 2:
        raise ValueError("round counts must be at least 2")
    if set(range(tune_seed, tune_seed + tune_rounds)) & set(range(eval_seed, eval_seed + eval_rounds)):
        raise ValueError("tuning and evaluation seeds overlap")
    model = load(checkpoint)
    tuning = {}
    for threshold in thresholds:
        if threshold == 0.0:
            # Exact neural baseline: zero is explicitly no fallback.
            tuning[str(threshold)] = {
                "paired_advantage": 0.0, "fallback_rate": 0.0,
                "changed_rate": 0.0, "games_per_policy": tune_rounds * 4,
            }
        else:
            tuning[str(threshold)] = paired(model, threshold, tune_rounds, tune_seed)
    selected = max(thresholds, key=lambda t: (tuning[str(t)]["paired_advantage"], -t))
    evaluation = (paired(model, selected, eval_rounds, eval_seed)
                  if selected > 0 else {
                      "paired_advantage": 0.0, "advantage_ci_low": 0.0,
                      "advantage_ci_high": 0.0, "fallback_rate": 0.0,
                      "changed_rate": 0.0, "games_per_policy": eval_rounds * 4,
                      "note": "No modification selected: self-comparison omitted.",
                  })
    # Separately benchmark both frozen policies on the same held-out seeds.
    baseline = lambda game: neural_decision(model, game)[0]
    champion_reference = paired_compare(
        baseline, champion, [heuristic_v2] * 3,
        rounds=eval_rounds, seed=eval_seed,
    )
    return {
        "phase": "12.8", "checkpoint": str(checkpoint),
        "thresholds": list(thresholds), "tune_rounds": tune_rounds,
        "eval_rounds": eval_rounds, "tune_seed": tune_seed, "eval_seed": eval_seed,
        "tuning": tuning, "selected_threshold": selected,
        "heldout_hybrid_vs_neural": evaluation,
        "heldout_neural_vs_champion": champion_reference,
        "promoted": False,
        "note": "Paired games with V2 opponents, four seats per seed. "
                "Normal-approximation CI clusters by seed. "
                "Comparisons against V2 are not direct hybrid-vs-champion matches.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.0, 0.025, 0.05, 0.1, 0.2])
    parser.add_argument("--tune-rounds", type=int, default=12)
    parser.add_argument("--eval-rounds", type=int, default=30)
    parser.add_argument("--tune-seed", type=int, default=12080001)
    parser.add_argument("--eval-seed", type=int, default=12090001)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/phase12/phase128_hybrid.json"))
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    report = run(args.checkpoint, args.thresholds, args.tune_rounds,
                 args.eval_rounds, args.tune_seed, args.eval_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
