"""Independent held-out evaluation of the frozen 120-state action-value model.

Weights are copied from the 2026-10-19 exploratory 120-state run.
No retraining, threshold tuning, or promotion takes place in this module.
"""
import argparse
import json
from pathlib import Path

from .champion import phase91_champion
from .controlled_strategy import paired_compare
from .decision_quality import policy
from .heuristic_v2 import heuristic_v2

FROZEN_WEIGHTS = (
    7.316526290373287e-34, 0.0, 9.190012946650607e-34, 0.0, 0.0,
    -0.020710467226715008, 0.05338450113214413,
    -0.0926056063043526, -0.009760136736374748,
    -0.005655617225870696, 0.046577911421847515,
    0.08235247612364781, 0.046211322675620656,
    0.027243457944153966,
)
FROZEN_MARGIN = 0.02
# Independent of train (20000000), validation (21000000), and
# exploratory evaluation (22000000) seeds.
HELDOUT_SEED = 23000000


def evaluate(rounds=100, seed=HELDOUT_SEED):
    if rounds < 2:
        raise ValueError("rounds must be at least two")
    if seed == 22000000 or seed == 20000000 or seed == 21000000:
        raise ValueError("use a fresh held-out seed")
    stats = {"decisions": 0, "overrides": 0, "pass_overrides": 0,
             "mark_overrides": 0, "lock_overrides": 0}
    learned = policy(FROZEN_WEIGHTS, FROZEN_MARGIN)

    def candidate(game):
        baseline = phase91_champion(game)
        chosen = learned(game)
        stats["decisions"] += 1
        if chosen != baseline:
            stats["overrides"] += 1
            category = ("pass_overrides" if chosen is None else
                        "lock_overrides" if chosen[2] else "mark_overrides")
            stats[category] += 1
        return chosen

    comparison = paired_compare(
        candidate, phase91_champion, [heuristic_v2] * 3, rounds=rounds, seed=seed
    )
    stats["override_rate"] = (
        stats["overrides"] / stats["decisions"] if stats["decisions"] else 0
    )
    return {
        "experiment": "frozen_decision_quality_heldout",
        "model": {"weights": list(FROZEN_WEIGHTS), "margin": FROZEN_MARGIN,
                  "feature_count": len(FROZEN_WEIGHTS)},
        "versus_phase91": comparison,
        "decision_stats": stats,
        "promoted": False,
        "note": "Frozen candidate, fresh seeds. Overrides are counted, but "
                "causal per-override benefit is not established by this test.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=HELDOUT_SEED)
    parser.add_argument("--output", type=Path, default=Path(
        "runs/decision_quality/frozen_heldout_23000000.json"))
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    report = evaluate(args.rounds, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
