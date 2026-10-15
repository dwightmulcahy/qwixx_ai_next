"""Phase 10.5.1: independently validate rollout labels on fresh future seeds.

Replays deterministic collection trajectories to recover game states. This does
not train, change checkpoints, or promote a policy.
"""
import argparse
import json
from pathlib import Path

from .champion import champion
from .core import Game
from .env import encode
from .heuristic_v2 import heuristic_v2
from .model import choose, load
from .phase9_search import evaluate_actions


def validate(dataset, checkpoint, futures=128, seed=3101000,
             min_gain=0.12, min_score_gain=0.0, max_samples=200):
    if futures < 2 or max_samples < 1 or min_gain < 0:
        raise ValueError("invalid diagnostic settings")
    payload = json.loads(Path(dataset).read_text())
    if payload["source"] != str(checkpoint):
        raise ValueError("dataset source does not match checkpoint")
    if "game_seed" not in payload or "samples" not in payload:
        raise ValueError(
            "Dataset lacks replay identifiers. Recollect with the updated "
            "Phase 10.5 collector before running this diagnostic.")
    targets = {(row["game_seed"], row["seat"], row["decision_index"]): row
               for row in payload["samples"]}
    if len(targets) != len(payload["samples"]):
        raise ValueError("duplicate dataset replay identifiers")
    from .phase105_correction import prepare
    selected = {}
    for row in payload["samples"]:
        labels = prepare([row], min_gain, min_score_gain)
        if labels:
            selected[(row["game_seed"], row["seat"], row["decision_index"])] = labels[0][1]
    if not selected:
        return {"evaluated": 0, "reason": "No training labels passed the filter",
                "promoted": False}
    if len(selected) > max_samples:
        selected = dict(list(sorted(selected.items()))[:max_samples])
    records = []
    model = load(checkpoint)
    games = payload["games_per_seat"]
    original_seed = payload["game_seed"]
    for i in range(games):
        for seat in range(4):
            game = Game(seed=original_seed+i)
            decision_index = 0
            while not game.done:
                if game.actor != seat:
                    game.step(heuristic_v2(game))
                    continue
                neural = choose(model, game, epsilon=0)
                reference = champion(game)
                if neural != reference:
                    key = (original_seed+i, seat+1, decision_index)
                    if key in selected:
                        alternative = selected[key]
                        # Independent future seeds, never the collection seeds.
                        estimates = evaluate_actions(game, futures=futures,
                                                     seed=seed+len(records))
                        lookup = {encode(e["action"]): e for e in estimates}
                        baseline = lookup[encode(neural)]
                        proposed = lookup[alternative]
                        records.append({
                            "game_seed": key[0], "seat": key[1],
                            "decision_index": key[2],
                            "training_gain": targets[key]["estimated_gain"],
                            "holdout_win_gain": proposed["win_share"]-baseline["win_share"],
                            "holdout_score_gain": proposed["mean_score"]-baseline["mean_score"],
                            "improved": proposed["win_share"] > baseline["win_share"],
                        })
                decision_index += 1
                game.step(neural)
    if len(records) != len(selected):
        raise ValueError("Failed to replay all selected states")
    gains = [r["holdout_win_gain"] for r in records]
    scores = [r["holdout_score_gain"] for r in records]
    return {
        "phase": "10.5.1", "dataset": str(dataset),
        "evaluated": len(records), "futures_per_action": futures,
        "independent_seed": seed,
        "fraction_positive_holdout": sum(g > 0 for g in gains)/len(gains),
        "mean_holdout_win_gain": sum(gains)/len(gains),
        "mean_holdout_score_gain": sum(scores)/len(scores),
        "mean_training_estimated_gain": sum(r["training_gain"] for r in records)/len(records),
        "by_seat": {
            str(seat): {"count": sum(r["seat"] == seat for r in records),
                        "mean_holdout_win_gain": sum(
                            r["holdout_win_gain"] for r in records
                            if r["seat"] == seat) / max(1, sum(
                                r["seat"] == seat for r in records))}
            for seat in range(1, 5)},
        "records": records, "promoted": False,
        "note": "Exploratory selected-label validation, not unbiased policy value. "
                "Future seeds are independent, but all continuations use V2.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--futures", type=int, default=128)
    p.add_argument("--seed", type=int, default=3101000)
    p.add_argument("--min-gain", type=float, default=0.12)
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--output", type=Path,
                   default=Path("runs/phase105/label_diagnostics.json"))
    args = p.parse_args(argv)
    result = validate(args.dataset, args.candidate, args.futures,
                      args.seed, args.min_gain, max_samples=args.max_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"},
                     indent=2))


if __name__ == "__main__":
    main()
