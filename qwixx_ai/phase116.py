"""Phase 11.6: representative-state anchored training and independent baseline gate.

Data collection uses V2 trajectories and game-level splits. This experiment
never promotes a candidate or changes the frozen champion.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .champion import champion
from .core import Game
from .env import encode, mask, observe
from .heuristic_v2 import heuristic_v2
from .model import choose, load
from .phase111 import split_for_game
from .phase112 import balanced_select, priority, relative_stage
from .phase115 import train as anchored_train


def collect(checkpoint, games=120, seed=11060001, max_records=6000):
    """Sample agreement and disagreement decisions without mixing game splits."""
    if games < 1 or max_records < 1:
        raise ValueError("games and max_records must be positive")
    model = load(checkpoint)
    candidates = []
    counts = Counter()
    for game_id in range(games):
        game = Game(n=4, seed=seed + game_id)
        split = split_for_game(seed, game_id)
        game_records = []
        while not game.done:
            if game.actor in (1, 2):
                neural = choose(model, game, epsilon=0)
                teacher = champion(game)
                agreement = neural == teacher
                counts["agreement" if agreement else "disagreement"] += 1
                game_records.append({
                    "game_id": game_id,
                    "game_seed": seed + game_id,
                    "split": split,
                    "turn": game.turn,
                    "phase": game.phase,
                    "actor": game.actor,
                    "priority": priority(seed, game_id, game.turn, game.phase, game.actor),
                    "observation": observe(game).tolist(),
                    "legal_actions": [int(i) for i in mask(game).nonzero()[0]],
                    "neural_action": encode(neural),
                    "champion_action": encode(teacher),
                    "agreement": agreement,
                    "teacher_source": "phase91_champion",
                })
            game.step(heuristic_v2(game))
        for record in game_records:
            record["stage"] = relative_stage(record["turn"], game.turn)
        candidates.extend(game_records)
    # Balance each split/stage/seat/agreement stratum independently, so
    # disagreement-heavy states cannot overwhelm representative states.
    for record in candidates:
        record["actor"] = int(record["actor"])
    buckets = {}
    for record in candidates:
        key = (record["split"], record["stage"], record["actor"], record["agreement"])
        buckets.setdefault(key, []).append(record)
    for bucket in buckets.values():
        bucket.sort(key=lambda r: r["priority"])
    selected = []
    keys = sorted(buckets)
    while len(selected) < max_records:
        progressed = False
        for key in keys:
            if len(selected) >= max_records:
                break
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                progressed = True
        if not progressed:
            break
    selected.sort(key=lambda r: (r["game_id"], r["turn"], r["phase"], r["actor"]))
    for record in selected:
        record.pop("priority")
    summary = {
        "phase": "11.6", "source_checkpoint": str(checkpoint),
        "games": games, "seed": seed, "sampled": len(candidates),
        "saved": len(selected),
        "sampled_agreement": counts["agreement"],
        "sampled_disagreement": counts["disagreement"],
        "saved_by_split": dict(Counter(r["split"] for r in selected)),
        "saved_by_stage": dict(Counter(r["stage"] for r in selected)),
        "saved_by_agreement": {
            "agreement": sum(r["agreement"] for r in selected),
            "disagreement": sum(not r["agreement"] for r in selected),
        },
        "split_rule": "game-level SHA256(seed:game_id)",
        "sampling": "round-robin split/stage/seat/agreement; deterministic priorities",
        "promoted": False,
    }
    return summary, selected


def save_dataset(path, summary, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--checkpoint", type=Path, required=True)
    collect_parser.add_argument("--games", type=int, default=120)
    collect_parser.add_argument("--seed", type=int, default=11060001)
    collect_parser.add_argument("--max-records", type=int, default=6000)
    collect_parser.add_argument("--output", type=Path, default=Path(
        "runs/phase11/phase116_representative.jsonl"))
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--dataset", type=Path, default=Path(
        "runs/phase11/phase116_representative.jsonl"))
    train_parser.add_argument("--checkpoint", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, default=Path(
        "runs/phase11/candidate_phase116.pt"))
    train_parser.add_argument("--epochs", type=int, default=6)
    train_parser.add_argument("--seed", type=int, default=11060002)
    train_parser.add_argument("--learning-rate", type=float, default=0.000005)
    train_parser.add_argument("--anchor-weight", type=float, default=4.0)
    train_parser.add_argument("--teacher-weight", type=float, default=0.1)
    args = p.parse_args(argv)
    if args.command == "collect":
        summary, records = collect(args.checkpoint, args.games, args.seed,
                                   args.max_records)
        save_dataset(args.output, summary, records)
        print(json.dumps({"dataset": str(args.output), **summary}, indent=2))
    else:
        report = anchored_train(
            args.dataset, args.checkpoint, args.output, epochs=args.epochs,
            seed=args.seed, learning_rate=args.learning_rate,
            anchor_weight=args.anchor_weight, teacher_weight=args.teacher_weight)
        print(json.dumps({k: v for k, v in report.items() if k != "history"},
                         indent=2))


if __name__ == "__main__":
    main()
