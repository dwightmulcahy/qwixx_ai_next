"""Phase 11.1: reproducible neural/champion disagreement audit.

Outputs descriptive examples, NOT verified optimal-action labels.
Game-level splits prevent states from the same game leaking across sets.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .champion import champion
from .core import Game
from .env import encode, mask, observe
from .heuristic_v2 import heuristic_v2
from .model import choose, load


def split_for_game(seed, game_id):
    digest = hashlib.sha256(f"{seed}:{game_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def collect(checkpoint, games=100, seed=11010001, max_records=2000):
    if games < 1 or max_records < 1:
        raise ValueError("games and max_records must be positive")
    model = load(checkpoint)
    records = []
    totals = Counter()
    per_split = Counter()
    for game_id in range(games):
        game = Game(n=4, seed=seed + game_id)
        split = split_for_game(seed, game_id)
        while not game.done:
            actor = game.actor
            # Collect only from seats 2 and 3, where fixed-64 was evaluated.
            if actor in (1, 2):
                neural = choose(model, game, epsilon=0)
                expert = champion(game)
                totals["sampled_decisions"] += 1
                if neural != expert:
                    totals["disagreements"] += 1
                    per_split[split] += 1
                    if len(records) < max_records:
                        valid = mask(game)
                        records.append({
                            "game_id": game_id,
                            "game_seed": seed + game_id,
                            "split": split,
                            "turn": game.turn,
                            "phase": game.phase,
                            "actor": actor,
                            "observation": observe(game).tolist(),
                            "legal_actions": [int(i) for i in valid.nonzero()[0]],
                            "neural_action": encode(neural),
                            "champion_action": encode(expert),
                            "scores_before": game.scores(),
                            "penalties": [b.penalties for b in game.boards],
                            "locks": sorted(game.locks),
                        })
            # Fixed deterministic V2 behavior supplies the state distribution.
            game.step(heuristic_v2(game))
        totals["completed_games"] += 1
    summary = {
        "phase": "11.1", "checkpoint": str(checkpoint),
        "seed": seed, "games": games, "max_records": max_records,
        "sampled_decisions": totals["sampled_decisions"],
        "disagreements": totals["disagreements"],
        "disagreement_rate": (
            totals["disagreements"] / totals["sampled_decisions"]
            if totals["sampled_decisions"] else 0),
        "saved_records": len(records),
        "disagreements_by_split": dict(per_split),
        "split_rule": "SHA256(seed:game_id) bucket 80/10/10",
        "policy_distribution": "4 x heuristic_v2; seats 2 and 3 audited",
        "note": "Champion actions are teacher proposals, not proven optimal. "
                "Records are truncated after max_records; counts are not. "
                "Do not train on validation or test examples.",
        "promoted": False,
    }
    return summary, records


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True)
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed", type=int, default=11010001)
    p.add_argument("--max-records", type=int, default=2000)
    p.add_argument("--output", type=Path, default=Path(
        "runs/phase11/phase111_disagreements.jsonl"))
    args = p.parse_args(argv)
    summary, records = collect(
        args.candidate, args.games, args.seed, args.max_records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    report = args.output.with_suffix(".summary.json")
    report.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"dataset": str(args.output),
                      "summary_path": str(report), **summary}, indent=2))


if __name__ == "__main__":
    main()
