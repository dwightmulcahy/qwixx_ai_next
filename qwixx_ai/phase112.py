"""Phase 11.2: balanced disagreement collection and optional rollout labels.

Search targets estimate V2-continuation outcomes, not optimal Q-values.
No checkpoint changes or automatic promotion.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .champion import champion
from .core import Game
from .env import encode, mask, observe
from .heuristic_v2 import heuristic_v2
from .model import choose, load
from .phase111 import split_for_game
from .phase9_search import evaluate_actions


def stage(turn):
    return "early" if turn < 12 else "middle" if turn < 35 else "late"


def priority(seed, game_id, turn, phase, actor):
    payload = f"{seed}:{game_id}:{turn}:{phase}:{actor}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def balanced_select(records, limit):
    """Round-robin deterministic buckets, selecting across splits/stages/seats."""
    if limit < 1:
        raise ValueError("limit must be positive")
    buckets = defaultdict(list)
    for record in records:
        key = (record["split"], record["stage"], record["actor"])
        buckets[key].append(record)
    for group in buckets.values():
        group.sort(key=lambda r: r["priority"])
    keys = sorted(buckets)
    selected = []
    # Each nonempty bucket gets a turn before another gets its next example.
    while len(selected) < limit:
        progress = False
        for key in keys:
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop(0))
                progress = True
        if not progress:
            break
    return sorted(selected, key=lambda r: (
        r["game_id"], r["turn"], r["phase"], r["actor"]))


def collect(checkpoint, games=100, seed=11020001, max_records=2000,
            search_futures=0, search_limit=0):
    if games < 1 or max_records < 1 or search_futures < 0 or search_limit < 0:
        raise ValueError("invalid collection settings")
    model = load(checkpoint)
    candidates = []
    totals = Counter()
    by_split = Counter()
    for game_id in range(games):
        game = Game(n=4, seed=seed + game_id)
        split = split_for_game(seed, game_id)
        while not game.done:
            actor = game.actor
            if actor in (1, 2):
                neural = choose(model, game, epsilon=0)
                teacher = champion(game)
                totals["sampled_decisions"] += 1
                if neural != teacher:
                    totals["disagreements"] += 1
                    by_split[split] += 1
                    candidates.append({
                        "game_id": game_id, "game_seed": seed + game_id,
                        "split": split, "stage": stage(game.turn),
                        "turn": game.turn, "phase": game.phase,
                        "actor": actor,
                        "priority": priority(seed, game_id, game.turn,
                                             game.phase, actor),
                        "observation": observe(game).tolist(),
                        "legal_actions": [int(i) for i in mask(game).nonzero()[0]],
                        "neural_action": encode(neural),
                        "champion_action": encode(teacher),
                        "teacher_source": "phase91_champion",
                        "scores_before": game.scores(),
                        "penalties": [b.penalties for b in game.boards],
                        "locks": sorted(game.locks),
                    })
            game.step(heuristic_v2(game))
        totals["completed_games"] += 1
    selected = balanced_select(candidates, max_records)
    # Reconstruct exactly the V2 trajectory to recover states for search.
    # Search only selected examples and only when explicitly requested.
    search_keys = {
        (r["game_id"], r["turn"], r["phase"], r["actor"])
        for r in selected[:search_limit]
    } if search_futures > 0 else set()
    lookup = {(r["game_id"], r["turn"], r["phase"], r["actor"]): r
              for r in selected}
    if search_keys:
        for game_id in sorted({k[0] for k in search_keys}):
            game = Game(n=4, seed=seed + game_id)
            while not game.done:
                key = (game_id, game.turn, game.phase, game.actor)
                if key in search_keys:
                    record = lookup[key]
                    estimates = evaluate_actions(
                        game, futures=search_futures, seed=seed + 700000)
                    record["search_targets"] = [
                        {"action": encode(item["action"]),
                         "win_share": item["win_share"],
                         "mean_score": item["mean_score"]}
                        for item in estimates
                    ]
                    record["search_futures"] = search_futures
                    record["search_source"] = "v2_continuation_rollouts"
                game.step(heuristic_v2(game))
    for r in selected:
        r.pop("priority")
    saved = Counter(r["split"] for r in selected)
    stages = Counter(r["stage"] for r in selected)
    summary = {
        "phase": "11.2", "checkpoint": str(checkpoint),
        "seed": seed, "games": games,
        "sampled_decisions": totals["sampled_decisions"],
        "disagreements": totals["disagreements"],
        "disagreement_rate": (
            totals["disagreements"] / totals["sampled_decisions"]
            if totals["sampled_decisions"] else 0),
        "saved_records": len(selected),
        "saved_by_split": dict(saved),
        "saved_by_stage": dict(stages),
        "disagreements_by_split": dict(by_split),
        "search_labeled": sum("search_targets" in r for r in selected),
        "search_futures": search_futures,
        "max_records": max_records,
        "split_rule": "SHA256(seed:game_id), game-level 80/10/10",
        "sampling": "deterministic round-robin by split, stage, seat; "
                    "SHA256 priorities within buckets",
        "note": "Teacher actions are not guaranteed optimal. Search estimates "
                "use simulated V2 continuations. Train only on train split; "
                "do not tune on test. No promotion.",
        "promoted": False,
    }
    return summary, selected


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True)
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed", type=int, default=11020001)
    p.add_argument("--max-records", type=int, default=2000)
    p.add_argument("--search-futures", type=int, default=0)
    p.add_argument("--search-limit", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path(
        "runs/phase11/phase112_disagreements.jsonl"))
    args = p.parse_args(argv)
    if args.search_limit and not args.search_futures:
        p.error("--search-limit requires --search-futures")
    summary, records = collect(
        args.candidate, args.games, args.seed, args.max_records,
        args.search_futures, args.search_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    report = args.output.with_suffix(".summary.json")
    report.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"dataset": str(args.output),
                      "summary_path": str(report), **summary}, indent=2))


if __name__ == "__main__":
    main()
