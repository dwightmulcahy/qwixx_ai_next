"""Phase 9.2: audit actual Phase 9.1 overrides on held-out rollout seeds.

The selection estimates use search_seed. The independent audit uses audit_seed,
which must differ. Only actual overrides are audited, avoiding selection bias
from reusing the same futures. Exploratory, not a policy promotion.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from .core import Game
from .heuristic_v2 import heuristic_v2
from .phase9_search import evaluate_actions, outcome, state_seed
from .phase91_search import decide, eligible


def audit(games=100, seed=993000, futures=32, audit_futures=128,
          search_seed=900000, audit_seed=901000, min_gain=0.10,
          min_score_gain=0.0, start_turn=12, end_turn=35,
          output="runs/phase92/overrides.jsonl"):
    if min(games, futures, audit_futures) < 1:
        raise ValueError("sample sizes must be positive")
    if search_seed == audit_seed:
        raise ValueError("audit_seed must differ from search_seed")
    stats = {"decisions": 0, "eligible": 0, "overrides": 0}
    records = []
    for game_id in range(games):
        game = Game(seed=seed + game_id)
        focal = game_id % game.n
        while not game.done:
            if game.actor != focal:
                action = heuristic_v2(game)
            else:
                stats["decisions"] += 1
                teacher = heuristic_v2(game)
                if eligible(game, teacher, start_turn, end_turn):
                    stats["eligible"] += 1
                action, reason = decide(
                    game, futures, search_seed, min_gain,
                    min_score_gain, start_turn, end_turn,
                )
                if reason == "override":
                    stats["overrides"] += 1
                    estimates = evaluate_actions(
                        game, futures=futures, seed=search_seed,
                    )
                    selected = next(
                        item for item in estimates if item["action"] == action
                    )
                    baseline = next(
                        item for item in estimates if item["action"] == teacher
                    )
                    rng = random.Random(state_seed(game, audit_seed))
                    differences = []
                    for _ in range(audit_futures):
                        future = rng.randrange(2**64)
                        candidate = outcome(game, action, future)
                        reference = outcome(game, teacher, future)
                        differences.append((
                            candidate[0] - reference[0],
                            candidate[1] - reference[1],
                        ))
                    mean = np.mean(differences, axis=0)
                    board = game.boards[focal]
                    records.append({
                        "game_id": game_id, "seat": focal,
                        "turn": game.turn, "phase": game.phase,
                        "active": game.active, "dice": list(game.dice),
                        "locks": sorted(game.locks),
                        "rows": [list(row) for row in board.rows],
                        "penalties": board.penalties,
                        "legal_actions": [list(a) for a in game.legal()],
                        "v2_action": list(teacher) if teacher else None,
                        "search_action": list(action) if action else None,
                        "predicted_win_share_delta": (
                            selected["win_share"] - baseline["win_share"]
                        ),
                        "predicted_score_delta": (
                            selected["mean_score"] - baseline["mean_score"]
                        ),
                        "audit_win_share_delta": float(mean[0]),
                        "audit_score_delta": float(mean[1]),
                    })
            game.step(action)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    summary = {
        "phase": "9.2", "games": games, "seed": seed,
        "selection_futures": futures, "audit_futures": audit_futures,
        "search_seed": search_seed, "audit_seed": audit_seed,
        "decisions": stats, "override_rate": (
            stats["overrides"] / stats["decisions"]
            if stats["decisions"] else 0.0
        ),
        "records": str(path),
        "warning": "Exploratory. Held-out seeds prevent reuse of selection "
                   "futures but action-dependent RNG trajectories can diverge.",
    }
    if records:
        for metric in ("win_share", "score"):
            summary[f"mean_predicted_{metric}_delta"] = float(np.mean([
                r[f"predicted_{metric}_delta"] for r in records
            ]))
            summary[f"mean_audit_{metric}_delta"] = float(np.mean([
                r[f"audit_{metric}_delta"] for r in records
            ]))
        summary["positive_audit_win_share_fraction"] = sum(
            r["audit_win_share_delta"] > 0 for r in records
        ) / len(records)
    else:
        summary["note"] = "No overrides occurred; no audit outcomes available."
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=993000)
    parser.add_argument("--futures", type=int, default=32)
    parser.add_argument("--audit-futures", type=int, default=128)
    parser.add_argument("--search-seed", type=int, default=900000)
    parser.add_argument("--audit-seed", type=int, default=901000)
    parser.add_argument("--min-gain", type=float, default=0.10)
    parser.add_argument("--min-score-gain", type=float, default=0.0)
    parser.add_argument("--start-turn", type=int, default=12)
    parser.add_argument("--end-turn", type=int, default=35)
    parser.add_argument("--output", default="runs/phase92/overrides.jsonl")
    args = parser.parse_args()
    print(json.dumps(audit(
        games=args.games, seed=args.seed, futures=args.futures,
        audit_futures=args.audit_futures,
        search_seed=args.search_seed, audit_seed=args.audit_seed,
        min_gain=args.min_gain, min_score_gain=args.min_score_gain,
        start_turn=args.start_turn, end_turn=args.end_turn,
        output=args.output,
    ), indent=2))


if __name__ == "__main__":
    main()
