"""Phase 12.1: replay and diagnose Phase 11.9 row-choice disagreements.

Replays deterministic heuristic-V2 trajectories; refuses mismatched states.
No model training, rollout labels, or promotion.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from .core import Game, ORDER
from .env import decode
from .heuristic_v2 import heuristic_v2


def action_details(game, action_id):
    action = decode(action_id)
    if action is None:
        return {"pass": True}
    color, number, lock = action
    board = game.boards[game.actor]
    row = board.rows[color]
    index = ORDER[color].index(number)
    previous = ORDER[color].index(row[-1]) if row else -1
    return {
        "color": color, "number": number, "lock": lock,
        "existing_marks": len(row), "skipped_cells": index - previous - 1,
        "remaining_after": len(ORDER[color]) - index - 1,
        "immediate_points": len(row) + 1 + (len(row) + 1 if lock else 0),
    }


def snapshot(game):
    return {
        "actor": game.actor, "turn": game.turn, "phase": game.phase,
        "active": game.active, "dice": list(game.dice),
        "locks": sorted(game.locks), "scores": game.scores(),
        "penalties": [b.penalties for b in game.boards],
        "rows": [[list(row) for row in b.rows] for b in game.boards],
        "legal_actions": sorted(
            __import__("qwixx_ai.env", fromlist=["encode"]).encode(a)
            for a in game.legal()
        ),
    }


def replay_decisions(report):
    """Replay the original V2 games and require exact decision identity."""
    from .env import encode
    by_seed = defaultdict(list)
    for row in report["decisions"]:
        by_seed[row["game_seed"]].append(row)
    reconstructed = []
    for seed, records in sorted(by_seed.items()):
        game = Game(n=4, seed=seed)
        pending = {(r["turn"], r["phase"], r["actor"]): r for r in records}
        if len(pending) != len(records):
            raise ValueError(f"duplicate decision identity in game {seed}")
        while not game.done:
            key = (game.turn, game.phase, game.actor)
            if key in pending:
                row = pending.pop(key)
                legal = {0, *(encode(a) for a in game.legal())}
                if row["source_action"] not in legal or row["champion_action"] not in legal:
                    raise ValueError(f"illegal recorded action at {seed}:{key}")
                source = action_details(game, row["source_action"])
                champion = action_details(game, row["champion_action"])
                if source.get("color") == champion.get("color"):
                    raise ValueError(f"not a row-choice disagreement: {seed}:{key}")
                reconstructed.append({
                    **row, "state": snapshot(game),
                    "source_move": source, "champion_move": champion,
                    "win_gap": row["paired_win_share"]["mean"],
                    "score_gap": row["paired_score"]["mean"],
                    "skip_difference": champion["skipped_cells"] - source["skipped_cells"],
                    "remaining_difference": champion["remaining_after"] - source["remaining_after"],
                })
            game.step(heuristic_v2(game))
        if pending:
            raise ValueError(f"unmatched decision positions in game {seed}: {list(pending)}")
    if len(reconstructed) != len(report["decisions"]):
        raise ValueError("replay count mismatch")
    return reconstructed


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups["all"].append(row)
        groups[row["stage"]].append(row)
        groups["champion_row_" + str(row["champion_move"]["color"])].append(row)
        groups["source_row_" + str(row["source_move"]["color"])].append(row)
    return {
        key: {
            "states": len(rows),
            "games": len({r["game_seed"] for r in rows}),
            "mean_win_gap": mean(r["win_gap"] for r in rows),
            "mean_score_gap": mean(r["score_gap"] for r in rows),
            "champion_ci_positive": sum(r["champion_preferred"] for r in rows),
            "source_ci_negative": sum(r["source_preferred"] for r in rows),
            "mean_skip_difference": mean(r["skip_difference"] for r in rows),
            "mean_remaining_difference": mean(r["remaining_difference"] for r in rows),
        } for key, rows in sorted(groups.items())
    }


def analyze(report):
    if report.get("phase") != "11.9":
        raise ValueError("expected Phase 11.9 report")
    records = replay_decisions(report)
    return {
        "phase": "12.1", "source_phase": "11.9",
        "sampled_states": len(records),
        "summary": summarize(records),
        "decisions": sorted(records, key=lambda r: -abs(r["win_gap"])),
        "promoted": False,
        "note": "Exploratory V2 continuation, selected disagreements; no optimal labels.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("runs/phase11/phase119_confirmation.json"))
    p.add_argument("--output", type=Path, default=Path("runs/phase12/phase121_diagnostics.json"))
    args = p.parse_args(argv)
    report = json.loads(args.input.read_text())
    result = analyze(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "decisions"}, indent=2))
    print(f"Detailed decisions: {args.output}")


if __name__ == "__main__":
    main()
