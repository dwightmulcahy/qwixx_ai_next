"""Phase 10.2: frozen-checkpoint disagreement and counterfactual audit.

Sample real challenger trajectories, compare against frozen Phase 9.1, and
estimate alternate outcomes with common seeded V2 continuations. Exploratory
only: this does not train or promote any policy.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from .champion import champion
from .core import COLORS, Game, ORDER
from .heuristic_v2 import heuristic_v2
from .model import choose, load
from .phase9_search import evaluate_actions


def describe(action):
    if action is None:
        return {"kind": "pass", "color": None, "number": None, "lock": False}
    color, number, lock = action
    return {"kind": "lock" if lock else "mark", "color": COLORS[color],
            "number": number, "lock": lock}


def category(game, candidate, reference):
    if candidate is None:
        return "neural_pass"
    if reference is None:
        return "champion_pass"
    if candidate[2] != reference[2]:
        return "lock_disagreement"
    board = game.boards[game.actor]
    def skipped(action):
        color, number, _ = action
        order = ORDER[color]
        row = board.rows[color]
        previous = order.index(row[-1]) if row else -1
        return order.index(number) - previous - 1
    if skipped(candidate) > skipped(reference):
        return "neural_skips_more"
    if skipped(candidate) < skipped(reference):
        return "champion_skips_more"
    return "different_mark"


def audit(checkpoint, games=8, seed=1601000, futures=8, max_samples=30,
          audit_seed=1701000):
    """Audit decisions for the challenger in each seat against three V2 players.

    Every disagreement is counted; only the first max_samples are subjected
    to costly counterfactual rollouts. These are *not* random samples and
    should not be interpreted as unbiased estimates of overall advantage.
    """
    if games < 1 or futures < 1 or max_samples < 0:
        raise ValueError("games and futures must be positive; max_samples >= 0")
    model = load(checkpoint)
    counts = Counter({"decisions": 0, "disagreements": 0})
    by_seat = {str(i + 1): Counter({"decisions": 0, "disagreements": 0})
               for i in range(4)}
    samples = []
    for index in range(games):
        for seat in range(4):
            game = Game(n=4, seed=seed + index)
            while not game.done:
                if game.actor != seat:
                    game.step(heuristic_v2(game))
                    continue
                counts["decisions"] += 1
                by_seat[str(seat + 1)]["decisions"] += 1
                neural = choose(model, game, epsilon=0)
                reference = champion(game)
                if neural != reference:
                    counts["disagreements"] += 1
                    by_seat[str(seat + 1)]["disagreements"] += 1
                    label = category(game, neural, reference)
                    counts[label] += 1
                    if len(samples) < max_samples:
                        estimates = evaluate_actions(
                            game, futures=futures, seed=audit_seed)
                        neural_est = next(x for x in estimates if x["action"] == neural)
                        champion_est = next(x for x in estimates if x["action"] == reference)
                        samples.append({
                            "game_seed": seed + index,
                            "seat": seat + 1,
                            "turn": game.turn + 1,
                            "phase": "colored" if game.phase == game.n else "white",
                            "active_seat": game.active + 1,
                            "dice": list(game.dice),
                            "locked_colors": [COLORS[c] for c in sorted(game.locks)],
                            "scores_before": game.scores(),
                            "marks_by_color": [list(row) for row in game.boards[seat].rows],
                            "penalties_before": game.boards[seat].penalties,
                            "legal_actions": [describe(a) for a in game.legal()],
                            "neural_action": describe(neural),
                            "champion_action": describe(reference),
                            "category": label,
                            "neural_estimate": {
                                "win_share": neural_est["win_share"],
                                "mean_score": neural_est["mean_score"]},
                            "champion_estimate": {
                                "win_share": champion_est["win_share"],
                                "mean_score": champion_est["mean_score"]},
                            "estimated_win_share_delta":
                                neural_est["win_share"] - champion_est["win_share"],
                            "estimated_score_delta":
                                neural_est["mean_score"] - champion_est["mean_score"],
                        })
                game.step(neural)
    result = {
        "phase": "10.2",
        "checkpoint": str(checkpoint),
        "games_per_seat": games,
        "seed": seed,
        "futures_per_action": futures,
        "audit_seed": audit_seed,
        "counts": dict(counts),
        "disagreement_rate": counts["disagreements"] / counts["decisions"]
        if counts["decisions"] else 0.0,
        "by_seat": {
            seat: {**dict(values),
                   "disagreement_rate": values["disagreements"] / values["decisions"]
                   if values["decisions"] else 0.0}
            for seat, values in by_seat.items()},
        "audited_samples": len(samples),
        "samples": samples,
        "note": "Exploratory, first-N disagreement samples only. V2 continuations "
                "use shared future seeds but may diverge after actions. "
                "No training, deployment, or promotion.",
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--games", type=int, default=8,
                        help="Challenger games in each of the four seats")
    parser.add_argument("--seed", type=int, default=1601000)
    parser.add_argument("--futures", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--audit-seed", type=int, default=1701000)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/phase10_disagreements.json"))
    args = parser.parse_args(argv)
    if not args.candidate.is_file():
        parser.error(f"checkpoint does not exist: {args.candidate}")
    if args.games < 1 or args.futures < 1 or args.max_samples < 0:
        parser.error("games/futures must be positive; max-samples nonnegative")
    result = audit(args.candidate, args.games, args.seed, args.futures,
                   args.max_samples, args.audit_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in result.items() if key != "samples"}
    summary["output"] = str(args.output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
