"""Phase 12.6: tune row-progress bonuses on fresh full-game seeds.

The challenger uses adjusted neural Q-values, not forced marks. All opponents
use frozen heuristic V2. Tuning and evaluation seeds are disjoint. Diagnostic
only; no model weights are changed and no policy is promoted.
"""
import argparse
import json
from pathlib import Path
from statistics import mean

import torch

from .core import Game
from .env import decode, mask, observe
from .heuristic_v2 import heuristic_v2
from .model import load


def adjusted_choice(model, game, bonus):
    """Score every legal action, including pass, using a progress-only bonus."""
    if bonus < 0:
        raise ValueError("bonus must be nonnegative")
    valid = mask(game)
    with torch.no_grad():
        q = model(torch.from_numpy(observe(game))).detach().cpu().numpy()
    best_action = None
    best_value = float("-inf")
    for index in range(len(q)):
        if not valid[index]:
            continue
        action = decode(index)
        progress = len(game.boards[game.actor].rows[action[0]]) if action else 0
        value = float(q[index]) + bonus * progress
        if value > best_value:
            best_value, best_action = value, action
    return best_action


def play(model, bonus, seed, seat):
    game = Game(n=4, seed=seed)
    overrides = 0
    opportunities = 0
    while not game.done:
        if game.actor == seat:
            baseline = adjusted_choice(model, game, 0.0)
            action = adjusted_choice(model, game, bonus)
            overrides += action != baseline
            opportunities += 1
        else:
            action = heuristic_v2(game)
        game.step(action)
    scores = game.scores()
    best = max(scores)
    share = (1 / sum(score == best for score in scores)
             if scores[seat] == best else 0.0)
    return {"win_share": share, "score": scores[seat],
            "score_margin": scores[seat] - max(s for i, s in enumerate(scores) if i != seat),
            "overrides": overrides, "decisions": opportunities}


def evaluate(model, bonuses, games, seed):
    """Each candidate receives identical game seeds and seat assignments."""
    if games < 1:
        raise ValueError("games must be positive")
    rows = []
    for game_index in range(games):
        game_seed = seed + game_index
        seat = game_index % 4
        for bonus in bonuses:
            result = play(model, bonus, game_seed, seat)
            rows.append({"seed": game_seed, "seat": seat, "bonus": bonus, **result})
    grouped = {}
    for bonus in bonuses:
        subset = [row for row in rows if row["bonus"] == bonus]
        grouped[str(bonus)] = {
            "games": len(subset),
            "win_share": mean(row["win_share"] for row in subset),
            "mean_score": mean(row["score"] for row in subset),
            "mean_score_margin": mean(row["score_margin"] for row in subset),
            "override_rate": sum(row["overrides"] for row in subset) /
                             max(1, sum(row["decisions"] for row in subset)),
        }
    return grouped, rows


def run(checkpoint, bonuses=(0.0, 0.05, 0.1, 0.2, 0.4),
        tune_games=80, eval_games=160, tune_seed=12060001,
        eval_seed=12070001):
    if len(set(bonuses)) != len(bonuses) or 0.0 not in bonuses or any(b < 0 for b in bonuses):
        raise ValueError("bonuses must be distinct, nonnegative and include 0")
    if tune_games < 1 or eval_games < 1:
        raise ValueError("game counts must be positive")
    if set(range(tune_seed, tune_seed + tune_games)) & set(range(eval_seed, eval_seed + eval_games)):
        raise ValueError("tuning and evaluation seeds must not overlap")
    model = load(checkpoint)
    tuning, tuning_rows = evaluate(model, bonuses, tune_games, tune_seed)
    # Select by win share; use mean score margin as deterministic tie-breaker.
    winner = max(bonuses, key=lambda b: (
        tuning[str(b)]["win_share"], tuning[str(b)]["mean_score_margin"], -b))
    evaluation, evaluation_rows = evaluate(model, (0.0, winner) if winner != 0.0 else (0.0,),
                                           eval_games, eval_seed)
    baseline = evaluation["0.0"]
    selected = evaluation[str(winner)]
    # Paired seed-level differences, preserving zero for the baseline candidate.
    base_rows = {r["seed"]: r for r in evaluation_rows if r["bonus"] == 0.0}
    selected_rows = {r["seed"]: r for r in evaluation_rows if r["bonus"] == winner}
    differences = [
        selected_rows[s]["win_share"] - base_rows[s]["win_share"]
        for s in sorted(base_rows)
    ]
    return {
        "phase": "12.6", "checkpoint": str(checkpoint),
        "opponents": "heuristic_v2", "tune_seed": tune_seed,
        "eval_seed": eval_seed, "tune_games": tune_games, "eval_games": eval_games,
        "bonuses": list(bonuses), "tuning": tuning,
        "selected_bonus": winner, "evaluation": evaluation,
        "heldout_win_share_delta": selected["win_share"] - baseline["win_share"],
        "heldout_mean_paired_win_share_delta": mean(differences),
        "heldout_paired_positive": sum(d > 0 for d in differences),
        "heldout_paired_negative": sum(d < 0 for d in differences),
        "heldout_paired_ties": sum(d == 0 for d in differences),
        "tuning_games_detail": tuning_rows, "evaluation_games_detail": evaluation_rows,
        "promoted": False,
        "note": "Fresh full-game evaluation versus V2, not a champion head-to-head. "
                "No inference of statistical significance or promotion from this exploratory run.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--bonuses", type=float, nargs="+", default=[0, 0.05, 0.1, 0.2, 0.4])
    p.add_argument("--tune-games", type=int, default=80)
    p.add_argument("--eval-games", type=int, default=160)
    p.add_argument("--tune-seed", type=int, default=12060001)
    p.add_argument("--eval-seed", type=int, default=12070001)
    p.add_argument("--output", type=Path, default=Path("runs/phase12/phase126_progress.json"))
    args = p.parse_args(argv)
    if not args.checkpoint.is_file():
        p.error(f"checkpoint does not exist: {args.checkpoint}")
    result = run(args.checkpoint, tuple(args.bonuses), args.tune_games,
                 args.eval_games, args.tune_seed, args.eval_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("tuning_games_detail", "evaluation_games_detail")}, indent=2))
    print(f"Detailed games: {args.output}")


if __name__ == "__main__":
    main()
