"""Phase 11.7: multi-seed paired validation and late-game disagreement diagnostics.

No training, checkpoint modification, or automatic promotion. Seed blocks are
independent; decisions are descriptive and do not establish optimal actions.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import NormalDist

from .champion import champion
from .core import Game
from .env import encode
from .heuristic_v2 import heuristic_v2
from .model import choose, load
from .phase112 import relative_stage
from .tournament import play


def summarize_blocks(blocks, confidence=0.95):
    """Confidence interval over independently seeded, seat-averaged blocks."""
    if len(blocks) < 2 or not 0 < confidence < 1:
        raise ValueError("at least two blocks and valid confidence required")
    mean = sum(blocks) / len(blocks)
    variance = sum((x - mean) ** 2 for x in blocks) / (len(blocks) - 1)
    margin = NormalDist().inv_cdf((1 + confidence) / 2) * math.sqrt(
        variance / len(blocks))
    return {"paired_advantage": mean, "advantage_ci_low": mean - margin,
            "advantage_ci_high": mean + margin, "confidence": confidence,
            "independent_blocks": len(blocks)}


def compare(challenger, reference, seeds, rounds, confidence=0.95):
    """Pair by dice seed and focal seat, with independent seed blocks."""
    if rounds < 2 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("rounds >= 2 and distinct seeds required")
    blocks = []
    per_seed = []
    total_candidate = total_reference = 0.0
    total_candidate_score = total_reference_score = 0.0
    for seed in seeds:
        seed_blocks = []
        for offset in range(rounds):
            seat_differences = []
            for seat in range(4):
                def trial(policy):
                    lineup = [heuristic_v2] * 3
                    lineup.insert(seat, policy)
                    scores, shares = play(lineup, seed + offset)
                    return scores[seat], shares[seat]
                candidate_score, candidate_share = trial(challenger)
                reference_score, reference_share = trial(reference)
                total_candidate += candidate_share
                total_reference += reference_share
                total_candidate_score += candidate_score
                total_reference_score += reference_score
                seat_differences.append(candidate_share - reference_share)
            seed_blocks.append(sum(seat_differences) / 4)
        blocks.extend(seed_blocks)
        per_seed.append({"seed": seed, "paired_advantage":
                         sum(seed_blocks) / len(seed_blocks)})
    result = summarize_blocks(blocks, confidence)
    n = len(blocks) * 4
    result.update({
        "rounds_per_seed": rounds, "seed_blocks": per_seed,
        "games_per_policy": n,
        "challenger_win_share": total_candidate / n,
        "reference_win_share": total_reference / n,
        "challenger_mean_score": total_candidate_score / n,
        "reference_mean_score": total_reference_score / n,
    })
    return result


def late_game_diagnostics(candidate, source, games=60, seed=11071001,
                          max_examples=20):
    """Measure policy disagreement on V2 trajectories, including late states."""
    if games < 1 or max_examples < 0:
        raise ValueError("invalid diagnostics settings")
    counts = Counter()
    examples = []
    for game_id in range(games):
        game = Game(n=4, seed=seed + game_id)
        decisions = []
        while not game.done:
            if game.actor in (1, 2):
                decisions.append((
                    game.turn, game.phase, game.actor,
                    encode(choose(candidate, game, epsilon=0)),
                    encode(choose(source, game, epsilon=0)),
                    encode(champion(game)),
                ))
            game.step(heuristic_v2(game))
        for turn, phase, actor, candidate_action, source_action, champion_action in decisions:
            stage = relative_stage(turn, game.turn)
            counts[f"{stage}_total"] += 1
            if candidate_action != source_action:
                counts[f"{stage}_vs_source"] += 1
            if candidate_action != champion_action:
                counts[f"{stage}_vs_champion"] += 1
            if (stage == "late" and candidate_action != source_action
                    and len(examples) < max_examples):
                examples.append({
                    "game_seed": seed + game_id, "turn": turn,
                    "phase": phase, "actor": actor,
                    "candidate_action": candidate_action,
                    "source_action": source_action,
                    "champion_action": champion_action,
                })
    return {
        "games": games, "seed": seed,
        "by_stage": {
            stage: {
                "decisions": counts[f"{stage}_total"],
                "candidate_source_disagreements": counts[f"{stage}_vs_source"],
                "candidate_champion_disagreements": counts[f"{stage}_vs_champion"],
                "candidate_source_disagreement_rate": (
                    counts[f"{stage}_vs_source"] / counts[f"{stage}_total"]
                    if counts[f"{stage}_total"] else 0),
            }
            for stage in ("early", "middle", "late")
        },
        "late_examples": examples,
        "note": "Action differences alone do not establish which policy is better.",
    }


def evaluate(candidate_path, source_path, rounds=250,
             seeds=(11070001, 11080001, 11090001), diagnostic_games=60):
    if rounds < 2 or len(seeds) < 2:
        raise ValueError("at least two rounds and two seed blocks required")
    # Non-overlapping seed ranges: also prevent overlap across comparisons.
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed blocks must be distinct")
    if any(abs(a - b) < rounds for i, a in enumerate(seeds)
           for b in seeds[i + 1:]):
        raise ValueError("seed blocks overlap")
    candidate = load(candidate_path)
    source = load(source_path)
    policies = {
        "candidate": lambda g: choose(candidate, g, epsilon=0),
        "source": lambda g: choose(source, g, epsilon=0),
        "champion": champion,
        "v2": heuristic_v2,
    }
    results = {}
    for index, opponent in enumerate(("source", "champion", "v2")):
        comparison_seeds = [s + index * 1000000 for s in seeds]
        results[f"versus_{opponent}"] = compare(
            policies["candidate"], policies[opponent], comparison_seeds, rounds)
    eligible = (
        all(results[f"versus_{name}"]["advantage_ci_low"] > 0
            for name in ("source", "champion", "v2"))
        and results["versus_champion"]["challenger_mean_score"]
        >= results["versus_champion"]["reference_mean_score"]
    )
    return {
        "phase": "11.7", "candidate": str(candidate_path),
        "source": str(source_path), "rounds_per_seed": rounds,
        "base_seeds": list(seeds), "versus_source": results["versus_source"],
        "versus_champion": results["versus_champion"],
        "versus_v2": results["versus_v2"],
        "late_game": late_game_diagnostics(
            candidate, source, games=diagnostic_games),
        "eligible_for_review": eligible, "promoted": False,
        "note": "Held-out paired evaluation; descriptive diagnostics; no promotion.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=250)
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[11070001, 11080001, 11090001])
    p.add_argument("--diagnostic-games", type=int, default=60)
    p.add_argument("--output", type=Path, default=Path(
        "runs/phase11/phase117_evaluation.json"))
    args = p.parse_args(argv)
    for path in (args.candidate, args.source):
        if not path.is_file():
            p.error(f"checkpoint does not exist: {path}")
    if args.diagnostic_games < 1:
        p.error("--diagnostic-games must be positive")
    result = evaluate(args.candidate, args.source, args.rounds,
                      args.seeds, args.diagnostic_games)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "late_game"},
                     indent=2))
    print(f"Full diagnostics: {args.output}")


if __name__ == "__main__":
    main()
