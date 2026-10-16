"""Phase 10.5.7: same-state verifier diagnostics and rollout profiling.

Collect disagreements from V2 games, then evaluate both verifiers on the
*same* states. This is diagnostic, not a policy-strength qualification.
"""
import argparse
import cProfile
import copy
import io
import json
import pstats
import random
import time
from pathlib import Path
from statistics import mean

from .champion import champion
from .core import Game
from .env import encode
from .heuristic_v2 import heuristic_v2
from .model import choose, load
from .phase1055 import evidence as holdout_evidence
from .phase105_verified import paired_evidence
from .phase9_search import outcome, state_seed


def collect_states(checkpoint, games=30, seed=10570001, limit=100,
                   seats=(2, 3)):
    """Use V2 game trajectories; store snapshots only at neural disagreements."""
    if games < 1 or limit < 1 or not seats:
        raise ValueError("games and limit must be positive; seats required")
    model = load(checkpoint)
    states = []
    encountered = 0
    for index in range(games):
        game = Game(seed=seed + index)
        while not game.done:
            if game.actor + 1 in seats:
                baseline = choose(model, game, epsilon=0)
                alternative = champion(game)
                if baseline != alternative:
                    encountered += 1
                    # Reservoir sampling prevents favoring early-game states.
                    rng = random.Random(seed + 700001 + encountered)
                    if len(states) < limit:
                        states.append((copy.deepcopy(game), baseline,
                                       alternative, index))
                    else:
                        slot = rng.randrange(encountered)
                        if slot < limit:
                            states[slot] = (copy.deepcopy(game), baseline,
                                            alternative, index)
            game.step(heuristic_v2(game))
    return states, encountered


def analyze(states, *, seed=10570001, max_futures=64,
            initial_futures=16, batch_size=16, min_gain=0.05,
            validation_futures=(16, 32, 64), confidence=0.95,
            audit_futures=128):
    if audit_futures < 2 or max_futures < 2:
        raise ValueError("futures must be >= 2")
    rows = []
    timings = {"fixed": 0.0}
    for v in validation_futures:
        if v < 2:
            raise ValueError("validation futures must be >= 2")
        timings[str(v)] = 0.0
    for game, baseline, alternative, game_index in states:
        # Same deterministic rollout stream for every method.
        start = time.perf_counter()
        fixed = paired_evidence(game, baseline, alternative,
                                futures=max_futures, seed=seed,
                                confidence=confidence)
        timings["fixed"] += time.perf_counter() - start
        fixed_override = (fixed["win_ci_low"] > min_gain
                          and fixed["score_gain"] >= 0)
        variants = {}
        for v in validation_futures:
            start = time.perf_counter()
            result = holdout_evidence(
                game, baseline, alternative,
                initial_futures=initial_futures,
                max_futures=max_futures, batch_size=batch_size,
                seed=seed, confidence=confidence,
                min_gain=min_gain, method="holdout",
                validation_futures=v)
            timings[str(v)] += time.perf_counter() - start
            variants[str(v)] = result
        # Disjoint audit futures: never used to select the action.
        rng = random.Random(state_seed(game, seed + 500000))
        win_deltas, score_deltas = [], []
        for _ in range(audit_futures):
            future = rng.randrange(2**64)
            aw, a_score = outcome(game, alternative, future)
            bw, b_score = outcome(game, baseline, future)
            win_deltas.append(aw - bw)
            score_deltas.append(a_score - b_score)
        rows.append({
            "source_game": game_index, "turn": game.turn,
            "seat": game.actor + 1, "phase": game.phase,
            "dice": list(game.dice),
            "neural_action": encode(baseline),
            "champion_action": encode(alternative),
            "fixed_override": fixed_override,
            "fixed_evidence": fixed,
            "holdout": variants,
            "audit_win_gain": mean(win_deltas),
            "audit_score_gain": mean(score_deltas),
        })
    count = len(rows)
    variants_summary = {}
    for v in validation_futures:
        key = str(v)
        changed = [row for row in rows
                   if row["holdout"][key]["override"]
                   != row["fixed_override"]]
        overrides = [row for row in rows if row["holdout"][key]["override"]]
        variants_summary[key] = {
            "overrides": len(overrides),
            "disagreements_with_fixed": len(changed),
            "fixed_only": sum(row["fixed_override"]
                              and not row["holdout"][key]["override"]
                              for row in rows),
            "holdout_only": sum(not row["fixed_override"]
                                and row["holdout"][key]["override"]
                                for row in rows),
            "mean_futures": (
                mean(row["holdout"][key]["futures"] for row in rows)
                if count else 0.0),
            "mean_seconds": timings[key] / count if count else 0.0,
            "audit_mean_gain_on_overrides": (
                mean(row["audit_win_gain"] for row in overrides)
                if overrides else None),
        }
    return {
        "states": count, "fixed_overrides": sum(
            row["fixed_override"] for row in rows),
        "fixed_mean_seconds": timings["fixed"] / count if count else 0.0,
        "validation_sensitivity": variants_summary,
        "records": rows,
        "note": "Audit rollouts use disjoint RNG streams, but simulated V2 "
                "continuations are not actual policy outcomes. "
                "The same-state dataset is not a tournament.",
    }


def profile_rollouts(states, futures=32, top=20):
    if futures < 1 or top < 1:
        raise ValueError("futures and top must be positive")
    profiler = cProfile.Profile()
    profiler.enable()
    for game, baseline, alternative, _ in states:
        rng = random.Random(state_seed(game, 10579999))
        for _ in range(futures):
            future = rng.randrange(2**64)
            outcome(game, baseline, future)
            outcome(game, alternative, future)
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats(
        "cumulative").print_stats(top)
    return stream.getvalue()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True)
    p.add_argument("--games", type=int, default=30)
    p.add_argument("--seed", type=int, default=10570001)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--audit-futures", type=int, default=128)
    p.add_argument("--profile-states", type=int, default=5)
    p.add_argument("--profile-futures", type=int, default=16)
    p.add_argument("--output", type=Path, default=Path(
        "runs/phase105/phase1057_diagnostics.json"))
    args = p.parse_args(argv)
    states, encountered = collect_states(
        args.candidate, games=args.games, seed=args.seed, limit=args.limit)
    result = analyze(states, seed=args.seed,
                     audit_futures=args.audit_futures)
    result.update({"phase": "10.5.7", "checkpoint": args.candidate,
                   "seed": args.seed, "games": args.games,
                   "encountered_disagreements": encountered,
                   "promoted": False})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    profile_path = args.output.with_suffix(".profile.txt")
    profile_path.write_text(profile_rollouts(
        states[:args.profile_states], futures=args.profile_futures))
    print(json.dumps({k: v for k, v in result.items()
                      if k != "records"}, indent=2))
    print(f"Profile: {profile_path}")


if __name__ == "__main__":
    main()
