"""Phase 10.5.8: validate clone speed and holdout-64 on fresh states.

All results are exploratory. No automatic champion promotion.
"""
import argparse
import copy
import json
import time
from pathlib import Path

from .core import Game
from .heuristic_v2 import heuristic_v2
from .phase1057 import analyze, collect_states


def clone_benchmark(seed=10580001, samples=1000):
    if samples < 1:
        raise ValueError("samples must be positive")
    game = Game(seed=seed)
    for _ in range(30):
        if game.done:
            break
        game.step(heuristic_v2(game))
    baseline = game.clone()
    expected = copy.deepcopy(game)
    assert baseline.rng.getstate() == expected.rng.getstate()
    assert baseline.scores() == expected.scores()
    for factory in (copy.deepcopy, lambda value: value.clone()):
        for _ in range(10):
            factory(game)
    timings = {}
    for name, factory in (("deepcopy", copy.deepcopy),
                          ("clone", lambda value: value.clone())):
        start = time.perf_counter()
        for _ in range(samples):
            factory(game)
        timings[name] = (time.perf_counter() - start) / samples
    timings["speedup"] = (
        timings["deepcopy"] / timings["clone"]
        if timings["clone"] else None)
    return timings


def run(checkpoint, *, games=150, limit=500, seed=10580001,
        audit_futures=32):
    states, encountered = collect_states(
        checkpoint, games=games, seed=seed, limit=limit)
    result = analyze(
        states, seed=seed, validation_futures=(32, 64),
        audit_futures=audit_futures)
    return {
        "phase": "10.5.8", "checkpoint": checkpoint, "seed": seed,
        "games": games, "encountered_disagreements": encountered,
        "clone_benchmark": clone_benchmark(seed),
        "diagnostics": result, "promoted": False,
        "note": "Same-state agreement is not tournament performance. "
                "Use fresh paired tournament seeds before qualification.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True)
    p.add_argument("--games", type=int, default=150)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--seed", type=int, default=10580001)
    p.add_argument("--audit-futures", type=int, default=32)
    p.add_argument("--output", type=Path, default=Path(
        "runs/phase105/phase1058_diagnostics.json"))
    args = p.parse_args(argv)
    result = run(args.candidate, games=args.games, limit=args.limit,
                 seed=args.seed, audit_futures=args.audit_futures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    summary = dict(result)
    summary["diagnostics"] = {
        k: v for k, v in result["diagnostics"].items() if k != "records"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
