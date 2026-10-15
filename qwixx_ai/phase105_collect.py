"""Phase 10.5: collect disagreement states with paired V2 rollout evidence.

No deployed policy or source checkpoint is modified.
"""
import argparse
import copy
import json
import random
from pathlib import Path

from .champion import champion
from .core import Game
from .env import encode, observe
from .heuristic_v2 import heuristic_v2
from .model import choose, load
from .phase9_search import evaluate_actions


def collect(checkpoint, games=30, seed=2401000, samples=200,
            futures=96, rollout_seed=2501000, sample_seed=2601000):
    if games < 1 or samples < 1 or futures < 2:
        raise ValueError("games/samples must be positive and futures >= 2")
    model = load(checkpoint)
    rng = random.Random(sample_seed)
    reservoir = []
    seen = 0
    weighted_seen = 0
    for i in range(games):
        for seat in range(4):
            game = Game(seed=seed+i)
            while not game.done:
                if game.actor != seat:
                    game.step(heuristic_v2(game))
                    continue
                neural = choose(model, game, epsilon=0)
                reference = champion(game)
                if neural != reference:
                    seen += 1
                    # Prioritize seats 2/3 without excluding other seats.
                    weight = 2 if seat in (1, 2) else 1
                    weighted_seen += weight
                    # Weighted reservoir: retain one entry per position.
                    # Higher priority for seats 2/3, without duplicating states.
                    key = rng.random() ** (1.0 / weight)
                    item = (copy.deepcopy(game), seat, neural, reference)
                    if len(reservoir) < samples:
                        reservoir.append((key, item))
                    else:
                        worst = min(range(len(reservoir)),
                                    key=lambda j: reservoir[j][0])
                        if key > reservoir[worst][0]:
                            reservoir[worst] = (key, item)
                game.step(neural)
    records = []
    for index, (_, (game, seat, neural, reference)) in enumerate(reservoir):
        estimates = evaluate_actions(game, futures=futures,
                                     seed=rollout_seed+index)
        by_action = {encode(item["action"]): item for item in estimates}
        baseline = by_action[encode(neural)]
        best = max(estimates, key=lambda item: item["win_share"])
        records.append({
            "observation": observe(game).tolist(),
            "seat": seat+1,
            "turn": game.turn,
            "neural_action": encode(neural),
            "champion_action": encode(reference),
            "best_action": encode(best["action"]),
            "estimated_gain": best["win_share"]-baseline["win_share"],
            "estimates": [
                {"action": encode(item["action"]),
                 "win_share": item["win_share"],
                 "mean_score": item["mean_score"]}
                for item in estimates],
        })
    return {"phase": "10.5", "source": str(checkpoint),
            "games_per_seat": games, "disagreements_seen": seen,
            "futures": futures, "samples": records,
            "note": "Exploratory paired-seed V2 continuations; estimates are noisy."}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--games", type=int, default=30)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--futures", type=int, default=96)
    p.add_argument("--seed", type=int, default=2401000)
    p.add_argument("--output", type=Path,
                   default=Path("runs/phase105/disagreements.json"))
    args = p.parse_args(argv)
    if not args.candidate.is_file():
        p.error("candidate checkpoint not found")
    result = collect(args.candidate, args.games, args.seed,
                     args.samples, args.futures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result)+"\n")
    print(json.dumps({k: len(v) if k == "samples" else v
                      for k, v in result.items() if k != "note"}, indent=2))


if __name__ == "__main__":
    main()
