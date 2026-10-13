"""Phase 6: generate reproducible, game-separated V2 imitation datasets."""
import argparse
import json
from pathlib import Path

import numpy as np

from .core import Game
from .env import encode, mask, observe
from .heuristic_v2 import heuristic_v2


def generate(games, seed=6000):
    if games < 2:
        raise ValueError("at least two games required for held-out validation")
    observations, masks, actions, game_ids, outcomes, scores = [], [], [], [], [], []
    for game_id in range(games):
        game = Game(seed=seed + game_id)
        starts = len(actions)
        actors = []
        while not game.done:
            actor = game.actor
            action = heuristic_v2(game)
            observations.append(observe(game))
            masks.append(mask(game))
            actions.append(encode(action))
            game_ids.append(game_id)
            actors.append(actor)
            game.step(action)
        final_scores = game.scores()
        top = max(final_scores)
        tied = sum(s == top for s in final_scores)
        for actor in actors:
            scores.append(final_scores[actor])
            outcomes.append(1 / tied if final_scores[actor] == top else 0.0)
        assert len(actions) - starts == len(actors)
    return {
        "observations": np.stack(observations),
        "masks": np.stack(masks),
        "actions": np.asarray(actions, dtype=np.int64),
        "game_ids": np.asarray(game_ids, dtype=np.int64),
        "win_shares": np.asarray(outcomes, dtype=np.float32),
        "final_scores": np.asarray(scores, dtype=np.float32),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=6000)
    parser.add_argument("--output", default="runs/phase6/v2_dataset.npz")
    args = parser.parse_args()
    dataset = generate(args.games, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **dataset)
    print(json.dumps({"output": str(output), "games": args.games,
                      "decisions": len(dataset["actions"]), "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
