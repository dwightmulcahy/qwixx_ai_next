"""Phase 7: fine-tune imitation on original and learner-visited V2 labels."""
import argparse
import json
from pathlib import Path

import numpy as np

from .phase6_train import train


def merge(original, targeted, output, targeted_repeat=3):
    if targeted_repeat < 1:
        raise ValueError("targeted_repeat must be positive")
    keys = ("observations", "masks", "actions", "game_ids")
    with np.load(original) as source, np.load(targeted) as extra:
        a = {key: source[key] for key in keys}
        b = {key: extra[key] for key in keys}
    if a["observations"].shape[1:] != b["observations"].shape[1:]:
        raise ValueError("incompatible observation shapes")
    if a["masks"].shape[1:] != b["masks"].shape[1:]:
        raise ValueError("incompatible action masks")
    if not len(a["game_ids"]) or not len(b["game_ids"]):
        raise ValueError("empty dataset")
    # Preserve whole-game splitting and prevent cross-source ID collisions.
    offset = int(a["game_ids"].max()) + 1 - int(b["game_ids"].min())
    b["game_ids"] = b["game_ids"] + offset
    combined = {
        key: np.concatenate([a[key]] + [b[key]] * targeted_repeat)
        for key in keys
    }
    # Repeated samples of one game retain the same ID and cannot cross splits.
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **combined)
    return {"dataset": str(destination),
            "decisions": len(combined["actions"]),
            "games": len(np.unique(combined["game_ids"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", default="runs/phase6/v2_dataset.npz")
    parser.add_argument("--targeted", default="runs/phase7/disagreements.npz")
    parser.add_argument("--merged", default="runs/phase7/combined.npz")
    parser.add_argument("--output", default="runs/phase7/targeted.pt")
    parser.add_argument("--targeted-repeat", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=720000)
    args = parser.parse_args()
    summary = merge(args.original, args.targeted, args.merged,
                    args.targeted_repeat)
    summary["training"] = train(args.merged, args.output, args.epochs,
                                args.batch_size, args.lr, seed=args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
