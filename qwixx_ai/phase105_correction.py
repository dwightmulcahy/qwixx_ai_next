"""Phase 10.5: train a conservative, gated neural action correction head.

The original neural checkpoint remains frozen. Corrections apply only where
the small head predicts a substantial gain with adequate training evidence.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .env import ACTION_COUNT, encode, mask, observe
from .model import choose, load


class Correction(nn.Module):
    def __init__(self, obs_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_size, 64), nn.ReLU(),
                                 nn.Linear(64, ACTION_COUNT))

    def forward(self, x):
        return self.net(x)


def prepare(records, min_gain=0.12, min_score_gain=0.0):
    """Only train on rollouts that substantially beat the frozen policy."""
    chosen = []
    for row in records:
        baseline = next(x for x in row["estimates"]
                        if x["action"] == row["neural_action"])
        alternatives = [x for x in row["estimates"]
                        if x["action"] != row["neural_action"]
                        and x["win_share"] >= baseline["win_share"] + min_gain
                        and x["mean_score"] >= baseline["mean_score"] + min_score_gain]
        if alternatives:
            winner = max(alternatives, key=lambda x:
                         (x["win_share"], x["mean_score"]))
            chosen.append((row["observation"], winner["action"]))
    return chosen


def fit(dataset, output, min_gain=0.12, epochs=60, seed=20261015):
    torch.manual_seed(seed)
    records = json.loads(Path(dataset).read_text())["samples"]
    examples = prepare(records, min_gain=min_gain)
    if len(examples) < 50:
        raise ValueError("Insufficient confident corrections (need >= 50); collect more evidence")
    x = torch.tensor(np.asarray([row[0] for row in examples]),
                     dtype=torch.float32)
    y = torch.tensor([row[1] for row in examples], dtype=torch.long)
    head = Correction(x.shape[1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
    for _ in range(epochs):
        loss = nn.functional.cross_entropy(head(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": head.state_dict(), "obs_size": x.shape[1],
                "examples": len(examples), "min_gain": min_gain},
               destination)
    return {"examples": len(examples), "total": len(records),
            "output": str(destination)}


def corrected_policy(base_checkpoint, correction_checkpoint,
                     margin=1.5, seat_filter=(2, 3),
                     enable_experimental_overrides=False):
    base = load(base_checkpoint)
    payload = torch.load(correction_checkpoint, map_location="cpu",
                         weights_only=True)
    head = Correction(payload["obs_size"])
    head.load_state_dict(payload["model"])
    head.eval()

    def act(game):
        original = choose(base, game, epsilon=0)
        if not enable_experimental_overrides or game.actor + 1 not in seat_filter:
            return original
        with torch.no_grad():
            logits = head(torch.from_numpy(observe(game)))
        valid = torch.from_numpy(mask(game))
        logits = logits.masked_fill(~valid, -1e9)
        proposed = int(logits.argmax())
        if proposed == encode(original):
            return original
        # This is a classifier logit margin, NOT a calibrated win probability.
        if (logits[proposed] - logits[encode(original)]).item() < margin:
            return original
        from .env import decode
        return decode(proposed)

    return act


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path,
                   default=Path("runs/phase105/correction.pt"))
    p.add_argument("--min-gain", type=float, default=0.12)
    p.add_argument("--epochs", type=int, default=60)
    args = p.parse_args(argv)
    if args.min_gain <= 0 or args.epochs < 1:
        p.error("min-gain and epochs must be positive")
    print(json.dumps(fit(args.dataset, args.output, args.min_gain,
                         args.epochs), indent=2))


if __name__ == "__main__":
    main()
