"""Phase 7.1: disagreement-weighted fine-tuning with a fresh holdout.

Only newly collected games are used for training/validation. The Phase 6
checkpoint is the initialization, not a source of supposedly unseen examples.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import load, save


def fit(dataset, initial, output, epochs=12, batch_size=512, lr=0.0001,
        seed=731000, holdout_fraction=0.2, disagreement_weight=3.0,
        jump_weight=1.5):
    if (epochs < 1 or batch_size < 1 or lr <= 0
            or not 0 < holdout_fraction < 1
            or disagreement_weight < 1 or jump_weight < 1):
        raise ValueError("invalid training parameters")
    with np.load(dataset) as data:
        x = data["observations"].copy()
        legal = data["masks"].copy()
        targets = data["actions"].copy()
        ids = data["game_ids"].copy()
        weights = data["weights"].copy() if "weights" in data else np.ones(len(ids))
    if not len(ids) or len(np.unique(ids)) < 2:
        raise ValueError("at least two games required")
    if not legal[np.arange(len(targets)), targets].all():
        raise ValueError("illegal labels")
    rng = np.random.default_rng(seed)
    games = np.unique(ids)
    rng.shuffle(games)
    n_val = max(1, min(len(games) - 1, round(len(games) * holdout_fraction)))
    train_idx = np.flatnonzero(~np.isin(ids, games[:n_val]))
    val_idx = np.flatnonzero(np.isin(ids, games[:n_val]))
    torch.manual_seed(seed)
    model = load(initial)
    if model[0].in_features != x.shape[1]:
        raise ValueError("incompatible checkpoint")
    features = torch.from_numpy(x).float()
    masks = torch.from_numpy(legal).bool()
    labels = torch.from_numpy(targets).long()
    weights = torch.from_numpy(weights).float()
    # These weights are computed at collection time using the frozen baseline.
    weights = torch.where(weights >= 2, weights * disagreement_weight / 2, weights)
    weights = torch.where((weights > 1) & (weights < 2),
                          weights * jump_weight, weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best = -1.0
    history = []
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        rng.shuffle(train_idx)
        for start in range(0, len(train_idx), batch_size):
            batch = train_idx[start:start + batch_size]
            logits = model(features[batch]).masked_fill(~masks[batch], -1e9)
            losses = nn.functional.cross_entropy(logits, labels[batch],
                                                 reduction="none")
            loss = (losses * weights[batch]).sum() / weights[batch].sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for start in range(0, len(val_idx), batch_size):
                batch = val_idx[start:start + batch_size]
                logits = model(features[batch]).masked_fill(~masks[batch], -1e9)
                correct += int((logits.argmax(-1) == labels[batch]).sum())
        accuracy = correct / len(val_idx)
        history.append({"epoch": epoch + 1, "holdout_accuracy": accuracy})
        if accuracy > best:
            best = accuracy
            save(model, destination, phase="phase7_1", epoch=epoch + 1,
                 holdout_accuracy=accuracy)
    return {"checkpoint": str(destination), "holdout_accuracy": best,
            "train_games": len(games) - n_val, "holdout_games": n_val,
            "history": history,
            "warning": "Holdout is new relative to Phase 6 but was used for epoch selection."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="runs/phase7_1/weighted.npz")
    parser.add_argument("--initial", default="runs/phase7/targeted.pt")
    parser.add_argument("--output", default="runs/phase7_1/candidate.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=731000)
    parser.add_argument("--disagreement-weight", type=float, default=3.0)
    parser.add_argument("--jump-weight", type=float, default=1.5)
    args = parser.parse_args()
    print(json.dumps(fit(args.dataset, args.initial, args.output, args.epochs,
                         args.batch_size, args.lr, args.seed,
                         disagreement_weight=args.disagreement_weight,
                         jump_weight=args.jump_weight), indent=2))


if __name__ == "__main__":
    main()
