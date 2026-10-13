"""Phase 6: supervised V2 imitation using the existing DQN network."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import network, save


def train(dataset, output, epochs=20, batch_size=512, lr=0.001,
          val_fraction=0.2, seed=6100):
    if epochs < 1 or batch_size < 1 or not 0 < val_fraction < 1:
        raise ValueError("invalid training parameters")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with np.load(dataset) as data:
        x = data["observations"].copy()
        masks = data["masks"].copy()
        actions = data["actions"].copy()
        ids = data["game_ids"].copy()
    unique = np.unique(ids)
    if len(unique) < 2:
        raise ValueError("dataset must contain at least two games")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, min(len(unique) - 1, round(len(unique) * val_fraction)))
    val = np.isin(ids, unique[:n_val])
    train_idx = np.flatnonzero(~val)
    val_idx = np.flatnonzero(val)
    if not masks[np.arange(len(actions)), actions].all():
        raise ValueError("dataset contains illegal labels")
    device = torch.device("cpu")
    model = network(x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    features = torch.from_numpy(x).float()
    legal = torch.from_numpy(masks).bool()
    targets = torch.from_numpy(actions).long()
    best_accuracy = -1.0
    history = []
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        rng.shuffle(train_idx)
        total_loss = 0.0
        for start in range(0, len(train_idx), batch_size):
            batch = train_idx[start:start + batch_size]
            logits = model(features[batch]).masked_fill(~legal[batch], -1e9)
            loss = nn.functional.cross_entropy(logits, targets[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch)
        model.eval()
        correct = 0
        with torch.no_grad():
            for start in range(0, len(val_idx), batch_size):
                batch = val_idx[start:start + batch_size]
                logits = model(features[batch]).masked_fill(~legal[batch], -1e9)
                correct += int((logits.argmax(dim=1) == targets[batch]).sum())
        accuracy = correct / len(val_idx)
        history.append({"epoch": epoch + 1, "train_loss": total_loss / len(train_idx),
                        "validation_accuracy": accuracy})
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save(model, output, phase="phase6_imitation",
                 validation_accuracy=accuracy, seed=seed, epoch=epoch + 1)
    return {"checkpoint": str(output), "train_states": len(train_idx),
            "validation_states": len(val_idx), "train_games": len(unique) - n_val,
            "validation_games": n_val, "best_validation_accuracy": best_accuracy,
            "history": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="runs/phase6/v2_dataset.npz")
    parser.add_argument("--output", default="runs/phase6/imitation.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=6100)
    args = parser.parse_args()
    print(json.dumps(train(args.dataset, args.output, args.epochs,
                           args.batch_size, args.lr, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
