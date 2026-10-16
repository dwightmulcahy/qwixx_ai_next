"""Phase 11.3: offline supervised fine-tuning on Phase 11.2 disagreements.

Train/validation only. Test data is never read for model selection.
No automatic promotion, deployment, or champion mutation.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .env import ACTION_COUNT
from .model import load


def read_dataset(path):
    splits = {"train": [], "validation": []}
    counts = {"train": 0, "validation": 0, "test": 0}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            split = record["split"]
            if split not in counts:
                raise ValueError(f"unknown split: {split}")
            counts[split] += 1
            if split == "test":
                continue
            actions = record["legal_actions"]
            teacher = record["champion_action"]
            if teacher not in actions or not actions or len(set(actions)) != len(actions):
                raise ValueError("invalid teacher or legal actions")
            if any(not 0 <= int(a) < ACTION_COUNT for a in actions):
                raise ValueError("invalid action index")
            record["observation"] = np.asarray(
                record["observation"], dtype=np.float32)
            if not np.isfinite(record["observation"]).all():
                raise ValueError("nonfinite observation")
            splits[split].append(record)
    if not splits["train"] or not splits["validation"]:
        raise ValueError("both train and validation examples required")
    train_games = {(r["game_seed"]) for r in splits["train"]}
    val_games = {(r["game_seed"]) for r in splits["validation"]}
    if train_games & val_games:
        raise ValueError("game leakage between train and validation")
    return splits, counts


def batch_loss(model, records, device, search_weight):
    observations = torch.from_numpy(
        np.stack([r["observation"] for r in records])).to(device)
    valid = torch.zeros((len(records), ACTION_COUNT),
                        dtype=torch.bool, device=device)
    for i, r in enumerate(records):
        valid[i, r["legal_actions"]] = True
    logits = model(observations).masked_fill(~valid, -1e9)
    labels = torch.tensor(
        [r["champion_action"] for r in records],
        dtype=torch.long, device=device)
    teacher_loss = nn.functional.cross_entropy(logits, labels,
                                               reduction="none")
    if search_weight:
        for i, record in enumerate(records):
            targets = record.get("search_targets")
            if not targets:
                continue
            # Rollout win-share estimates define a soft ranking among legal
            # actions. They are NOT Bellman targets or optimal Q-values.
            target_actions = [int(t["action"]) for t in targets]
            if set(target_actions) != set(record["legal_actions"]):
                raise ValueError("search targets must cover all legal actions")
            wins = torch.tensor(
                [float(t["win_share"]) for t in targets],
                dtype=torch.float32, device=device)
            if not torch.isfinite(wins).all():
                raise ValueError("nonfinite search targets")
            probs = torch.softmax(wins * 8.0, dim=0)
            search_loss = -(probs * nn.functional.log_softmax(
                logits[i, target_actions], dim=0)).sum()
            teacher_loss[i] = (
                (1 - search_weight) * teacher_loss[i]
                + search_weight * search_loss)
    return teacher_loss.mean()


def accuracy(model, records, batch_size=128):
    correct = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            chunk = records[start:start + batch_size]
            x = torch.from_numpy(
                np.stack([r["observation"] for r in chunk]))
            scores = model(x)
            for i, record in enumerate(chunk):
                legal = record["legal_actions"]
                chosen = legal[int(scores[i, legal].argmax())]
                correct += chosen == record["champion_action"]
    return correct / len(records)


def train(dataset, checkpoint, output, epochs=30, seed=11030001,
          batch_size=64, learning_rate=0.0001, patience=6,
          search_weight=0.25):
    if (epochs < 1 or batch_size < 1 or patience < 1
            or learning_rate <= 0 or not 0 <= search_weight <= 1):
        raise ValueError("invalid training settings")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    splits, counts = read_dataset(dataset)
    model = load(checkpoint)
    size = model[0].in_features
    if any(len(r["observation"]) != size
           for rows in splits.values() for r in rows):
        raise ValueError("dataset observation shape differs from checkpoint")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best = float("inf")
    best_epoch = 0
    history = []
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(splits["train"]))
        losses = []
        for start in range(0, len(order), batch_size):
            batch = [splits["train"][int(i)] for i in
                     order[start:start + batch_size]]
            optimizer.zero_grad()
            loss = batch_loss(model, batch, "cpu", search_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_losses = [
                float(batch_loss(model, splits["validation"][i:i + batch_size],
                                 "cpu", search_weight))
                for i in range(0, len(splits["validation"]), batch_size)
            ]
        validation = float(np.mean(validation_losses))
        item = {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "validation_loss": validation,
            "validation_teacher_agreement": accuracy(
                model, splits["validation"]),
        }
        history.append(item)
        if validation < best - 1e-5:
            best, best_epoch = validation, epoch
            torch.save({
                "model": model.state_dict(), "obs_size": size,
                "phase": "11.3", "epoch": epoch,
                "source_checkpoint": str(checkpoint),
                "dataset": str(dataset), "seed": seed,
            }, output)
        if epoch - best_epoch >= patience:
            break
    report = {
        "phase": "11.3", "dataset": str(dataset),
        "source_checkpoint": str(checkpoint),
        "candidate": str(output), "seed": seed,
        "split_counts": counts,
        "best_epoch": best_epoch, "best_validation_loss": best,
        "epochs_completed": len(history),
        "search_weight": search_weight,
        "history": history,
        "promoted": False,
        "note": "Offline teacher fine-tuning only. Test split excluded. "
                "Teacher agreement is not playing strength. "
                "Independent paired tournaments required.",
    }
    report_path = output.with_suffix(".training.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path(
        "runs/phase11/candidate_phase113.pt"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=11030001)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.0001)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--search-weight", type=float, default=0.25)
    args = p.parse_args(argv)
    result = train(args.dataset, args.checkpoint, args.output,
                   args.epochs, args.seed, args.batch_size,
                   args.learning_rate, args.patience, args.search_weight)
    print(json.dumps({k: v for k, v in result.items()
                      if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
