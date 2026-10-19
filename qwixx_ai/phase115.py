"""Phase 11.5: conservative recovery after Phase 11.3 policy collapse.

Train on disagreement examples with a frozen source-policy anchor.
Validation selects checkpoints; playing strength requires independent tournaments.
Never reads test examples for model selection or promotes a checkpoint.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import load
from .phase113 import read_dataset


def anchored_loss(model, anchor, records, anchor_weight=4.0, teacher_weight=0.1):
    """Keep original action values close while nudging toward teacher actions."""
    x = torch.from_numpy(np.stack([r["observation"] for r in records]))
    predictions = model(x)
    with torch.no_grad():
        originals = anchor(x)
    legal = torch.zeros_like(predictions, dtype=torch.bool)
    for i, record in enumerate(records):
        legal[i, record["legal_actions"]] = True
    # Preserve the original Q-value scale and ordering on legal actions.
    drift = ((predictions - originals).square() * legal).sum() / legal.sum()
    labels = torch.tensor([r["champion_action"] for r in records], dtype=torch.long)
    logits = predictions.masked_fill(~legal, -1e9)
    teacher = nn.functional.cross_entropy(logits, labels)
    return anchor_weight * drift + teacher_weight * teacher, drift, teacher


def train(dataset, checkpoint, output, epochs=6, seed=11050001,
          batch_size=64, learning_rate=0.000005, patience=3,
          anchor_weight=4.0, teacher_weight=0.1):
    if (epochs < 1 or batch_size < 1 or patience < 1 or learning_rate <= 0
            or anchor_weight <= 0 or teacher_weight < 0):
        raise ValueError("invalid recovery settings")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    splits, counts = read_dataset(dataset)
    model = load(checkpoint)
    anchor = load(checkpoint)
    anchor.eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
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
    # Include epoch zero as the safety baseline; never overwrite it with a
    # trained checkpoint unless validation improves on the frozen original.
    with torch.no_grad():
        baseline = sum(
            float(anchored_loss(model, anchor, splits["validation"][i:i + batch_size],
                                anchor_weight, teacher_weight)[0]) *
            len(splits["validation"][i:i + batch_size])
            for i in range(0, len(splits["validation"]), batch_size)
        ) / len(splits["validation"])
    best = baseline
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(splits["train"]))
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = [splits["train"][int(i)] for i in
                     order[start:start + batch_size]]
            optimizer.zero_grad()
            loss, _, _ = anchored_loss(model, anchor, batch,
                                       anchor_weight, teacher_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        model.eval()
        with torch.no_grad():
            val_total = 0.0
            drift_total = 0.0
            for start in range(0, len(splits["validation"]), batch_size):
                batch = splits["validation"][start:start + batch_size]
                loss, drift, _ = anchored_loss(
                    model, anchor, batch, anchor_weight, teacher_weight)
                val_total += float(loss) * len(batch)
                drift_total += float(drift) * len(batch)
        validation = val_total / len(splits["validation"])
        history.append({
            "epoch": epoch, "train_loss": total / len(splits["train"]),
            "validation_loss": validation,
            "validation_drift": drift_total / len(splits["validation"]),
        })
        if validation < best - 1e-6:
            best, best_epoch = validation, epoch
            torch.save({
                "model": model.state_dict(), "obs_size": size,
                "phase": "11.5", "epoch": epoch,
                "source_checkpoint": str(checkpoint),
                "dataset": str(dataset), "seed": seed,
            }, output)
        if epoch - best_epoch >= patience:
            break
    report = {
        "phase": "11.5", "dataset": str(dataset),
        "source_checkpoint": str(checkpoint), "candidate": str(output),
        "seed": seed, "split_counts": counts,
        "baseline_validation_loss": baseline,
        "best_validation_loss": best, "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "anchor_weight": anchor_weight, "teacher_weight": teacher_weight,
        "history": history, "checkpoint_saved": best_epoch > 0,
        "promoted": False,
        "note": "Recovery experiment only; tournament approval required. "
                "Disagreement-only training does not guarantee playing strength.",
    }
    output.with_suffix(".training.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(
        "runs/phase11/candidate_phase115.pt"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=11050001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.000005)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--anchor-weight", type=float, default=4.0)
    parser.add_argument("--teacher-weight", type=float, default=0.1)
    args = parser.parse_args(argv)
    result = train(args.dataset, args.checkpoint, args.output,
                   args.epochs, args.seed, args.batch_size, args.learning_rate,
                   args.patience, args.anchor_weight, args.teacher_weight)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
