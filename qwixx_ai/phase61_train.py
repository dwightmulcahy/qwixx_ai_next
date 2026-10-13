"""Phase 6.1: on-policy V2-anchored policy-gradient fine-tuning.

The Phase 6 network outputs are treated as policy logits, not Q-values.
No existing champion or distributed trainer is modified.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .core import Game
from .env import decode, encode, mask, observe
from .heuristic_v2 import heuristic_v2
from .model import load, save


def sample_action(policy, game, generator):
    """Sample only legal actions; return action and differentiable log probability."""
    features = torch.from_numpy(observe(game))
    legal = torch.from_numpy(mask(game))
    logits = policy(features).masked_fill(~legal, -1e9)
    probabilities = torch.softmax(logits, dim=0)
    index = int(torch.multinomial(probabilities.detach(), 1, generator=generator))
    return decode(index), torch.log_softmax(logits, dim=0)[index]


def play_episode(policy, seed, generator, epsilon=0.05):
    game = Game(seed=seed)
    learner = seed % game.n
    log_probs, imitation_losses = [], []
    while not game.done:
        actor = game.actor
        if actor == learner:
            state = torch.from_numpy(observe(game))
            legal = torch.from_numpy(mask(game))
            teacher = encode(heuristic_v2(game))
            logits = policy(state).masked_fill(~legal, -1e9)
            imitation_losses.append(nn.functional.cross_entropy(
                logits.unsqueeze(0), torch.tensor([teacher])))
            action, log_prob = sample_action(policy, game, generator)
            # Occasional teacher moves are intentionally excluded from policy
            # gradient; otherwise the gradient would use the wrong behavior policy.
            if epsilon and torch.rand((), generator=generator).item() < epsilon:
                action = decode(teacher)
            else:
                log_probs.append(log_prob)
        else:
            action = heuristic_v2(game)
        game.step(action)
    scores = game.scores()
    highest = max(scores)
    share = (1 / sum(s == highest for s in scores)
             if scores[learner] == highest else 0.0)
    margin = scores[learner] - max(s for i, s in enumerate(scores) if i != learner)
    # Terminal win share dominates; bounded score margin breaks ties.
    reward = share + 0.1 * max(-1.0, min(1.0, margin / 50.0))
    return reward, share, scores[learner], log_probs, imitation_losses


def train(initial, output_dir="runs/phase6_1", episodes=3000, seed=6300,
          lr=0.00003, imitation_weight=0.05, decay=0.5,
          batch_games=16, eval_interval=250, epsilon=0.05):
    if (episodes < 1 or batch_games < 1 or eval_interval < 1 or lr <= 0
            or imitation_weight < 0 or not 0 <= decay <= 1
            or not 0 <= epsilon <= 1):
        raise ValueError("invalid training parameters")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = load(initial)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "metrics.jsonl"
    rewards, shares, scores, gradients, imitation = [], [], [], [], []
    baseline = 0.0
    for episode in range(episodes):
        reward, share, score, log_probs, imitation_losses = play_episode(
            model, seed + episode, generator, epsilon)
        # Running baseline from previous episodes, detached from current reward.
        advantage = reward - baseline
        baseline = 0.99 * baseline + 0.01 * reward
        if log_probs:
            gradients.append(-torch.stack(log_probs).sum() * advantage)
        if imitation_losses:
            imitation.append(torch.stack(imitation_losses).mean())
        rewards.append(reward)
        shares.append(share)
        scores.append(score)
        if (episode + 1) % batch_games == 0 or episode == episodes - 1:
            weight = imitation_weight * (
                1.0 - (1.0 - decay) * episode / max(1, episodes - 1))
            terms = []
            if gradients:
                terms.append(torch.stack(gradients).mean())
            if imitation and weight:
                terms.append(weight * torch.stack(imitation).mean())
            if terms:
                loss = sum(terms)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
            gradients.clear()
            imitation.clear()
        if (episode + 1) % eval_interval == 0 or episode == episodes - 1:
            row = {
                "episode": episode + 1,
                "training_win_share": float(np.mean(shares[-eval_interval:])),
                "training_mean_score": float(np.mean(scores[-eval_interval:])),
                "training_reward": float(np.mean(rewards[-eval_interval:])),
                "imitation_weight": weight,
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            save(model, output / "latest.pt", phase="phase6_1",
                 episode=episode + 1, seed=seed)
            print(json.dumps(row), flush=True)
    return {"checkpoint": str(output / "latest.pt"), "episodes": episodes,
            "metrics": str(log_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", default="runs/phase6/imitation.pt")
    parser.add_argument("--out", default="runs/phase6_1")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=6300)
    parser.add_argument("--lr", type=float, default=0.00003)
    parser.add_argument("--imitation-weight", type=float, default=0.05)
    parser.add_argument("--decay", type=float, default=0.5)
    parser.add_argument("--batch-games", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--epsilon", type=float, default=0.05)
    args = parser.parse_args()
    print(json.dumps(train(args.initial, args.out, args.episodes, args.seed,
                           args.lr, args.imitation_weight, args.decay,
                           args.batch_games, args.eval_interval,
                           args.epsilon), indent=2))


if __name__ == "__main__":
    main()
