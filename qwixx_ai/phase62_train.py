"""Phase 6.2: V2-initialized actor-critic with GAE and opponent diversity.

Standalone experiment. The policy head retains the existing 89-action
checkpoint format; the value head is trained separately and never promoted.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .controlled_strategy import paired_compare
from .core import Game
from .env import decode, encode, mask, observe
from .heuristic_v2 import heuristic_v2
from .model import choose, load, save


class Critic(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(size, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, states):
        return self.net(states).squeeze(-1)


def gae(rewards, values, gamma=0.99, lam=0.95):
    """Terminal-only rewards, one step per learner decision."""
    if len(rewards) != len(values):
        raise ValueError("rewards and values must have equal lengths")
    advantages = torch.zeros(len(rewards), dtype=torch.float32)
    carry = 0.0
    next_value = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        delta = rewards[i] + gamma * next_value - values[i]
        carry = delta + gamma * lam * carry
        advantages[i] = carry
        next_value = values[i]
    return advantages, advantages + torch.tensor(values)


def episode(policy, critic, seed, generator, imitation, snapshots):
    game = Game(seed=seed)
    learner = seed % 4
    states, masks, actions, teachers, log_probs, values = [], [], [], [], [], []
    # Rotate diverse opponent combinations across episodes.
    options = [heuristic_v2, imitation, *snapshots]
    opponents = [options[(seed + seat) % len(options)] for seat in range(4)]
    while not game.done:
        if game.actor == learner:
            state = torch.from_numpy(observe(game))
            legal = torch.from_numpy(mask(game))
            with torch.no_grad():
                logits = policy(state).masked_fill(~legal, -1e9)
                probabilities = torch.softmax(logits, dim=0)
                action_id = int(torch.multinomial(probabilities, 1, generator=generator))
                value = float(critic(state))
            states.append(state)
            masks.append(legal)
            actions.append(action_id)
            teachers.append(encode(heuristic_v2(game)))
            log_probs.append(float(torch.log_softmax(logits, dim=0)[action_id]))
            values.append(value)
            action = decode(action_id)
        else:
            action = opponents[game.actor](game)
        game.step(action)
    scores = game.scores()
    top = max(scores)
    share = (1 / sum(score == top for score in scores)
             if scores[learner] == top else 0.0)
    margin = scores[learner] - max(
        score for i, score in enumerate(scores) if i != learner)
    reward = share + 0.1 * max(-1.0, min(1.0, margin / 50))
    rewards = [0.0] * len(states)
    if rewards:
        rewards[-1] = reward
    advantages, returns = gae(rewards, values)
    return (states, masks, actions, teachers, log_probs,
            advantages.tolist(), returns.tolist(), share, scores[learner])


def train(initial="runs/phase6/imitation.pt", out="runs/phase6_2",
          episodes=2000, seed=6500, lr=3e-5, batch_games=16,
          eval_interval=250, eval_rounds=80, imitation_weight=0.05,
          gamma=0.99, gae_lambda=0.95, clip=0.15):
    if (episodes < 1 or batch_games < 1 or eval_interval < 1
            or eval_rounds < 2 or lr <= 0 or imitation_weight < 0
            or not 0 < gamma <= 1 or not 0 <= gae_lambda <= 1
            or not 0 < clip < 1):
        raise ValueError("invalid training parameters")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    policy = load(initial)
    imitation_model = load(initial)
    imitation = lambda game: choose(imitation_model, game)
    snapshots = []
    critic = Critic(policy[0].in_features)
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()),
                                 lr=lr)
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    best_advantage = float("-inf")
    best_path = directory / "best_dev.pt"
    for index in range(episodes):
        records.append(episode(policy, critic, seed + index, generator,
                               imitation, snapshots))
        if len(records) == batch_games or index == episodes - 1:
            states, legal, actions, teachers, old_logp, adv, ret = (
                [item for record in records for item in record[field]]
                for field in range(7))
            states = torch.stack(states)
            legal = torch.stack(legal)
            actions = torch.tensor(actions, dtype=torch.long)
            teachers = torch.tensor(teachers, dtype=torch.long)
            old_logp = torch.tensor(old_logp)
            adv = torch.tensor(adv)
            ret = torch.tensor(ret)
            if len(adv) > 1:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-6)
            logits = policy(states).masked_fill(~legal, -1e9)
            logp = torch.log_softmax(logits, dim=-1).gather(
                1, actions[:, None]).squeeze(1)
            ratio = torch.exp(logp - old_logp)
            objective = torch.minimum(ratio * adv,
                                      ratio.clamp(1 - clip, 1 + clip) * adv)
            weight = imitation_weight * (1 - 0.75 * index / max(1, episodes - 1))
            loss = (-objective.mean()
                    + 0.5 * nn.functional.mse_loss(critic(states), ret)
                    + weight * nn.functional.cross_entropy(logits, teachers))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(critic.parameters()), 2.0)
            optimizer.step()
            records.clear()
        if (index + 1) % eval_interval == 0 or index == episodes - 1:
            policy.eval()
            challenger = lambda game: choose(policy, game)
            # Fixed development seeds only. Never use these for final promotion.
            report = paired_compare(challenger, imitation,
                                    [heuristic_v2] * 3,
                                    rounds=eval_rounds, seed=765432)
            row = {"episode": index + 1, "development": report}
            with (directory / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            save(policy, directory / "latest.pt", phase="phase6_2",
                 episode=index + 1)
            if report["paired_advantage"] > best_advantage:
                best_advantage = report["paired_advantage"]
                save(policy, best_path, phase="phase6_2", episode=index + 1,
                     development_advantage=best_advantage)
            # Snapshots are frozen and deliberately limited.
            frozen = load(directory / "latest.pt")
            snapshots.append(lambda game, model=frozen: choose(model, game))
            snapshots = snapshots[-3:]
            print(json.dumps(row), flush=True)
            policy.train()
    return {"best_development_checkpoint": str(best_path),
            "latest_checkpoint": str(directory / "latest.pt"),
            "best_development_advantage": best_advantage}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", default="runs/phase6/imitation.pt")
    parser.add_argument("--out", default="runs/phase6_2")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=6500)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-games", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-rounds", type=int, default=80)
    parser.add_argument("--imitation-weight", type=float, default=0.05)
    args = parser.parse_args()
    print(json.dumps(train(args.initial, args.out, args.episodes, args.seed,
                           args.lr, args.batch_games, args.eval_interval,
                           args.eval_rounds, args.imitation_weight), indent=2))


if __name__ == "__main__":
    main()
