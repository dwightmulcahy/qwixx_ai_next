# Qwixx AI — self-play + tournament + browser

Python 3.10+; standard rules engine, public-state observation, shared-policy DQN self-play, replay buffer, frozen opponent snapshots, evaluation and best/latest checkpoints. **No pretrained weights included.**

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

## Train

```bash
python -m qwixx_ai.train --episodes 10000 --eval-games 200 --out runs
python -m qwixx_ai.tournament runs/best.pt --games 1000 --opponent heuristic
python -m qwixx_ai.tournament runs/best.pt --games 1000 --opponent random
```

Training uses one rotated learner seat with opponents sampled from current-policy self-play, heuristic, random and up to eight frozen historical snapshots. Every evaluation checkpoint measures performance against fixed heuristic players using a fixed seed schedule, and saves `best.pt` on improved win share. `latest.pt` saves the latest model. JSONL metrics are in `runs/metrics.jsonl`. A win tie counts fractionally.

## Play in your browser

```bash
python -m qwixx_ai.web --model runs/best.pt
```

Open http://127.0.0.1:8765. Without `--model`, it plays against heuristic bots. Web server is **local-only by default** and keeps one shared in-memory game (not multi-user authenticated). Don't expose it to the internet.

## Rules and modeling

The game engine separates white+white (active then each nonactive player) and white+color (active player). Active player can make two marks; nonactive one. If the active player makes no mark, they receive a penalty. Four penalties on any player or two locked colors ends the game. Locks remove a color from future dice opportunities. Score uses triangular mark count, including lock symbol as an extra mark for the player who locked the row. The game is fully visible-information. The active player may pass a phase.

**Caveats:** This is an experimental implementation. Validate edge cases against your specific Qwixx edition. The trainer is not yet optimized for millions of episodes, and no model-strength guarantee is implied. It trains on one learner seat per game rather than storing transitions from all four seats. Gymnasium-compatible wrapper is available, and Gymnasium is an optional install. No automatic git tagging/version releases.
