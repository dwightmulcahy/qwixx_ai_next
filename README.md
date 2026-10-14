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

## Champion policy (four-player Qwixx)

The promoted champion is **Phase 9.1 conservative Monte Carlo search** in
`qwixx_ai/champion.py`. It is a deterministic policy wrapper, not a trained
checkpoint. It uses V2 as its baseline and falls back to V2 for games with
other than four players.

Frozen search settings: 32 futures, search seed 900000, minimum estimated
win-share improvement 0.10, minimum score improvement 0, turn window
12 (inclusive) to 35 (exclusive), and at least two legal marks.

The browser **Champion** difficulty now uses this policy even without a
model checkpoint. **Advanced** uses the loaded neural checkpoint with
exploration, or V2 if no checkpoint is supplied. Search may introduce
noticeable latency on eligible moves.

### Confirmation evidence

Paired tournaments against three V2 opponents, four focal seats per seed
block, tie-adjusted win share:

| Seed | Blocks | Champion win share | V2 win share | Advantage | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| 994000 | 250 | 27.45% | 25.00% | +2.45 pp | +1.33 to +3.57 pp |
| 995000 | 1000 | 26.5625% | 25.00% | +1.5625 pp | +0.9624 to +2.1626 pp |

The 1000-block run used 959 overrides in 130248 decisions (0.736%).
These are user-reported local runs, not CI results. Evidence applies to
the tested opponent lineup, not every player count or opponent mix.
Rollout RNG streams can diverge across action-dependent game lengths.

Run a fresh independent, expensive regression benchmark:

```bash
python -m qwixx_ai.champion_benchmark --rounds 1000 --seed 996000
```

The benchmark exits nonzero unless its 95% lower confidence bound for
the paired win-share advantage is positive. It does not automatically
promote, tag, or release any version. Heuristic V2 remains available as
the lightweight fallback.
