# Qwixx AI — training and deployment

## Improvements

- **Rules:** White-dice decisions are resolved for every player before a row lock closes the color. Two players may lock the same color in the same shared phase. The browser simulator follows the same delayed-closure rule.
- **Recovery:** The SQLite `training.sqlite3` database stores seed, move trace, source and scores. At startup the most recent 100 deterministic game logs are replayed into the experience buffer. Earlier logs remain stored, but are not loaded into RAM.
- **Learning:** Double DQN, proportional TD-error prioritized replay with importance weighting, adaptive epsilon, and 192-game seat-rotated comparisons against heuristic opponents. Champion promotion requires at least 3 percentage points better win share on the fixed evaluation set than the current champion. This is a practical threshold, **not a statistical confidence interval**.
- **Concurrency:** One model-training thread and 1–4 central rollout threads, configured using `QWIXX_CENTRAL_WORKERS` (default 2). Browsers can contribute via dedicated Web Workers.
- **Playing:** Choose Beginner (random), Intermediate (heuristic), Advanced (exploratory model) or Champion (greedy model). Before a champion is trained, Advanced and Champion use the heuristic fallback. The move analysis shows immediate score and permanently skipped numbers; it is not a faithful explanation of neural-network internal reasoning.
- **Operations:** Training pause/resume, manual checkpoint and rollback to best model are exposed through an administrator API.

## Deploy on QNAP

```sh
git pull origin main
docker compose up -d --build
docker compose logs -f qwixx
```

The `./runs` directory is persistent. Back it up before upgrading. SQLite WAL means that a live backup must include the WAL and SHM files or use SQLite's backup API. The container has a 2 GB memory limit; reduce worker count if memory pressure occurs.

## Administrator controls

Set a strong `QWIXX_ADMIN_TOKEN` in the container environment (for example via a QNAP secret). Without it, the admin API is disabled.

```sh
curl -X POST http://localhost:8765/api/admin \
  -H 'Content-Type: application/json' \
  -H "X-Admin-Token: $QWIXX_ADMIN_TOKEN" \
  -d '{"command":"pause"}'
```

Supported commands: `pause`, `resume`, `checkpoint`, `rollback`. Rollback restores the best-model weights and resets the Adam optimizer; it does not delete historical games. The admin token must not be embedded in browser JavaScript.

## Security

Per-IP throttling is a basic safeguard, not a substitute for authentication. Browser workers cannot send the optional `QWIXX_WORKER_TOKEN`; enabling it prevents normal browser contributions. Keep the application on a trusted LAN or protect it with an authenticated reverse proxy. The API is **not hardened for direct public exposure**.

## Validation

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
python -m compileall -q qwixx_ai tests
docker build -t qwixx-ai:test .
```

The test suite covers rules, replay restoration, TD replay alignment, AI difficulty selection, training pause, central rollout and a deterministic JavaScript-to-Python browser game replay (when Node.js is available). GitHub Actions runs tests, compilation and Docker build on pushes and PRs.

## Limitations and follow-up

Full Python/JavaScript state-by-state property-based parity, statistically significant paired champion evaluation, safe online SQLite snapshots, distributed browser authentication and full RNG state recovery remain future improvements. No automatic version tags or releases are created.
