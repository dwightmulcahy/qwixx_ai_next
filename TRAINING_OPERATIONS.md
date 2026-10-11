# Training operations and limitations

## QNAP deployment

Use `docker compose up -d --build`. Persist `./runs` and back it up before upgrading. The container has a 2 GB memory cap and a 60-second stop grace period. Browser simulations use dedicated Web Workers (up to two per tab).

## Persistence

`training.sqlite3` stores accepted game scores and browser move traces, `latest.pt` stores the neural network, and `trainer.pt` stores Adam optimizer state and counters. SQLite uses WAL mode. The in-memory replay buffer is **not yet reconstructed after restart**; stored browser move traces provide the foundation for future replay restoration. Training should be stopped cleanly before copying all files in `runs`.

## Learning

The learner uses Double DQN targets, reward-aware replay sampling, and a larger fixed-seed evaluation (192 games). Reward-aware sampling is an approximation, **not full TD-error prioritized replay**. Champion promotion is still based on fixed-seed evaluation and does not yet implement confidence-interval testing.

## Security

Per-IP request throttling protects job and submission endpoints. `QWIXX_WORKER_TOKEN` can restrict server-to-server workers, but the browser worker does not supply this secret; setting it disables browser contributions. Keep the service on a trusted LAN or behind an authenticated reverse proxy. The API is **not appropriate for unrestricted public Internet exposure**.

## Dashboard

The web UI displays a recent-game score chart, training counters, uptime, and saved game history from `/api/history`. Game history is not user-authenticated; treat it as LAN-only telemetry.

## Tests

Run `python -m pytest -q` before deployment. GitHub Actions runs the suite on pushes and pull requests. Docker image builds and runtime checks should also be performed on your QNAP.
