# Distributed Qwixx AI

The container continuously runs simulated games and trains a shared DQN in a background thread. Browsers can opt in to generate additional games locally using JavaScript. Completed action histories are uploaded and replayed against server-issued dice; only legal, completed games enter the central replay buffer. All training and checkpoints live in the central container's mounted `runs` directory.

## Deploy

```bash
mkdir -p runs
sudo chown -R 10001:10001 runs
docker compose up -d --build
```

Visit `http://YOUR_NAS_IP:8765` and click **Start browser training** on as many computers as desired. The central container trains even with no browsers connected. Browser contributions stop when the tab is closed or suspended. Check `GET /api/stats` for contributions and learning metrics. Models are saved as `runs/latest.pt` and `runs/best.pt`, with `runs/metrics.jsonl` for historical metrics.

## Important limitations

- Browser clients run heuristic/exploration game rollouts; **neural network optimization happens centrally**, not in the browser. The server's learner mixes browser experience with centrally generated experience.
- Browser work is opt-in, only while the tab is open. Browser clients do not download model weights, so they are experience generators rather than policy actors.
- The web game itself currently uses heuristic opponents; new model checkpoints are **not automatically loaded into the interactive game**.
- Browser training is subject to throttling and tab suspension. One browser sends one game at a time.
- A single browser game session is shared among visitors. Multi-user independent play sessions are not implemented.
- The HTTP endpoints have no login; keep on a trusted LAN. A token is supported for API workers but **not for the bundled browser JS**, so do not enable `QWIXX_WORKER_TOKEN` when using browser workers.
- The Qwixx rules engine is experimental and has not been certified against all official edge cases.
- The in-memory replay buffer is not durable; completed model checkpoints and metrics are persistent. A container restart clears replay experience.
- The current trainer has no explicit CPU throttling other than PyTorch thread count and a short pause per rollout.
