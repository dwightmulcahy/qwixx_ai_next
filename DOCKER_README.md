# Qwixx AI Docker deployment

## Start web interface

```sh
mkdir -p runs
# Runs as UID 10001 inside container. On Linux/QNAP ensure write access for training:
sudo chown -R 10001:10001 runs
docker compose up -d --build qwixx-web
```

Open `http://YOUR_NAS_IP:8765`.

**Security:** The web app has no authentication and stores one shared game session for all visitors. Keep the port on a trusted LAN; do not expose directly to the Internet or a public Cloudflare Tunnel.

## Train model (separate optional service)

```sh
cp .env.example .env
docker compose --profile train run --rm qwixx-train
```

Adjust TRAIN_EPISODES and EVAL_GAMES in `.env`. Results persist under `./runs/` (best.pt, latest.pt, metrics.jsonl). Training can use significant CPU/RAM on a NAS.

## Use the trained model

The web service defaults to heuristic opponents. To load the trained DQN model, add this argument to the **end** of the `qwixx-web` command in `compose.yaml`:

```yaml
      - --model
      - /app/runs/best.pt
```

Then restart:

```sh
docker compose up -d qwixx-web
```

The web container mounts runs read-only. Train first before enabling the model flag, or the web service will fail on missing checkpoint.

## Maintenance

```sh
docker compose logs -f qwixx-web
docker compose ps
docker compose down
```

Note: Qwixx AI v3 remains an experimental learning engine, not a verified tournament-strength implementation.
