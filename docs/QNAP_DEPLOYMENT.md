# QNAP deployment

The QNAP Compose file pulls a published image rather than building locally. It does **not** migrate or copy your existing training data.

1. Identify the existing host directory currently mounted to `/app/runs`. Back up this directory, including SQLite WAL files, while the application is stopped.
2. In a deployment directory, create a `.env` with `IMAGE=<dockerhub-user>/qwixx-ai:<manually-created-version>` and `QNAP_DATA_DIR=<absolute-existing-host-runs-directory>`. Add the SMTP settings you already use. Never commit credentials.
3. Validate with `docker compose -f compose.qnap.yaml config --quiet`.
4. Stop the old instance, then run `docker compose -f compose.qnap.yaml up -d`. Do not run two trainers against the same database.
5. Verify `curl -fsS http://localhost:8765/health` (or your HOST_PORT), inspect logs, and confirm existing training history.

The root filesystem is read-only; training databases and checkpoints remain writable through the bind mount. `/tmp` is writable tmpfs. Container runs as UID 10001; ensure the host data directory permits that UID to write. Avoid changing ownership blindly on QNAP shared directories.

The existing `compose.yaml` remains the local-build development configuration. No version tags are created automatically.
