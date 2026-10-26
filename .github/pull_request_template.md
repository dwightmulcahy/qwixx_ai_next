## Summary

<!-- What changed and why? -->

## Validation

- [ ] `python -m ruff check qwixx_ai tests scripts`
- [ ] `python -m pytest -q`
- [ ] Browser/Chromium checks pass when UI changes
- [ ] `docker compose -f compose.yaml config --quiet`
- [ ] `docker compose -f compose.qnap.yaml config --quiet` with IMAGE and QNAP_DATA_DIR set

## Impact

- [ ] Game rules and simultaneous Step 1 locking remain correct.
- [ ] AI checkpoints, training databases, and existing `runs/` data remain compatible.
- [ ] Credentials, authorization, and public endpoints reviewed.
- [ ] Docker/QNAP deployment and writable data volumes reviewed.
- [ ] Docs and examples updated if behavior changed.
- [ ] PR title follows Conventional Commits.
