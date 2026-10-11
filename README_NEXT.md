# Distributed Qwixx AI improvements

- Separate per-browser game sessions identified by unguessable session IDs (in-memory, lost on restart).
- Browser workers run the promoted PyTorch policy locally via a JavaScript dense-network forward pass; they periodically fetch updated weights.
- Server replays browser submissions against seeded dice and checks every move before accepting experience.
- Central trainer uses current policy, promoted policy, heuristic/random play and a small historical snapshot league.
- Central evaluation saves latest.pt and promotes best.pt only on an improvement against fixed seeds; newly created games use promoted weights.
- Continuous training and metrics persist to `./runs`. Models survive restarts, but replay and sessions do not.

## Start
`mkdir -p runs && sudo chown -R 10001:10001 runs && docker compose up -d --build`

Open http://NAS_IP:8765. Enable browser training on trusted LAN clients.

## Caveats
This is an experimental distributed DQN system. Browser workers simulate games and perform neural-network inference, but **weight updates remain central**. The tournament gate uses a small fixed evaluation set and may overfit. Sessions are not authenticated, have no TTL cleanup yet, and are not persistent. Restrict access to trusted LAN users. Do not publish publicly.
