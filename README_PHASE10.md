# Phase 10 — Neural challenger evaluation

Phase 10 adds an **offline, opt-in** evaluation gate for checkpoints produced by
distributed training. It does not change the deployed Phase 9.1 champion,
automatically promote checkpoints, or stop training.

## Run

From the repository root, with a compatible checkpoint at `runs/best.pt`:

```bash
python -m qwixx_ai.phase10_challenger --candidate runs/best.pt --rounds 250 --seed 1001000
```

For a stronger, independently seeded confirmation, repeat with
`--rounds 1000 --seed 1101000` after reviewing the first result.
The default JSONL history is `runs/phase10_evaluations.jsonl`;
override it with `--history PATH`.

Each evaluation compares the **same neural checkpoint** with (1) the frozen
Phase 9.1 champion and (2) Heuristic V2. Each paired comparison uses the
same seed and all four focal player seats for challenger and reference, with
three fixed V2 opponents. Neural inference is greedy (epsilon = 0).

`eligible_for_review` requires positive 95% lower confidence bounds against
both references, and no loss in mean score against Phase 9.1. It is **not**
an automatic promotion decision. Results are exploratory; rerun on new,
precommitted seeds before any promotion discussion. Reusing seeds while
tuning candidate checkpoints can bias selection.

The training dashboard's `best_win_share` measures a different opponent and
evaluation setup and **must not** be compared directly with Phase 9.1's
benchmark. This phase intentionally does not alter `/api/stats` or its
existing internal `champion_win_share` field.

## Validation

```bash
python -m ruff check qwixx_ai tests
python -m pytest -q
```

Note: Phase 9.1 search simulations can make large evaluations slow. Run this
offline rather than in the coordinator's checkpoint loop.


## Git tag in the game heading

The UI displays the Git tag beside **Qwixx AI**. It reads
`QWIXX_GIT_VERSION` embedded at image build time, rather than depending on
`.git` being present inside the running container.

For Docker Compose, use the helper script from a Git checkout:

```bash
sh scripts/deploy-tagged.sh
```

The helper uses `git describe --tags --abbrev=0` to find the most recent
reachable tag and passes it as a Docker build argument. It **does not create
or change tags**. To use a specific tag instead, run:

```bash
QWIXX_GIT_VERSION=v1.2.3 sh scripts/deploy-tagged.sh
```

If no tag is available, the heading displays `dev`. Plain
`docker compose up -d --build` also displays `dev` unless you set
`QWIXX_GIT_VERSION` in the shell or Compose environment. For a standalone
Python server, set `QWIXX_GIT_VERSION` before starting it.
