"""Phase 12.7: inspect neural Q-value rankings on verified Phase 12.3 states.

Q-values are model scores, not calibrated win probabilities. Diagnostic only.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

from .core import Game
from .env import encode, mask, observe
from .heuristic_v2 import heuristic_v2
from .model import load
from .phase123 import fingerprint
from .phase121 import snapshot


def replay(report):
    by_seed = defaultdict(list)
    for record in report["decisions"]:
        by_seed[record["game_seed"]].append(record)
    for seed, records in sorted(by_seed.items()):
        pending = {(r["turn"], r["phase"], r["actor"]): r for r in records}
        if len(pending) != len(records):
            raise ValueError(f"duplicate state identity in game {seed}")
        game = Game(n=4, seed=seed)
        while not game.done:
            key = (game.turn, game.phase, game.actor)
            if key in pending:
                row = pending.pop(key)
                if fingerprint(snapshot(game)) != row["state_sha256"]:
                    raise ValueError(f"state mismatch for {seed}:{key}")
                if row["source_action"] not in {0, *(encode(a) for a in game.legal())}:
                    raise ValueError(f"source action illegal for {seed}:{key}")
                if row["champion_action"] not in {0, *(encode(a) for a in game.legal())}:
                    raise ValueError(f"champion action illegal for {seed}:{key}")
                yield row, game.clone()
            game.step(heuristic_v2(game))
        if pending:
            raise ValueError(f"unmatched states for game {seed}: {list(pending)}")


def inspect(model, row, game):
    with torch.no_grad():
        q = model(torch.from_numpy(observe(game))).detach().cpu().numpy()
    legal = [index for index, allowed in enumerate(mask(game)) if allowed]
    ranked = sorted(legal, key=lambda i: (-float(q[i]), i))
    source, champion = row["source_action"], row["champion_action"]
    if ranked[0] != source:
        raise ValueError(f"checkpoint does not reproduce challenger action at {row['state_sha256']}")
    return {
        "game_seed": row["game_seed"], "state_sha256": row["state_sha256"],
        "stage": row["stage"], "actor": row["actor"],
        "skip_group": row["skip_group"],
        "win_gap": row["win_gap"], "score_gap": row["score_gap"],
        "q_margin": float(q[source] - q[champion]),
        "top2_margin": float(q[ranked[0]] - q[ranked[1]]) if len(ranked) > 1 else None,
        "champion_rank": ranked.index(champion) + 1,
        "legal_actions": len(legal),
        "source_q": float(q[source]), "champion_q": float(q[champion]),
    }


def summarize(rows):
    if not rows:
        return {"states": 0}
    return {
        "states": len(rows), "games": len({r["game_seed"] for r in rows}),
        "mean_win_gap": mean(r["win_gap"] for r in rows),
        "mean_q_margin": mean(r["q_margin"] for r in rows),
        "mean_champion_rank": mean(r["champion_rank"] for r in rows),
        "champion_rollout_better": sum(r["win_gap"] > 0 for r in rows),
        "challenger_rollout_better": sum(r["win_gap"] < 0 for r in rows),
    }


def analyze(report, model):
    if report.get("phase") != "12.3":
        raise ValueError("expected Phase 12.3 report")
    rows = [inspect(model, row, game) for row, game in replay(report)]
    if len(rows) != len(report["decisions"]):
        raise ValueError("replayed decision count mismatch")
    # Quantile bins from Q margins; no threshold chosen based on outcomes.
    ordered = sorted(rows, key=lambda r: (r["q_margin"], r["state_sha256"]))
    bins = {}
    for index, name in enumerate(("lowest", "low", "high", "highest")):
        subset = ordered[index * len(ordered) // 4:(index + 1) * len(ordered) // 4]
        bins[name] = summarize(subset)
    rank_groups = {
        "rank_2": [r for r in rows if r["champion_rank"] == 2],
        "rank_3_plus": [r for r in rows if r["champion_rank"] >= 3],
    }
    return {
        "phase": "12.7", "source_phase": "12.3",
        "overall": summarize(rows),
        "q_margin_quartiles": bins,
        "champion_action_rank": {name: summarize(subset)
                                 for name, subset in rank_groups.items()},
        "by_stage": {stage: summarize([r for r in rows if r["stage"] == stage])
                     for stage in ("middle", "late")},
        "by_skip": {group: summarize([r for r in rows if r["skip_group"] == group])
                    for group in ("0-2", "3+")},
        "decisions": rows, "promoted": False,
        "note": "Selected disagreements with V2-continuation rollout estimates. "
                "Q margins are not win probabilities; quartiles are descriptive, "
                "not calibration or independent confirmation.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=Path("runs/phase12/phase123_confirmation.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase12/phase127_qvalues.json"))
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    report = analyze(json.loads(args.input.read_text()), load(args.checkpoint))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "decisions"}, indent=2))
    print(f"Detailed decisions: {args.output}")


if __name__ == "__main__":
    main()
