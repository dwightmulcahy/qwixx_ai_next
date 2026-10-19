"""Phase 12.5: progress-matched row-choice diagnostics from Phase 12.3.

No new rollouts, training, or promotion. Matching is observational and
does not identify the causal effect of skipping cells.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from .core import ORDER
from .env import decode


def features(row):
    state = row["state"]
    actor = row["actor"]
    source, champion = row["source_move"], row["champion_move"]
    board = state["rows"][actor]
    legal = [decode(action) for action in state["legal_actions"]]
    legal = [action for action in legal if action is not None]
    source_progress = source["existing_marks"]
    champion_progress = champion["existing_marks"]
    source_remaining = source["remaining_after"]
    champion_remaining = champion["remaining_after"]
    source_points = (source_progress + 1) * (source_progress + 2) // 2 - source_progress * (source_progress + 1) // 2
    champion_points = (champion_progress + 1) * (champion_progress + 2) // 2 - champion_progress * (champion_progress + 1) // 2
    if source["lock"]:
        source_points += source_progress + 2
    if champion["lock"]:
        champion_points += champion_progress + 2
    alternatives = [
        {"color": color, "number": number, "lock": lock,
         "existing_marks": len(board[color]),
         "remaining_after": len(ORDER[color]) - ORDER[color].index(number) - 1}
        for color, number, lock in legal
    ]
    return {
        "game_seed": row["game_seed"], "stage": row["stage"],
        "actor": actor, "skip_group": row["skip_group"],
        "win_gap": row["win_gap"], "score_gap": row["score_gap"],
        "state_sha256": row["state_sha256"],
        "source_color": source["color"], "champion_color": champion["color"],
        "source_progress": source_progress,
        "champion_progress": champion_progress,
        "progress_delta": champion_progress - source_progress,
        "skip_delta": champion["skipped_cells"] - source["skipped_cells"],
        "remaining_delta": champion_remaining - source_remaining,
        "immediate_delta": champion_points - source_points,
        "source_lock": bool(source["lock"]),
        "champion_lock": bool(champion["lock"]),
        "legal_count": len(alternatives),
        "legal_colors": len({a["color"] for a in alternatives}),
        "max_legal_progress": max((a["existing_marks"] for a in alternatives), default=0),
        "max_legal_remaining": max((a["remaining_after"] for a in alternatives), default=0),
        "developed_row_abandoned": champion_progress > source_progress,
        "new_row_started": source_progress == 0 and champion_progress > 0,
        "less_future_space": source_remaining < champion_remaining,
        "champion_immediate_gain": champion_points > source_points,
        "champion_lock_only": bool(champion["lock"]) and not bool(source["lock"]),
    }


MATCH_KEYS = ("stage", "actor", "source_progress", "legal_colors")


def matched_pairs(rows):
    """Exact-match strata; match distinct games without replacement."""
    by_group = defaultdict(lambda: {"0-2": [], "3+": []})
    for row in rows:
        key = tuple(row[k] for k in MATCH_KEYS)
        by_group[key][row["skip_group"]].append(row)
    pairs = []
    for key, groups in sorted(by_group.items()):
        left = sorted(groups["0-2"], key=lambda r: (r["game_seed"], r["state_sha256"]))
        right = sorted(groups["3+"], key=lambda r: (r["game_seed"], r["state_sha256"]))
        used = set()
        for high in right:
            choices = [(abs(high["legal_count"] - low["legal_count"]),
                        abs(high["max_legal_progress"] - low["max_legal_progress"]),
                        abs(high["champion_progress"] - low["champion_progress"]),
                        index, low)
                       for index, low in enumerate(left)
                       if index not in used and low["game_seed"] != high["game_seed"]]
            if not choices:
                continue
            _, _, _, index, low = min(choices, key=lambda v: v[:4])
            used.add(index)
            pairs.append({
                "match_key": dict(zip(MATCH_KEYS, key)),
                "small_seed": low["game_seed"], "large_seed": high["game_seed"],
                "small_state_sha256": low["state_sha256"],
                "large_state_sha256": high["state_sha256"],
                "win_gap_difference": high["win_gap"] - low["win_gap"],
                "legal_count_difference": high["legal_count"] - low["legal_count"],
                "champion_progress_difference": high["champion_progress"] - low["champion_progress"],
            })
    return pairs


def summary(rows):
    if not rows:
        return {"states": 0}
    return {
        "states": len(rows), "games": len({r["game_seed"] for r in rows}),
        "mean_win_gap": mean(r["win_gap"] for r in rows),
        "mean_score_gap": mean(r["score_gap"] for r in rows),
        "mean_progress_delta": mean(r["progress_delta"] for r in rows),
        "mean_immediate_delta": mean(r["immediate_delta"] for r in rows),
        "mean_remaining_delta": mean(r["remaining_delta"] for r in rows),
    }


def analyze(report):
    if report.get("phase") != "12.3":
        raise ValueError("expected Phase 12.3 report")
    rows = [features(r) for r in report["decisions"]]
    pairs = matched_pairs(rows)
    categories = {
        "developed_row_abandoned": lambda r: r["developed_row_abandoned"],
        "new_row_started": lambda r: r["new_row_started"],
        "less_future_space": lambda r: r["less_future_space"],
        "champion_immediate_gain": lambda r: r["champion_immediate_gain"],
        "champion_lock_only": lambda r: r["champion_lock_only"],
    }
    breakdown = {}
    for name, predicate in categories.items():
        breakdown[name] = {
            "yes": summary([r for r in rows if predicate(r)]),
            "no": summary([r for r in rows if not predicate(r)]),
        }
    return {
        "phase": "12.5", "source_phase": "12.3",
        "overall": summary(rows),
        "groups": {g: summary([r for r in rows if r["skip_group"] == g])
                   for g in ("0-2", "3+")},
        "patterns": breakdown,
        "match_keys": list(MATCH_KEYS),
        "matched_pairs": len(pairs),
        "matched_difference": mean(p["win_gap_difference"] for p in pairs) if pairs else None,
        "matched_diagnostics": {
            "mean_abs_legal_count_difference": mean(abs(p["legal_count_difference"]) for p in pairs) if pairs else None,
            "mean_abs_champion_progress_difference": mean(abs(p["champion_progress_difference"]) for p in pairs) if pairs else None,
            "positive": sum(p["win_gap_difference"] > 0 for p in pairs),
            "negative": sum(p["win_gap_difference"] < 0 for p in pairs),
        },
        "unmatched_by_group": dict(Counter(r["skip_group"] for r in rows) -
                                   Counter({g: len(pairs) for g in ("0-2", "3+")})),
        "pair_details": pairs,
        "decisions": rows,
        "promoted": False,
        "note": "Exploratory, selected-state exact-stratum matching with greedy nearest neighbors. No causal or optimal-action labels.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("runs/phase12/phase123_confirmation.json"))
    p.add_argument("--output", type=Path, default=Path("runs/phase12/phase125_diagnostics.json"))
    args = p.parse_args(argv)
    report = analyze(json.loads(args.input.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("decisions", "pair_details")}, indent=2))
    print(f"Detailed diagnostics: {args.output}")


if __name__ == "__main__":
    main()
