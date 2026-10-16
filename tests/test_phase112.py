"""Phase 11.2 deterministic balancing and optional target tests."""
from qwixx_ai.phase112 import balanced_select, priority, stage


def test_stage_boundaries():
    assert [stage(x) for x in (0, 11, 12, 34, 35)] == [
        "early", "early", "middle", "middle", "late"]


def test_priority_is_stable():
    assert priority(4, 3, 2, 1, 0) == priority(4, 3, 2, 1, 0)
    assert priority(4, 3, 2, 1, 0) != priority(4, 3, 2, 1, 1)


def test_balancing_preserves_groups_before_second_round():
    records = [
        {"game_id": i, "turn": i, "phase": 0, "actor": 1,
         "split": "train", "stage": "early", "priority": i}
        for i in range(10)
    ]
    records += [
        {"game_id": 99, "turn": 5, "phase": 0, "actor": 2,
         "split": "test", "stage": "late", "priority": 99},
        {"game_id": 98, "turn": 5, "phase": 0, "actor": 2,
         "split": "validation", "stage": "middle", "priority": 98},
    ]
    chosen = balanced_select(records, 3)
    assert {r["split"] for r in chosen} == {
        "train", "test", "validation"}
    assert chosen == balanced_select(list(reversed(records)), 3)


def test_balancing_rejects_invalid_limit():
    try:
        balanced_select([], 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
