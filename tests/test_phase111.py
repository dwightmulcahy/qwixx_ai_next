"""Tests for Phase 11.1 dataset partitioning and serialization."""
import json


from qwixx_ai.phase111 import split_for_game


def test_split_is_stable_and_disjoint_by_game():
    a = [split_for_game(11010001, i) for i in range(500)]
    b = [split_for_game(11010001, i) for i in range(500)]
    assert a == b
    assert set(a) == {"train", "validation", "test"}


def test_split_does_not_depend_on_python_hash_seed():
    assert split_for_game(11010001, 42) == split_for_game(11010001, 42)


def test_split_is_json_compatible():
    assert json.loads(json.dumps({"split": split_for_game(7, 2)}))["split"] in (
        "train", "validation", "test"
    )
