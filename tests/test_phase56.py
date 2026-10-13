"""Phase 5.6 factorial experiment regression tests."""
from argparse import Namespace
from pathlib import Path

import pytest

from qwixx_ai.phase56 import ARMS, run, training_command


def args(tmp_path):
    return Namespace(
        champion="runs/phase5_1/best_competitive.pt",
        historical=["runs/phase4_1/champion.pt", "runs/phase3/best.pt"],
        out=str(tmp_path), episodes=20, eval_interval=5, eval_games=4,
        target_interval=100, learning_rate=0.00005,
        promotion_games=16, rollback_patience=3, alpha=0.05,
        seed=46, validation_seed=7890123, rounds=2,
        evaluate_only=False,
    )


def test_factorial_design_has_four_distinct_arms():
    assert {(weight, interval) for _, weight, interval in ARMS} == {
        (0.05, 4), (0.05, 1), (0.0, 4), (0.0, 1)
    }


def test_training_commands_hold_other_parameters_constant(tmp_path):
    a = args(tmp_path)
    commands = [training_command(a, *arm) for arm in ARMS]
    for command in commands:
        assert command[command.index("--seed") + 1] == "46"
        assert command[command.index("--champion") + 1] == a.champion
        assert command[command.index("--episodes") + 1] == "20"
        assert "--phase55" in command
    assert {c[c.index("--anchor-weight") + 1] for c in commands} == {"0.05", "0.0"}
    assert {c[c.index("--update-every") + 1] for c in commands} == {"4", "1"}


def test_requires_distinct_validation_seed(tmp_path):
    a = args(tmp_path)
    a.validation_seed = a.seed
    with pytest.raises(ValueError, match="seeds must differ"):
        run(a)


def test_missing_checkpoint_in_evaluate_only_mode(tmp_path):
    a = args(tmp_path)
    a.evaluate_only = True
    with pytest.raises(FileNotFoundError, match="no candidate saved"):
        run(a)
