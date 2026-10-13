"""Phase 9.1 conservative search tests."""
import pytest

from qwixx_ai.core import Game
from qwixx_ai.heuristic_v2 import heuristic_v2
from qwixx_ai.phase91_search import decide, eligible, policy, validate


def test_early_turn_skips_search():
    game = Game(seed=5)
    action, reason = decide(game, futures=1)
    assert action == heuristic_v2(game)
    assert reason == "v2_not_eligible"


def test_invalid_parameters():
    with pytest.raises(ValueError):
        decide(Game(seed=5), futures=0)
    with pytest.raises(ValueError):
        decide(Game(seed=5), min_gain=-0.1)
    with pytest.raises(ValueError):
        decide(Game(seed=5), start_turn=35, end_turn=12)


def test_policy_records_decisions():
    stats = {"decisions": 0, "v2_not_eligible": 0,
             "v2_safeguard": 0, "override": 0}
    game = Game(seed=5)
    action = policy(futures=1, stats=stats)(game)
    assert action == heuristic_v2(game)
    assert stats["decisions"] == 1
    assert stats["v2_not_eligible"] == 1


def test_eligibility_requires_choices():
    game = Game(seed=5)
    assert not eligible(game, heuristic_v2(game))


def test_validation_smoke():
    report = validate(rounds=2, seed=45, futures=1)
    assert report["against_v2"]["rounds"] == 2
    assert report["decisions"]["decisions"] > 0
    assert 0 <= report["override_rate"] <= 1
