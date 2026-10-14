"""Phase 10.2 disagreement audit regression tests."""
import json
import pytest

from qwixx_ai.core import Game
from qwixx_ai.phase10_disagreement import audit, category, describe, main


def test_action_descriptions_and_categories():
    game = Game(n=4, seed=42)
    assert describe(None)["kind"] == "pass"
    assert describe((0, 12, True)) == {
        "kind": "lock", "color": "red", "number": 12, "lock": True}
    assert category(game, None, (0, 4, False)) == "neural_pass"
    assert category(game, (0, 4, False), None) == "champion_pass"
    assert category(game, (0, 12, True), (0, 11, False)) == "lock_disagreement"
    assert category(game, (0, 7, False), (0, 5, False)) == "neural_skips_more"


def test_audit_counts_and_counterfactuals(monkeypatch):
    import qwixx_ai.phase10_disagreement as mod

    monkeypatch.setattr(mod, "load", lambda _: object())
    monkeypatch.setattr(mod, "choose", lambda model, game, epsilon: None)
    monkeypatch.setattr(mod, "champion", lambda game: game.legal()[0] if game.legal() else None)
    def estimates(game, futures, seed):
        return [{"action": action, "win_share": 0.25, "mean_score": 30.0}
                for action in [None, *game.legal()]]
    monkeypatch.setattr(mod, "evaluate_actions", estimates)
    result = audit("frozen.pt", games=1, seed=42, futures=2, max_samples=2)
    assert result["counts"]["decisions"] > 0
    assert result["counts"]["disagreements"] > 0
    assert result["audited_samples"] == 2
    assert sum(result["by_seat"][str(i)]["decisions"] for i in range(1, 5)) == result["counts"]["decisions"]
    assert all(s["estimated_win_share_delta"] == 0 for s in result["samples"])
    assert all(s["legal_actions"] for s in result["samples"])


def test_audit_without_samples_does_not_run_rollouts(monkeypatch):
    import qwixx_ai.phase10_disagreement as mod

    monkeypatch.setattr(mod, "load", lambda _: object())
    monkeypatch.setattr(mod, "choose", lambda model, game, epsilon: None)
    monkeypatch.setattr(mod, "champion", lambda game: None)
    monkeypatch.setattr(mod, "evaluate_actions",
                        lambda *args, **kwargs: pytest.fail("unexpected rollout"))
    result = audit("frozen.pt", games=1, max_samples=0)
    assert result["audited_samples"] == 0
    assert result["counts"]["disagreements"] == 0


def test_audit_validation():
    with pytest.raises(ValueError):
        audit("frozen.pt", games=0)
    with pytest.raises(ValueError):
        audit("frozen.pt", max_samples=-1)


def test_cli_saves_full_samples_but_prints_summary(tmp_path, monkeypatch, capsys):
    import qwixx_ai.phase10_disagreement as mod

    checkpoint = tmp_path / "candidate.pt"
    checkpoint.touch()
    monkeypatch.setattr(mod, "audit", lambda *args: {
        "phase": "10.2", "samples": [{"seat": 1}], "audited_samples": 1})
    output = tmp_path / "report.json"
    assert main(["--candidate", str(checkpoint), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["samples"] == [{"seat": 1}]
    assert "samples" not in json.loads(capsys.readouterr().out)


def test_reservoir_sampling_is_reproducible_and_not_first_n(monkeypatch):
    import qwixx_ai.phase10_disagreement as mod

    monkeypatch.setattr(mod, "load", lambda _: object())
    monkeypatch.setattr(mod, "choose", lambda model, game, epsilon: None)
    monkeypatch.setattr(
        mod, "champion",
        lambda game: game.legal()[0] if game.legal() else None,
    )
    monkeypatch.setattr(
        mod, "evaluate_actions",
        lambda game, futures, seed: [
            {"action": action, "win_share": 0.25, "mean_score": 30.0}
            for action in [None, *game.legal()]
        ],
    )
    args = {"games": 4, "seed": 42, "futures": 2,
            "max_samples": 3, "sample_seed": 123}
    first = audit("frozen.pt", **args)
    again = audit("frozen.pt", **args)
    assert first == again
    assert first["sampling"] == "uniform_reservoir_without_replacement"
    assert first["audited_samples"] == 3
    assert first["counts"]["disagreements"] > 3
    assert len({(x["game_seed"], x["seat"], x["turn"], x["phase"])
                for x in first["samples"]}) == 3
    assert sum(x["samples"] for x in first["counterfactual_by_category"].values()) == 3
