"""Phase 10.4 training objective and CLI tests."""
import pytest

from qwixx_ai.phase104_train import main


def test_baseline_uses_original_reward_and_no_promotion(tmp_path, monkeypatch):
    import qwixx_ai.phase104_train as module

    checkpoint = tmp_path / "frozen.pt"
    checkpoint.touch()
    calls = []
    monkeypatch.setattr(module, "train", lambda **kwargs: calls.append(kwargs))
    assert main(["--candidate", str(checkpoint), "--variant", "baseline",
                 "--out", str(tmp_path / "baseline")]) == 0
    assert calls[0]["terminal_win_weight"] == 1.0
    assert calls[0]["relative_score_weight"] == 0.0
    assert calls[0]["init_checkpoint"] == checkpoint
    assert "champion_path" not in calls[0]


def test_win_variant_enables_relative_objective(tmp_path, monkeypatch):
    import qwixx_ai.phase104_train as module

    checkpoint = tmp_path / "frozen.pt"
    checkpoint.touch()
    calls = []
    monkeypatch.setattr(module, "train", lambda **kwargs: calls.append(kwargs))
    assert main(["--candidate", str(checkpoint), "--variant", "win",
                 "--win-weight", "3", "--relative-weight", "0.75",
                 "--out", str(tmp_path / "win")]) == 0
    assert calls[0]["terminal_win_weight"] == 3.0
    assert calls[0]["relative_score_weight"] == 0.75


def test_refuses_source_directory_overwrite(tmp_path):
    checkpoint = tmp_path / "frozen.pt"
    checkpoint.touch()
    with pytest.raises(SystemExit):
        main(["--candidate", str(checkpoint), "--variant", "win",
              "--out", str(tmp_path)])


def test_rejects_negative_objective_weight(tmp_path):
    checkpoint = tmp_path / "frozen.pt"
    checkpoint.touch()
    with pytest.raises(SystemExit):
        main(["--candidate", str(checkpoint), "--variant", "win",
              "--relative-weight", "-1"])
