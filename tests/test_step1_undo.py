"""Step 2 back navigation restores the complete Step 1 turn state."""
import copy

import pytest

from qwixx_ai.web import HTML, Session


def test_back_restores_white_step_and_history():
    session = Session(difficulty="intermediate")
    initial = copy.deepcopy(session.view())
    assert initial["phase"] == 0
    choice = session.game.legal()[0] if session.game.legal() else None
    session.apply_action(choice, "human")
    session.advance()
    assert session.game.phase == session.game.n
    assert session.view()["can_undo_step1"]
    session.undo_step1()
    restored = session.view()
    assert restored["phase"] == 0
    assert restored["turn"] == initial["turn"]
    assert restored["dice"] == initial["dice"]
    assert restored["boards"] == initial["boards"]
    assert restored["action_history"] == initial["action_history"]
    assert not restored["can_undo_step1"]


def test_back_only_before_step2_committed():
    session = Session(difficulty="intermediate")
    with pytest.raises(ValueError, match="cannot be undone"):
        session.undo_step1()
    session.apply_action(None, "human")
    session.advance()
    session.apply_action(None, "human")
    with pytest.raises(ValueError, match="cannot be undone"):
        session.undo_step1()


def test_back_button_and_api_present():
    assert "← Back to Step 1" in HTML
    assert "state.can_undo_step1" in HTML
    assert "/undo-step1" in HTML
