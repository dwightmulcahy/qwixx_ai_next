"""Verify interactive inference admission and slot recovery."""
from threading import BoundedSemaphore
from unittest.mock import Mock

from qwixx_ai.distributed import Handler


def handler(monkeypatch):
    monkeypatch.setattr(Handler, "inference_slots", BoundedSemaphore(1))
    request = object.__new__(Handler)
    request.path = "/reset"
    request.game_limited = Mock(return_value=False)
    request.reply = Mock()
    request.read_json_body = Mock(return_value={"guest": False})
    request.player_id = Mock(return_value="player")
    request.coord = Mock()
    request.coord.new_session.return_value = {"session_id": "game"}
    return request


def test_busy_game_request_does_not_start_inference(monkeypatch):
    request = handler(monkeypatch)
    assert request.inference_slots.acquire(blocking=False)
    try:
        request.do_POST()
        request.reply.assert_called_once_with({"error": "Game server busy; retry shortly"}, 503)
        request.coord.new_session.assert_not_called()
    finally:
        request.inference_slots.release()


def test_success_releases_inference_slot(monkeypatch):
    request = handler(monkeypatch)
    request.do_POST()
    request.reply.assert_called_once_with({"session_id": "game"})
    assert request.inference_slots.acquire(blocking=False)
    request.inference_slots.release()


def test_validation_error_releases_inference_slot(monkeypatch):
    request = handler(monkeypatch)
    request.read_json_body.side_effect = ValueError("Invalid JSON")
    request.do_POST()
    request.reply.assert_called_once_with({"error": "Invalid JSON"}, 400)
    assert request.inference_slots.acquire(blocking=False)
    request.inference_slots.release()


def test_body_read_does_not_hold_inference_slot(monkeypatch):
    request = handler(monkeypatch)

    def read_body(**kwargs):
        assert request.inference_slots.acquire(blocking=False)
        request.inference_slots.release()
        return {"guest": False}

    request.read_json_body.side_effect = read_body
    request.do_POST()
    request.coord.new_session.assert_called_once()


def test_coordinator_error_releases_acquired_slot(monkeypatch):
    request = handler(monkeypatch)
    request.coord.new_session.side_effect = RuntimeError("Inference failed")
    request.do_POST()
    request.reply.assert_called_once_with({"error": "Inference failed"}, 400)
    assert request.inference_slots.acquire(blocking=False)
    request.inference_slots.release()


def test_foreign_session_rejection_releases_slot(monkeypatch):
    request = handler(monkeypatch)
    request.path = "/move"
    request.read_json_body.return_value = {"session_id": "foreign", "action": None}
    request.coord.sessions = {"foreign": {"owner": "other"}}
    request.owns_session = Mock(return_value=False)
    request.do_POST()
    request.reply.assert_called_once_with({"error": "Session belongs to another player"}, 403)
    request.coord.session_action.assert_not_called()
    assert request.inference_slots.acquire(blocking=False)
    request.inference_slots.release()


def test_undo_releases_slot(monkeypatch):
    request = handler(monkeypatch)
    request.path = "/undo-step1"
    request.read_json_body.return_value = {"session_id": "game", "expected_turn": 2, "expected_phase": 1}
    request.coord.sessions = {}
    request.coord.session_undo_step1.return_value = {"session_id": "game"}
    request.do_POST()
    request.coord.session_undo_step1.assert_called_once_with("game", 2, 1)
    request.reply.assert_called_once_with({"session_id": "game"})
    assert request.inference_slots.acquire(blocking=False)
    request.inference_slots.release()
