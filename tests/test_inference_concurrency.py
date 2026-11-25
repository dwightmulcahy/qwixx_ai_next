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
