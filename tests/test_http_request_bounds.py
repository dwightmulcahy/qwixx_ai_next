"""Regression tests for bounded HTTP JSON request parsing."""
import io

import pytest

from qwixx_ai.distributed import Handler


def parse_body(payload, length=None, limit=64):
    handler = object.__new__(Handler)
    handler.rfile = io.BytesIO(payload)
    handler.headers = {} if length is None else {"Content-Length": str(length)}
    return handler.read_json_body(max_bytes=limit)


@pytest.mark.parametrize("payload", [b"[]", b"null", b"1", b'"hello"'])
def test_json_body_requires_object(payload):
    with pytest.raises(ValueError, match="JSON object required"):
        parse_body(payload, len(payload))


def test_json_body_accepts_small_object():
    assert parse_body(b'{"guest":true}', 14) == {"guest": True}


@pytest.mark.parametrize("length", [None, -1, 0, 65, "invalid"])
def test_json_body_rejects_missing_or_invalid_size(length):
    with pytest.raises(ValueError, match="Content-Length|body size"):
        parse_body(b"{}", length)


@pytest.mark.parametrize("payload", [b"{", b"not json", b"\xff"])
def test_json_body_rejects_invalid_json(payload):
    with pytest.raises(ValueError, match="Invalid JSON"):
        parse_body(payload, len(payload))
