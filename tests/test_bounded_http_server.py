import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from qwixx_ai.distributed import BoundedHTTPServer


def test_saturation_and_recovery():
    entered = threading.Event()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def handle(self):
            entered.set()
            release.wait(3)

    with BoundedHTTPServer(("127.0.0.1", 0), Handler, max_connections=1) as server:
        first, peer = socket.socketpair()
        second, other = socket.socketpair()
        try:
            server.process_request(first, ("local", 1))
            assert entered.wait(2)
            server.process_request(second, ("local", 2))
            other.settimeout(2)
            assert other.recv(1) == b""
            release.set()
            assert server.connection_slots.acquire(timeout=2)
            server.connection_slots.release()
        finally:
            release.set()
            peer.close()
            other.close()


def test_thread_start_failure_releases_slot(monkeypatch):
    def fail(*args):
        raise RuntimeError("thread unavailable")

    monkeypatch.setattr(ThreadingHTTPServer, "process_request", fail)
    with BoundedHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler, max_connections=1) as server:
        with pytest.raises(RuntimeError):
            server.process_request(None, None)
        assert server.connection_slots.acquire(blocking=False)
        server.connection_slots.release()
