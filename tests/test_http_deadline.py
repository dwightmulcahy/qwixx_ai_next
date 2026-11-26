import socket
import threading

from qwixx_ai.distributed import BoundedHTTPServer, Handler


def test_trickling_headers_hit_absolute_deadline():
    with BoundedHTTPServer(("127.0.0.1", 0), Handler, max_connections=1) as server:
        server.request_deadline = 0.3
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with socket.create_connection(server.server_address, timeout=2) as client:
                client.sendall(b"GET / HTTP/1.1\r\n")
                for _ in range(20):
                    try:
                        client.sendall(b"x")
                        if not client.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT):
                            break
                    except BlockingIOError:
                        threading.Event().wait(0.03)
                    except OSError:
                        break
                client.settimeout(2)
                assert client.recv(1) == b""
                assert server.connection_slots.acquire(timeout=2)
                server.connection_slots.release()
        finally:
            server.shutdown()
            thread.join(timeout=2)


def test_deadline_handles_already_closed_socket(monkeypatch):
    callbacks = []

    class ImmediateTimer:
        def __init__(self, interval, callback):
            callbacks.append(callback)

        def start(self):
            callbacks[-1]()

        def cancel(self):
            pass

    monkeypatch.setattr(threading, "Timer", ImmediateTimer)
    with BoundedHTTPServer(("127.0.0.1", 0), Handler, max_connections=1) as server:
        request, peer = socket.socketpair()
        request.close()
        peer.close()
        assert server.connection_slots.acquire(blocking=False)
        monkeypatch.setattr(
            "http.server.ThreadingHTTPServer.process_request_thread",
            lambda *args: None,
        )
        server.process_request_thread(request, ("local", 1))
        assert server.connection_slots.acquire(blocking=False)
        server.connection_slots.release()
