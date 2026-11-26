import socket
import threading
from http.server import ThreadingHTTPServer

from qwixx_ai.distributed import Handler


def test_idle_partial_headers_close_connection():
    class FastTimeoutHandler(Handler):
        timeout = 0.1

    with ThreadingHTTPServer(("127.0.0.1", 0), FastTimeoutHandler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with socket.create_connection(server.server_address, timeout=2) as client:
                client.sendall(b"GET / HTTP/1.1\r\nHost: localhost\r\n")
                assert client.recv(1024) == b""
        finally:
            server.shutdown()
            thread.join(timeout=2)
