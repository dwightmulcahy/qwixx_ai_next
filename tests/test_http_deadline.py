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
                assert server.connection_slots.acquire(timeout=2)
                server.connection_slots.release()
        finally:
            server.shutdown()
            thread.join(timeout=2)
