"""A lifetime-scoped CONNECT tunnel with an exact harness-owned allowlist."""

from __future__ import annotations

from contextlib import contextmanager
import select
import socket
import socketserver
import threading
from typing import Iterator


def allowed_addresses(value: str) -> frozenset[tuple[str, int]]:
    """Only explicit DNS-name/IPv4:port pairs; no wildcards or URL parsing."""
    addresses = set()
    for item in value.split(","):
        if not item.strip():
            continue
        host, separator, port = item.strip().rpartition(":")
        if (
            not separator
            or not host
            or any(
                c not in "abcdefghijklmnopqrstuvwxyz0123456789.-" for c in host.lower()
            )
            or not port.isascii()
            or not port.isdecimal()
            or not 1 <= int(port) <= 65535
        ):
            raise ValueError("Invalid GEPA_HARNESS_ALLOWED_HOSTS; use host:port pairs.")
        addresses.add((host.lower(), int(port)))
    return frozenset(addresses)


@contextmanager
def connect_proxy(addresses: frozenset[tuple[str, int]]) -> Iterator[int]:
    """Never forward HTTP requests, DNS lookups, or bytes to unlisted hosts."""
    stopped = threading.Event()
    connections: set[socket.socket] = set()
    lock = threading.Lock()

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            self.connection.settimeout(5)
            with lock:
                connections.add(self.connection)
            upstream = None
            try:
                line = self.rfile.readline(4097)
                if len(line) > 4096:
                    return
                parts = line.decode("ascii").strip().split(" ")
                if len(parts) != 3 or parts[0] != "CONNECT" or parts[2] != "HTTP/1.1":
                    self.wfile.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
                    return
                target = allowed_addresses(parts[1])
                if len(target) != 1 or not target.issubset(addresses):
                    self.wfile.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
                    return
                total = 0
                while True:
                    header = self.rfile.readline(4097)
                    total += len(header)
                    if not header or total > 16384:
                        return
                    if header == b"\r\n":
                        break
                upstream = socket.create_connection(next(iter(target)), timeout=5)
                with lock:
                    connections.add(upstream)
                self.wfile.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                self.wfile.flush()
                while not stopped.is_set():
                    ready, _, _ = select.select(
                        [self.connection, upstream], [], [], 0.2
                    )
                    for source in ready:
                        data = source.recv(65536)
                        if not data:
                            return
                        (
                            upstream if source is self.connection else self.connection
                        ).sendall(data)
            except (OSError, ValueError, UnicodeError):
                return
            finally:
                with lock:
                    connections.discard(self.connection)
                    if upstream is not None:
                        connections.discard(upstream)
                        upstream.close()

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True
        # Do not buffer ahead of CONNECT's headers: tunneled bytes stay on the socket.

    Handler.rbufsize = 0
    with Server(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
        )
        thread.start()
        try:
            yield server.server_address[1]
        finally:
            stopped.set()
            with lock:
                for connection in connections:
                    try:
                        connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    connection.close()
            server.shutdown()
            thread.join()
