"""TLS-only probe launched under the scoring child's profile and environment."""

from __future__ import annotations

import json
import socket
import ssl
import sys


def handshake(host: str, port: int, proxy_port: int) -> None:
    try:
        from httpx2._config import create_ssl_context
    except ImportError:
        context = ssl.create_default_context()
    else:
        context = create_ssl_context()
    with socket.create_connection(("127.0.0.1", proxy_port), timeout=5) as connection:
        connection.sendall(
            f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode(
                "ascii"
            )
        )
        response = bytearray()
        while not response.endswith(b"\r\n\r\n"):
            block = connection.recv(1)
            if not block or len(response) >= 16384:
                raise ConnectionError("Incomplete proxy response")
            response.extend(block)
        if not response.startswith(b"HTTP/1.1 200 "):
            raise ConnectionError("Proxy refused CONNECT")
        with context.wrap_socket(connection, server_hostname=host):
            pass


def main() -> None:
    try:
        handshake(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    except Exception as error:
        print(json.dumps({"error_class": type(error).__name__}))
        raise SystemExit(1)
    print(json.dumps({"error_class": None}))
