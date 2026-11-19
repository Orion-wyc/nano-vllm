from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _BootstrapHandler(BaseHTTPRequestHandler):

    server_version = "nano-vllm-mooncake/1.0"

    def do_GET(self):
        if self.path == "/health":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/query":
            body = json.dumps(
                {"0": {"engine_id": self.server.engine_id}},
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def log_message(self, format, *args):
        return


class MooncakeBootstrapServer:

    def __init__(self, engine_id: str, port: int, host: str = "127.0.0.1"):
        self.engine_id = engine_id
        self.port = port
        self.host = host
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self):
        if self._server is not None:
            return
        server = ThreadingHTTPServer((self.host, self.port), _BootstrapHandler)
        server.daemon_threads = True
        server.engine_id = self.engine_id
        thread = threading.Thread(
            target=server.serve_forever,
            name="MooncakeBootstrap",
            daemon=True,
        )
        thread.start()
        self._server = server
        self._thread = thread

    def stop(self):
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None
