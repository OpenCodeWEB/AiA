"""AiA health/status API — the runtime surface marketplace apps will call.

Threading HTTP server with two endpoints:
  GET /health  → liveness probe
  GET /status  → engine + swarm summary (no private data)
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional


class AiAHealthHandler(BaseHTTPRequestHandler):
    engine: Any = None

    def _send(self, code: int, obj: Any) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/health"):
            self._send(200, {"ok": True, "service": "aia-master-engine"})
        elif self.path.startswith("/status"):
            try:
                self._send(200, self.engine.status())
            except Exception as e:  # noqa: BLE001
                self._send(500, {"ok": False, "error": str(e)})
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
        pass


def serve(engine: Any, host: str = "127.0.0.1", port: int = 8686) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), AiAHealthHandler)
    AiAHealthHandler.engine = engine
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
