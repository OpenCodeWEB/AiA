"""AiA neural API - the live surface of the self-evolving brain.

Threading HTTP server exposing the SelfEvolvingNN as JSON endpoints so the
AiA engine (and any local tooling) can train, query and evolve the network:

  GET  /health                liveness probe
  GET  /status                network snapshot (flat primitives)
  GET  /predict?input=a,b,c   one-shot prediction
  POST /train   {"inputs":[..],"target":[..]}  online training step
  POST /replay  {"batch":16}  experience replay rehearsal
  POST /evolve  {"reason":"manual"}  force an architecture evolution
  POST /reset   {"input":2,"hidden":[6],"output":1}  fresh brain
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .self_evolving_nn import SelfEvolvingNN


class NeuralHandler(BaseHTTPRequestHandler):
    nn: SelfEvolvingNN = SelfEvolvingNN()

    def _send(self, code: int, obj: Any) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-ABsUP-Auth", "neural")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    def do_GET(self) -> None:  # noqa: N802
        try:
            if self.path.startswith("/health"):
                self._send(200, {"ok": True, "service": "aia-neural"})
            elif self.path.startswith("/status"):
                self._send(200, self.nn.status())
            elif self.path.startswith("/predict"):
                query = self.path.split("?", 1)[-1]
                parts = query.split("input=", 1)[-1].split("&", 1)[0]
                try:
                    x = [float(v) for v in parts.split(",") if v != ""]
                except ValueError:
                    self._send(400, {"ok": False, "error": "input must be comma-separated numbers"})
                    return
                if len(x) != self.nn.input_size:
                    self._send(400, {"ok": False, "error": f"expected {self.nn.input_size} inputs"})
                    return
                self._send(200, {"ok": True, "input": x, "output": self.nn.forward(x), "status": self.nn.status()})
            else:
                self._send(404, {"ok": False, "error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"ok": False, "error": str(e)})

    def do_POST(self) -> None:  # noqa: N802
        try:
            data = self._read_json()
            if self.path.startswith("/train"):
                inputs = data.get("inputs")
                target = data.get("target")
                if not inputs or not target or len(inputs) != self.nn.input_size or len(target) != self.nn.output_size:
                    self._send(400, {"ok": False, "error": "inputs/target size mismatch"})
                    return
                loss = self.nn.train([float(v) for v in inputs], [float(v) for v in target])
                self._send(200, {"ok": True, "loss": round(loss, 6), "status": self.nn.status()})
            elif self.path.startswith("/replay"):
                batch = int(data.get("batch", 16))
                avg = self.nn.replay_pass(batch)
                self._send(200, {"ok": True, "avg_loss": round(avg, 6), "status": self.nn.status()})
            elif self.path.startswith("/evolve"):
                event = self.nn.evolve(reason=str(data.get("reason", "manual")))
                self._send(200, {"ok": True, "event": event, "status": self.nn.status()})
            elif self.path.startswith("/reset"):
                self.nn = SelfEvolvingNN(
                    input_size=int(data.get("input", 2)),
                    hidden=[int(v) for v in data.get("hidden", [6])],
                    output_size=int(data.get("output", 1)),
                )
                NeuralHandler.nn = self.nn
                self._send(200, {"ok": True, "status": self.nn.status()})
            else:
                self._send(404, {"ok": False, "error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"ok": False, "error": str(e)})

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
        pass


def serve(host: str = "127.0.0.1", port: int = 9095) -> ThreadingHTTPServer:
    """Start the neural API in a daemon thread. Port 9095 = absup:neural."""
    server = ThreadingHTTPServer((host, port), NeuralHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
