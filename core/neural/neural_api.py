"""AiA neural API - the live surface of the self-evolving brain.

Threading HTTP server exposing the SelfEvolvingNN and its companion neural
modules as JSON endpoints so the AiA engine (and any local tooling) can
train, query, evolve and inspect the network:

  GET  /health                liveness probe
  GET  /status                network snapshot (flat primitives)
  GET  /modules               status of every neural module
  GET  /predict?input=a,b,c   one-shot prediction
  POST /train   {"inputs":[..],"target":[..]}  online training step
  POST /replay  {"batch":16}  experience replay rehearsal
  POST /evolve  {"reason":"manual"}  force an architecture evolution
  POST /reset   {"input":2,"hidden":[6],"output":1}  fresh brain

  POST /compress_memory       compress the replay buffer into latent vectors
  POST /anomaly/train         teach the anomaly detector a normal sample
  POST /anomaly/check         score one input for anomalies
  POST /curriculum/update     feed a training outcome to the curriculum
  POST /memory/store          store a pattern in associative memory
  POST /memory/recall         reconstruct a corrupted pattern
  POST /predict_sequence      forecast the next steps of a sequence
  POST /federated/sync        publish a signed weight delta to the mesh
  POST /federated/receive     ingest a signed update from another node
  POST /federated/aggregate   FedAvg-merge queued updates into the brain
  POST /rl/act                epsilon-greedy action selection
  POST /rl/step               environment transition -> replay learning
  POST /attention/weights     scaled dot-product attention map + context
  POST /attention/train       online attention projection training
  GET  /spatial/coords        this node's (x,y,z) grid position + signed metadata
  POST /spatial/register      register a peer on the 3D grid (signed block)
  POST /spatial/competence    feed module mastery scores (steers the X axis)
  POST /spatial/cluster/heartbeat   announce a peer in this Y-region
  GET  /spatial/cluster       cluster status + elected regional anchor
  POST /spatial/sync/pulse    ingest a spatially filtered signed weight delta
  GET  /spatial/sync          spatial sync status (pulses, rejections, queue)
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .anomaly_detector import AnomalyDetector
from .attention_layer import AttentionLayer
from .autoencoder_compressor import AutoencoderCompressor
from .cluster_heartbeat_election import ClusterHeartbeatElection
from .curriculum_controller import CurriculumController
from .dqn_agent import DQNAgent
from .federated_sync import FederatedSync
from .hopfield_memory import HopfieldMemory
from .rnn_sequence import ElmanRNN
from .self_evolving_nn import SelfEvolvingNN
from .spatial_grid_coordinator import SpatialGridCoordinator
from .spatial_temporal_sync import SpatialTemporalSync


def _f64(value: Any) -> float:
    return float(value)


def _f64_list(values: Any) -> list[float]:
    return [_f64(v) for v in values]


class NeuralHandler(BaseHTTPRequestHandler):
    nn: SelfEvolvingNN = SelfEvolvingNN()
    compressor: AutoencoderCompressor = AutoencoderCompressor(input_size=2, bottleneck=1, hidden=6)
    anomaly: AnomalyDetector = AnomalyDetector(input_size=2, bottleneck=1, hidden=6)
    curriculum: CurriculumController = CurriculumController()
    memory: HopfieldMemory = HopfieldMemory(size=16)
    rnn: ElmanRNN = ElmanRNN(input_size=2, hidden_size=8, output_size=1)
    federated: FederatedSync = FederatedSync()
    dqn: DQNAgent = DQNAgent(state_size=2, action_size=3)
    attention: AttentionLayer = AttentionLayer(d_model=4)
    spatial: SpatialGridCoordinator = SpatialGridCoordinator()
    cluster: ClusterHeartbeatElection = ClusterHeartbeatElection()
    spatial_sync: SpatialTemporalSync = SpatialTemporalSync()
    lock: threading.RLock = threading.RLock()

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

    # ------------------------------------------------------------------ #
    # GET
    # ------------------------------------------------------------------ #
    def do_GET(self) -> None:  # noqa: N802
        try:
            if self.path.startswith("/health"):
                self._send(200, {"ok": True, "service": "aia-neural"})
            elif self.path.startswith("/modules"):
                with NeuralHandler.lock:
                    self._send(
                        200,
                        {
                            "brain": self.nn.status(),
                            "compressor": self.compressor.status(),
                            "anomaly": self.anomaly.status(),
                            "curriculum": self.curriculum.status(),
                            "memory": self.memory.status(),
                            "rnn": self.rnn.status(),
                            "federated": self.federated.status(),
                            "dqn": self.dqn.status(),
                            "attention": self.attention.status(),
                            "spatial": self.spatial.status(),
                            "cluster": self.cluster.status(),
                            "spatial_sync": self.spatial_sync.status(),
                        },
                    )
            elif self.path.startswith("/spatial/coords"):
                with NeuralHandler.lock:
                    block = self.spatial.register(self.spatial.node_id)
                    self._send(200, {"ok": True, **block, "spatial": self.spatial.status()})
            elif self.path.startswith("/spatial/cluster"):
                with NeuralHandler.lock:
                    self._send(200, {"ok": True, **self.cluster.status()})
            elif self.path.startswith("/spatial/sync"):
                with NeuralHandler.lock:
                    self.spatial_sync.set_coords(self.spatial.coords())
                    self._send(200, {"ok": True, **self.spatial_sync.status()})
            elif self.path.startswith("/status"):
                with NeuralHandler.lock:
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
                with NeuralHandler.lock:
                    self._send(200, {"ok": True, "input": x, "output": self.nn.forward(x), "status": self.nn.status()})
            else:
                self._send(404, {"ok": False, "error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"ok": False, "error": str(e)})

    # ------------------------------------------------------------------ #
    # POST
    # ------------------------------------------------------------------ #
    def do_POST(self) -> None:  # noqa: N802
        try:
            data = self._read_json()
            if self.path.startswith("/train"):
                inputs = data.get("inputs")
                target = data.get("target")
                if not inputs or not target or len(inputs) != self.nn.input_size or len(target) != self.nn.output_size:
                    self._send(400, {"ok": False, "error": "inputs/target size mismatch"})
                    return
                with NeuralHandler.lock:
                    loss = self.nn.train(_f64_list(inputs), _f64_list(target))
                    self._send(200, {"ok": True, "loss": round(loss, 6), "status": self.nn.status()})
            elif self.path.startswith("/replay"):
                batch = int(data.get("batch", 16))
                with NeuralHandler.lock:
                    avg = self.nn.replay_pass(batch)
                    self._send(200, {"ok": True, "avg_loss": round(avg, 6), "status": self.nn.status()})
            elif self.path.startswith("/evolve"):
                with NeuralHandler.lock:
                    event = self.nn.evolve(reason=str(data.get("reason", "manual")))
                    self._send(200, {"ok": True, "event": event, "status": self.nn.status()})
            elif self.path.startswith("/reset"):
                with NeuralHandler.lock:
                    self.nn = SelfEvolvingNN(
                        input_size=int(data.get("input", 2)),
                        hidden=[int(v) for v in data.get("hidden", [6])],
                        output_size=int(data.get("output", 1)),
                    )
                    NeuralHandler.nn = self.nn
                    self._send(200, {"ok": True, "status": self.nn.status()})
            elif self.path.startswith("/compress_memory"):
                with NeuralHandler.lock:
                    # compress the brain's replay buffer into latent vectors
                    samples = [list(x) for x, _ in self.nn.replay]
                    if samples:
                        avg = self.compressor.train_batch(samples[: min(len(samples), 256)])
                        latent = self.compressor.compress(samples)
                    else:
                        avg, latent = 0.0, []
                    self._send(
                        200,
                        {
                            "ok": True,
                            "samples_compressed": len(latent),
                            "latent_dim": self.compressor.bottleneck,
                            "compression_ratio": self.compressor.status()["compression_ratio"],
                            "avg_reconstruction_loss": round(avg, 6),
                            "compressed_vectors": [list(map(lambda v: round(v, 6), z)) for z in latent],
                        },
                    )
            elif self.path.startswith("/anomaly/train"):
                inputs = data.get("inputs")
                if not inputs or len(inputs) != self.anomaly.input_size:
                    self._send(400, {"ok": False, "error": f"expected {self.anomaly.input_size} inputs"})
                    return
                with NeuralHandler.lock:
                    loss = self.anomaly.train(_f64_list(inputs))
                    self._send(200, {"ok": True, "loss": round(loss, 6), "anomaly": self.anomaly.status()})
            elif self.path.startswith("/anomaly/check"):
                inputs = data.get("inputs")
                if not inputs or len(inputs) != self.anomaly.input_size:
                    self._send(400, {"ok": False, "error": f"expected {self.anomaly.input_size} inputs"})
                    return
                with NeuralHandler.lock:
                    result = self.anomaly.check(_f64_list(inputs))
                    self._send(200, {"ok": True, **result})
            elif self.path.startswith("/curriculum/update"):
                inputs = data.get("inputs")
                loss = data.get("loss")
                if not inputs or loss is None:
                    self._send(400, {"ok": False, "error": "inputs and loss are required"})
                    return
                with NeuralHandler.lock:
                    state = self.curriculum.update(_f64_list(inputs), _f64(loss))
                    self._send(200, {"ok": True, **state})
            elif self.path.startswith("/memory/store"):
                pattern = data.get("pattern")
                if not pattern or len(pattern) != self.memory.size:
                    self._send(400, {"ok": False, "error": f"pattern must have {self.memory.size} elements"})
                    return
                with NeuralHandler.lock:
                    idx = self.memory.store(_f64_list(pattern))
                    self._send(200, {"ok": True, "index": idx, "memory": self.memory.status()})
            elif self.path.startswith("/memory/recall"):
                corrupted = data.get("corrupted")
                if not corrupted or len(corrupted) != self.memory.size:
                    self._send(400, {"ok": False, "error": f"corrupted must have {self.memory.size} elements"})
                    return
                with NeuralHandler.lock:
                    result = self.memory.recall(_f64_list(corrupted))
                    self._send(200, {"ok": True, **result, "memory": self.memory.status()})
            elif self.path.startswith("/predict_sequence"):
                sequence = data.get("sequence")
                if not sequence or not all(isinstance(s, (list, tuple)) for s in sequence):
                    self._send(400, {"ok": False, "error": "sequence must be a list of vectors"})
                    return
                seq = [_f64_list(s) for s in sequence]
                if any(len(s) != self.rnn.input_size for s in seq):
                    self._send(400, {"ok": False, "error": f"expected vectors of {self.rnn.input_size}"})
                    return
                steps_ahead = max(1, int(data.get("steps_ahead", 1)))
                with NeuralHandler.lock:
                    preds = self.rnn.predict_sequence(seq, steps_ahead)
                    self._send(
                        200,
                        {
                            "ok": True,
                            "predictions": [list(map(lambda v: round(v, 6), p)) for p in preds],
                            "rnn": self.rnn.status(),
                        },
                    )
            elif self.path.startswith("/rnn/train"):
                sequence = data.get("sequence")
                targets = data.get("targets")
                if not sequence or not targets or len(sequence) != len(targets):
                    self._send(400, {"ok": False, "error": "sequence and targets must match"})
                    return
                seq = [_f64_list(s) for s in sequence]
                tgt = [_f64_list(t) for t in targets]
                if any(len(s) != self.rnn.input_size for s in seq):
                    self._send(400, {"ok": False, "error": f"expected vectors of {self.rnn.input_size}"})
                    return
                with NeuralHandler.lock:
                    loss = self.rnn.train_sequence(seq, tgt)
                    self._send(200, {"ok": True, "loss": round(loss, 6), "rnn": self.rnn.status()})
            elif self.path.startswith("/federated/sync"):
                sample_count = max(1, int(data.get("sample_count", 1)))
                with NeuralHandler.lock:
                    update = self.federated.make_update(self.nn.weights, self.nn.biases, sample_count)
                    if update is None:
                        self._send(200, {"ok": True, "published": False, "note": "brain delta below threshold"})
                        return
                    self._send(
                        200,
                        {
                            "ok": True,
                            "published": True,
                            "node_id": update["node_id"],
                            "round": update["round"],
                            "sample_count": update["sample_count"],
                            "signature": update["signature"][:16] + "...",
                            "federated": self.federated.status(),
                        },
                    )
            elif self.path.startswith("/federated/receive"):
                update = data.get("update")
                if not isinstance(update, dict):
                    self._send(400, {"ok": False, "error": "update must be an object"})
                    return
                with NeuralHandler.lock:
                    accepted = self.federated.receive(update)
                    self._send(200, {"ok": True, "accepted": accepted, "federated": self.federated.status()})
            elif self.path.startswith("/federated/aggregate"):
                with NeuralHandler.lock:
                    report = self.federated.aggregate(self.nn.weights, self.nn.biases)
                    self._send(200, {"ok": True, **report, "federated": self.federated.status()})
            elif self.path.startswith("/rl/act"):
                state = data.get("state")
                if not state or len(state) != self.dqn.state_size:
                    self._send(400, {"ok": False, "error": f"expected {self.dqn.state_size} state values"})
                    return
                with NeuralHandler.lock:
                    result = self.dqn.act(_f64_list(state))
                    self._send(200, {"ok": True, **result, "dqn": self.dqn.status()})
            elif self.path.startswith("/rl/step"):
                state = data.get("state")
                next_state = data.get("next_state")
                action = data.get("action")
                reward = data.get("reward")
                if (
                    not state
                    or not next_state
                    or len(state) != self.dqn.state_size
                    or len(next_state) != self.dqn.state_size
                    or action is None
                    or reward is None
                ):
                    self._send(400, {"ok": False, "error": "state/next_state/action/reward required"})
                    return
                with NeuralHandler.lock:
                    result = self.dqn.step(
                        _f64_list(state),
                        int(action),
                        _f64(reward),
                        _f64_list(next_state),
                        bool(data.get("done", False)),
                    )
                    self._send(200, {"ok": True, **result, "dqn": self.dqn.status()})
            elif self.path.startswith("/attention/train"):
                matrix = data.get("input_matrix")
                if not matrix or not all(isinstance(row, (list, tuple)) for row in matrix):
                    self._send(400, {"ok": False, "error": "input_matrix must be a list of vectors"})
                    return
                mat = [_f64_list(row) for row in matrix]
                if any(len(row) != self.attention.d_model for row in mat):
                    self._send(400, {"ok": False, "error": f"expected vectors of {self.attention.d_model}"})
                    return
                with NeuralHandler.lock:
                    result = self.attention.train(mat, steps=max(1, int(data.get("steps", 20))))
                    self._send(200, {"ok": True, **result, "attention": self.attention.status()})
            elif self.path.startswith("/attention/weights"):
                matrix = data.get("input_matrix")
                if not matrix or not all(isinstance(row, (list, tuple)) for row in matrix):
                    self._send(400, {"ok": False, "error": "input_matrix must be a list of vectors"})
                    return
                mat = [_f64_list(row) for row in matrix]
                if any(len(row) != self.attention.d_model for row in mat):
                    self._send(400, {"ok": False, "error": f"expected vectors of {self.attention.d_model}"})
                    return
                with NeuralHandler.lock:
                    result = self.attention.forward(mat)
                    self._send(200, {"ok": True, **result, "attention": self.attention.status()})
            elif self.path.startswith("/spatial/register"):
                peer_id = data.get("peer_id")
                if not peer_id or not isinstance(peer_id, str):
                    self._send(400, {"ok": False, "error": "peer_id is required"})
                    return
                with NeuralHandler.lock:
                    block = self.spatial.register(peer_id)
                    self._send(200, {"ok": True, **block, "spatial": self.spatial.status()})
            elif self.path.startswith("/spatial/competence"):
                scores = data.get("scores")
                if not isinstance(scores, dict):
                    self._send(400, {"ok": False, "error": "scores must be an object of module->mastery"})
                    return
                with NeuralHandler.lock:
                    for module, score in scores.items():
                        try:
                            self.spatial.update_competence(str(module), _f64(score))
                        except (TypeError, ValueError):
                            self._send(400, {"ok": False, "error": f"score for {module} must be a number"})
                            return
                    self._send(200, {"ok": True, **self.spatial.spatial_metadata(), "spatial": self.spatial.status()})
            elif self.path.startswith("/spatial/cluster/heartbeat"):
                peer_id = data.get("peer_id")
                coords = data.get("coords")
                if not peer_id or not isinstance(coords, dict):
                    self._send(400, {"ok": False, "error": "peer_id and coords are required"})
                    return
                with NeuralHandler.lock:
                    result = self.cluster.heartbeat(
                        str(peer_id),
                        {"x": coords.get("x", 0), "y": coords.get("y", 0), "z": coords.get("z", 0)},
                        stage=int(coords.get("z", 0)),
                    )
                    self._send(200, {"ok": True, **result, "cluster": self.cluster.status()})
            elif self.path.startswith("/spatial/sync/pulse"):
                peer_id = data.get("peer_id")
                coords = data.get("coords")
                update = data.get("update")
                if not peer_id or not isinstance(coords, dict) or not isinstance(update, dict):
                    self._send(400, {"ok": False, "error": "peer_id, coords and update are required"})
                    return
                with NeuralHandler.lock:
                    self.spatial_sync.set_coords(self.spatial.coords())
                    accepted = self.spatial_sync.ingest(
                        str(peer_id),
                        {"x": coords.get("x", 0), "y": coords.get("y", 0), "z": coords.get("z", 0)},
                        update,
                        secret=self.federated.secret,
                    )
                    self._send(200, {"ok": True, "accepted": accepted, "spatial_sync": self.spatial_sync.status()})
            elif self.path.startswith("/spatial/sync/aggregate"):
                with NeuralHandler.lock:
                    self.spatial_sync.set_coords(self.spatial.coords())
                    report = self.spatial_sync.aggregate(self.nn.weights, self.nn.biases)
                    self._send(200, {"ok": True, **report, "spatial_sync": self.spatial_sync.status()})
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
