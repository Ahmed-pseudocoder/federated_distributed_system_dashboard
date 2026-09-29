from __future__ import annotations

from dataclasses import asdict, dataclass
from time import time


@dataclass(slots=True)
class ClientRecord:
    client_id: int
    status: str = "STARTING"
    pid: int | None = None
    start_time: float = 0.0
    last_heartbeat: float = 0.0
    reliability: float = 1.0
    recent_latency: float = 0.0
    available: bool = True
    current_round: int = 0
    current_epoch: int = 0
    current_batch: int = 0
    global_step: int = 0
    last_loss: float | None = None
    last_accuracy: float | None = None


class WorkerRegistry:
    def __init__(self, client_ids: list[int]):
        self.clients = {client_id: ClientRecord(client_id) for client_id in client_ids}

    def register(self, client_id: int, pid: int, timestamp: float | None = None) -> None:
        record = self.clients[client_id]
        record.pid = pid
        record.start_time = timestamp or time()
        record.last_heartbeat = timestamp or time()
        record.status = "HEALTHY"
        record.available = True

    def heartbeat(self, client_id: int, payload: dict, received_at: float | None = None) -> None:
        record = self.clients[client_id]
        record.last_heartbeat = received_at or time()
        record.status = payload.get("status", record.status).upper()
        for source, target in (
            ("round", "current_round"),
            ("epoch", "current_epoch"),
            ("batch", "current_batch"),
            ("global_step", "global_step"),
        ):
            if source in payload:
                setattr(record, target, int(payload[source]))

    def telemetry(self, client_id: int, payload: dict) -> None:
        record = self.clients[client_id]
        self.heartbeat(client_id, payload)
        record.last_loss = payload.get("loss")
        record.last_accuracy = payload.get("accuracy")

    def mark_failed(self, client_id: int) -> None:
        record = self.clients[client_id]
        record.status = "FAILED"
        record.available = False
        record.reliability = max(0.0, record.reliability * 0.25)

    def mark_success(self, client_id: int, latency: float) -> None:
        record = self.clients[client_id]
        record.status = "HEALTHY"
        record.available = True
        record.reliability = 0.9 * record.reliability + 0.1
        record.recent_latency = latency if record.recent_latency == 0 else 0.7 * record.recent_latency + 0.3 * latency

    def snapshot(self, client_id: int) -> dict:
        return asdict(self.clients[client_id])

