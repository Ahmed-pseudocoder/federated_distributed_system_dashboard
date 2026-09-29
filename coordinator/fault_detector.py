from __future__ import annotations

from time import time


class FaultDetector:
    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds

    def is_timed_out(self, last_heartbeat: float, now: float | None = None) -> bool:
        current = time() if now is None else now
        return last_heartbeat > 0 and current - last_heartbeat > self.timeout_seconds

    def failed_clients(self, records, pending: set[int], now: float | None = None) -> list[int]:
        return [
            client_id
            for client_id in pending
            if self.is_timed_out(records[client_id].last_heartbeat, now)
        ]
