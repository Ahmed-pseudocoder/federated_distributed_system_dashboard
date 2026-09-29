from __future__ import annotations

from coordinator.registry import ClientRecord


class DynamicScheduler:
    @staticmethod
    def score(record: ClientRecord) -> float:
        latency_score = 1.0 / (1.0 + max(0.0, record.recent_latency))
        return record.reliability * 0.75 + latency_score * 0.25

    def select(self, records: list[ClientRecord], limit: int | None = None) -> list[int]:
        eligible = [record for record in records if record.available and record.status != "FAILED"]
        eligible.sort(key=lambda record: (-self.score(record), record.client_id))
        if limit is not None:
            eligible = eligible[:limit]
        return [record.client_id for record in eligible]

