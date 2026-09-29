from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(slots=True)
class Telemetry:
    client_id: int
    round: int
    epoch: int
    batch: int
    global_step: int
    loss: float | None
    accuracy: float | None
    timestamp: str
    status: str
    pid: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def message(kind: str, **payload: Any) -> dict[str, Any]:
    return {"kind": kind, "timestamp": utc_now(), **payload}
