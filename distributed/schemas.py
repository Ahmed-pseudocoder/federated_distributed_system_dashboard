from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RegisterRequest(BaseModel):
    client_id: int = Field(ge=1)
    pid: int
    hostname: str
    platform: str
    sample_count: int = Field(ge=1)
    distribution: dict[str, int]


class HeartbeatRequest(BaseModel):
    client_id: int = Field(ge=1)
    pid: int | None = None
    round: int = 0
    epoch: int = 0
    batch: int = 0
    global_step: int = 0
    status: str = "healthy"
    timestamp: str | None = None


class TelemetryRequest(HeartbeatRequest):
    loss: float | None = None
    accuracy: float | None = None


class ClientEventRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    kind: str = "event"
    timestamp: str | None = None
    event_type: str
    source: str
    client_id: int = Field(ge=1)
    round: int | None = None


class UpdateRequest(BaseModel):
    client_id: int = Field(ge=1)
    round: int = Field(ge=1)
    task_id: str
    sample_count: int = Field(ge=1)
    metrics: dict[str, Any]
    state_dict: str
    complete: bool = True
