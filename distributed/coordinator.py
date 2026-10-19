from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from torch import nn

from coordinator.aggregator import weighted_fedavg
from coordinator.checkpoint import CheckpointManager
from coordinator.event_logger import EventLogger
from coordinator.fault_detector import FaultDetector
from coordinator.registry import WorkerRegistry
from coordinator.scheduler import DynamicScheduler
from core.data import ensure_mnist_available, load_test_data, set_global_seed
from core.messages import utc_now
from core.model import create_model
from core.serialization import cpu_state_dict
from distributed.codec import decode_state_dict, encode_state_dict
from distributed.schemas import (
    ClientEventRequest,
    HeartbeatRequest,
    RegisterRequest,
    TelemetryRequest,
    UpdateRequest,
)


class NetworkCoordinator:
    def __init__(self, config, shared_token: str):
        self.config = config
        self.shared_token = shared_token
        self.config.ensure_directories()
        self.events = EventLogger(config.database_path)
        self.registry = WorkerRegistry(list(range(1, config.num_clients + 1)))
        self.scheduler = DynamicScheduler()
        self.detector = FaultDetector(config.heartbeat_timeout)
        self.checkpoints = CheckpointManager(config.checkpoint_dir)
        self.model = create_model(config.seed)
        self.test_loader = None
        self.latest_run_checkpoint: Path | None = None
        self.registered: set[int] = set()
        self.worker_metadata: dict[int, dict[str, Any]] = {}
        self.tasks: dict[int, dict[str, Any]] = {}
        self.updates: dict[int, dict] = {}
        self.accepted_tasks: set[str] = set()
        self.current_round = 0
        self.stop_event = threading.Event()
        self.training_thread: threading.Thread | None = None
        self.condition = threading.Condition(threading.RLock())
        self.logger = self._create_logger()

    def _create_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"fl.network.coordinator.{os.getpid()}")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            file_handler = logging.FileHandler(
                self.config.log_dir / "network_coordinator.log", encoding="utf-8"
            )
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        logger.propagate = False
        return logger

    def _validate_client_id(self, client_id: int) -> None:
        if client_id not in self.registry.clients:
            raise ValueError(f"client_id must be between 1 and {self.config.num_clients}")

    def _persist_worker(self, client_id: int) -> None:
        payload = self.registry.snapshot(client_id)
        payload.update(self.worker_metadata.get(client_id, {}))
        payload["transport"] = "http"
        self.events.upsert_worker(client_id, payload)

    def public_worker_config(self, client_id: int) -> dict:
        self._validate_client_id(client_id)
        return {
            "client_id": client_id,
            "num_clients": self.config.num_clients,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "local_epochs": self.config.local_epochs,
            "max_train_samples_per_client": self.config.max_train_samples_per_client,
            "seed": self.config.seed,
            "heartbeat_interval": self.config.heartbeat_interval,
            "telemetry_every_batches": self.config.telemetry_every_batches,
        }

    def register(self, payload: RegisterRequest, remote_addr: str | None) -> dict:
        self._validate_client_id(payload.client_id)
        now = time.time()
        with self.condition:
            self.registry.register(payload.client_id, payload.pid, now)
            self.registered.add(payload.client_id)
            self.worker_metadata[payload.client_id] = {
                "hostname": payload.hostname,
                "platform": payload.platform,
                "remote_addr": remote_addr,
                "sample_count": payload.sample_count,
                "distribution": payload.distribution,
            }
            self._persist_worker(payload.client_id)
            self.events.log_event(
                "CLIENT_CONNECTED",
                source=f"client-{payload.client_id}",
                client_id=payload.client_id,
                pid=payload.pid,
                hostname=payload.hostname,
                remote_addr=remote_addr,
                transport="http",
                sample_count=payload.sample_count,
                distribution=payload.distribution,
            )
            self.condition.notify_all()
        self.logger.info(
            "Client %s registered from %s host=%s pid=%s",
            payload.client_id,
            remote_addr,
            payload.hostname,
            payload.pid,
        )
        return {"registered": True, "client_id": payload.client_id}

    def heartbeat(self, payload: HeartbeatRequest) -> dict:
        self._validate_client_id(payload.client_id)
        with self.condition:
            if payload.client_id not in self.registered:
                raise ValueError("client must register before sending heartbeats")
            values = payload.model_dump()
            self.registry.heartbeat(payload.client_id, values)
            self._persist_worker(payload.client_id)
            self.events.log_event(
                "CLIENT_HEARTBEAT",
                source=f"client-{payload.client_id}",
                client_id=payload.client_id,
                round_number=payload.round,
                epoch=payload.epoch,
                batch=payload.batch,
                global_step=payload.global_step,
                status=payload.status,
                pid=payload.pid,
            )
            self.condition.notify_all()
        return {"accepted": True}

    def telemetry(self, payload: TelemetryRequest) -> dict:
        self._validate_client_id(payload.client_id)
        values = payload.model_dump()
        values["timestamp"] = values.get("timestamp") or utc_now()
        with self.condition:
            self.registry.telemetry(payload.client_id, values)
            self.events.record_telemetry(values)
            self._persist_worker(payload.client_id)
            self.condition.notify_all()
        return {"accepted": True}

    def client_event(self, payload: ClientEventRequest) -> dict:
        values = payload.model_dump()
        reserved = {"kind", "timestamp", "event_type", "source", "client_id", "round"}
        details = {key: value for key, value in values.items() if key not in reserved}
        with self.condition:
            self.registry.heartbeat(payload.client_id, values)
            self._persist_worker(payload.client_id)
            self.events.log_event(
                payload.event_type,
                source=payload.source,
                client_id=payload.client_id,
                round_number=payload.round,
                timestamp=payload.timestamp,
                **details,
            )
            self.condition.notify_all()
        return {"accepted": True}

    def get_task(self, client_id: int) -> dict:
        self._validate_client_id(client_id)
        with self.condition:
            if client_id not in self.registered:
                raise ValueError("client must register before polling for work")
            task = self.tasks.get(client_id)
            if task and task["status"] == "pending":
                return {key: value for key, value in task.items() if key != "status"} | {"action": "train"}
            current = self._current_state()
            if current.get("status") in {"COMPLETED", "FAILED"}:
                return {"action": "shutdown", "training_status": current.get("status")}
            return {"action": "wait", "poll_after": 1.0}

    def submit_update(self, payload: UpdateRequest) -> dict:
        self._validate_client_id(payload.client_id)
        if not payload.complete:
            raise ValueError("incomplete updates are not accepted")
        state = decode_state_dict(payload.state_dict)
        with self.condition:
            if payload.task_id in self.accepted_tasks:
                return {"accepted": True, "duplicate": True}
            task = self.tasks.get(payload.client_id)
            if not task or task.get("task_id") != payload.task_id:
                raise ValueError("task_id does not match the assigned task")
            if payload.round != self.current_round or payload.round != task.get("round"):
                raise ValueError("update belongs to a stale or future round")
            update = {
                "state_dict": state,
                "sample_count": payload.sample_count,
                "metrics": payload.metrics,
                "complete": True,
            }
            self.updates[payload.client_id] = update
            self.accepted_tasks.add(payload.task_id)
            task["status"] = "completed"
            latency = float(payload.metrics.get("duration", 0.0))
            self.registry.mark_success(payload.client_id, latency)
            self._persist_worker(payload.client_id)
            self.events.log_event(
                "MODEL_UPDATE_RECEIVED",
                client_id=payload.client_id,
                round_number=payload.round,
                sample_count=payload.sample_count,
                complete=True,
                latency=latency,
                transport="http",
            )
            self.condition.notify_all()
        return {"accepted": True, "duplicate": False}

    def _current_state(self) -> dict:
        rows = self.events.fetch_all("SELECT value_json FROM state WHERE key='current'")
        if not rows:
            return {}
        import json

        return json.loads(rows[0]["value_json"])

    def status(self) -> dict:
        with self.condition:
            return {
                "state": self._current_state(),
                "registered_clients": sorted(self.registered),
                "workers": [
                    self.registry.snapshot(client_id) | self.worker_metadata.get(client_id, {})
                    for client_id in sorted(self.registry.clients)
                ],
            }

    def recent_events(self, limit: int = 100) -> list[dict]:
        return self.events.fetch_all(
            "SELECT timestamp,event_type,source,client_id,round,payload_json FROM events ORDER BY id DESC LIMIT ?",
            (min(max(limit, 1), 500),),
        )

    def start_training_thread(self) -> None:
        if self.training_thread and self.training_thread.is_alive():
            return
        self.training_thread = threading.Thread(
            target=self.run_training, name="network-fl-training", daemon=True
        )
        self.training_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()

    def _wait_for_workers(self) -> None:
        self.events.set_state(
            "current",
            {
                "status": "WAITING_FOR_WORKERS",
                "round": 0,
                "total_rounds": self.config.num_rounds,
                "required_clients": self.config.num_clients,
                "registered_clients": 0,
            },
        )
        self.logger.info("Waiting for %s network workers", self.config.num_clients)
        with self.condition:
            while len(self.registered) < self.config.num_clients and not self.stop_event.is_set():
                self.events.set_state(
                    "current",
                    {
                        "status": "WAITING_FOR_WORKERS",
                        "round": 0,
                        "total_rounds": self.config.num_rounds,
                        "required_clients": self.config.num_clients,
                        "registered_clients": len(self.registered),
                    },
                )
                self.condition.wait(timeout=1.0)
        if self.stop_event.is_set():
            raise RuntimeError("coordinator stopped before all workers registered")

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        criterion = nn.CrossEntropyLoss(reduction="sum")
        loss_sum = 0.0
        correct = 0
        count = 0
        for images, labels in self.test_loader:
            logits = self.model(images)
            loss_sum += float(criterion(logits, labels).item())
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            count += labels.size(0)
        return {"loss": loss_sum / max(1, count), "accuracy": correct / max(1, count)}

    def _last_telemetry(self, client_id: int) -> dict:
        rows = self.events.fetch_all(
            "SELECT * FROM telemetry WHERE client_id=? ORDER BY id DESC LIMIT 1",
            (client_id,),
        )
        return rows[0] if rows else {
            "round": self.current_round,
            "epoch": 0,
            "batch": 0,
            "global_step": 0,
            "loss": None,
            "accuracy": None,
            "timestamp": None,
        }

    def _recover_failure(self, client_id: int, reason: str) -> None:
        detected_epoch = time.time()
        detected_at = datetime.fromtimestamp(detected_epoch, timezone.utc).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z")
        record = self.registry.clients[client_id]
        detection_latency_ms = max(0.0, (detected_epoch - record.last_heartbeat) * 1000)
        telemetry = self._last_telemetry(client_id)
        recovery_started = time.time()
        recovery_started_at = utc_now()
        self.registry.mark_failed(client_id)
        self._persist_worker(client_id)
        checkpoint = self.latest_run_checkpoint
        checkpoint_name = checkpoint.name if checkpoint else "initial_global_model"
        failure = {
            "event_id": f"lan_{self.current_round:03d}_{client_id:03d}_{telemetry.get('global_step', 0)}",
            "failure_type": reason.upper(),
            "epoch": telemetry.get("epoch", 0),
            "batch": telemetry.get("batch", 0),
            "global_step": telemetry.get("global_step", 0),
            "last_loss": telemetry.get("loss"),
            "last_accuracy": telemetry.get("accuracy"),
            "last_heartbeat": datetime.fromtimestamp(record.last_heartbeat, timezone.utc).isoformat(
                timespec="milliseconds"
            ).replace("+00:00", "Z"),
            "detected_at": detected_at,
            "detection_latency_ms": detection_latency_ms,
            "checkpoint_used": checkpoint_name,
            "recovery_started_at": recovery_started_at,
            "recovery_completed_at": None,
            "recovery_time_ms": None,
            "status": "RECOVERING",
            "incomplete_update_discarded": True,
            "hostname": self.worker_metadata.get(client_id, {}).get("hostname"),
            "remote_addr": self.worker_metadata.get(client_id, {}).get("remote_addr"),
            "transport": "http",
        }
        row_id = self.events.log_event(
            "WORKER_FAILURE",
            client_id=client_id,
            round_number=self.current_round,
            timestamp=detected_at,
            **failure,
        )
        if checkpoint:
            restored_path, restored = self.checkpoints.restore(self.model, checkpoint)
            self.events.log_event(
                "CHECKPOINT_RESTORED",
                client_id=client_id,
                round_number=self.current_round,
                checkpoint=restored_path.name,
                restored_round=restored["round"],
            )
        healthy = self.scheduler.select(
            [self.registry.clients[cid] for cid in self.registered]
        )
        self.events.log_event(
            "WORKLOAD_REASSIGNED",
            client_id=client_id,
            round_number=self.current_round,
            failed_client=client_id,
            assigned_clients=healthy,
        )
        failure.update(
            recovery_completed_at=utc_now(),
            recovery_time_ms=(time.time() - recovery_started) * 1000,
            status="RECOVERED",
        )
        self.events.update_event_payload(row_id, failure)
        self.logger.error(
            "Network worker %s (%s) failed at round=%s epoch=%s batch=%s; restored %s",
            client_id,
            failure["hostname"],
            self.current_round,
            failure["epoch"],
            failure["batch"],
            checkpoint_name,
        )

    def _run_round(self, round_number: int) -> dict:
        with self.condition:
            eligible = [self.registry.clients[cid] for cid in self.registered]
            selected = self.scheduler.select(eligible)
            if not selected:
                raise RuntimeError("no healthy network workers remain")
            self.current_round = round_number
            self.updates = {}
            model_payload = encode_state_dict(cpu_state_dict(self.model))
            for client_id in selected:
                task_id = f"round-{round_number}-client-{client_id}-{secrets.token_hex(8)}"
                self.tasks[client_id] = {
                    "task_id": task_id,
                    "round": round_number,
                    "model_state": model_payload,
                    "training": {
                        "local_epochs": self.config.local_epochs,
                        "batch_size": self.config.batch_size,
                        "learning_rate": self.config.learning_rate,
                        "failure_mode": (
                            self.config.failure_mode
                            if client_id == self.config.failure_client_id
                            else "none"
                        ),
                        "failure_client_id": self.config.failure_client_id,
                        "failure_round": self.config.failure_round,
                        "failure_epoch": self.config.failure_epoch,
                        "failure_batch": self.config.failure_batch,
                    },
                    "status": "pending",
                }
                self.registry.clients[client_id].status = "TRAINING"
                self._persist_worker(client_id)
                self.events.log_event(
                    "CLIENT_SELECTED",
                    client_id=client_id,
                    round_number=round_number,
                    score=self.scheduler.score(self.registry.clients[client_id]),
                    hostname=self.worker_metadata.get(client_id, {}).get("hostname"),
                )
            self.events.log_event(
                "ROUND_STARTED", round_number=round_number, selected_clients=selected
            )
            self.events.set_state(
                "current",
                {
                    "status": "RUNNING",
                    "round": round_number,
                    "total_rounds": self.config.num_rounds,
                    "selected_clients": selected,
                },
            )
            self.condition.notify_all()

        started = time.monotonic()
        pending = set(selected)
        deadline = started + self.config.round_timeout
        while pending and not self.stop_event.is_set():
            with self.condition:
                pending -= set(self.updates)
                timed_out = self.detector.failed_clients(
                    self.registry.clients, pending, time.time()
                )
                for client_id in timed_out:
                    reason = (
                        self.config.failure_mode
                        if client_id == self.config.failure_client_id
                        and round_number == self.config.failure_round
                        else "heartbeat_timeout"
                    )
                    self._recover_failure(client_id, reason)
                    self.tasks[client_id]["status"] = "failed"
                    pending.remove(client_id)
                if time.monotonic() > deadline:
                    for client_id in list(pending):
                        self._recover_failure(client_id, "round_timeout")
                        self.tasks[client_id]["status"] = "failed"
                        pending.remove(client_id)
                if pending:
                    self.condition.wait(timeout=0.2)

        complete_updates = [self.updates[cid] for cid in selected if cid in self.updates]
        if len(complete_updates) < self.config.min_updates:
            raise RuntimeError(
                f"round {round_number} received only {len(complete_updates)} complete updates"
            )
        aggregated = weighted_fedavg(complete_updates, cpu_state_dict(self.model))
        self.model.load_state_dict(aggregated)
        evaluation = self.evaluate()
        duration = time.monotonic() - started
        active_clients = len(
            self.scheduler.select([self.registry.clients[cid] for cid in self.registered])
        )
        metrics = {
            **evaluation,
            "duration": duration,
            "active_clients": active_clients,
            "successful_updates": len(complete_updates),
            "selected_clients": selected,
        }
        checkpoint = self.checkpoints.save(round_number, self.model, metrics)
        self.latest_run_checkpoint = checkpoint
        self.events.log_event(
            "CHECKPOINT_SAVED",
            round_number=round_number,
            checkpoint=checkpoint.name,
            **evaluation,
        )
        self.events.record_round(round_number, metrics)
        self.events.log_event("ROUND_COMPLETED", round_number=round_number, **metrics)
        self.events.set_state(
            "current",
            {
                "status": "RUNNING",
                "round": round_number,
                "total_rounds": self.config.num_rounds,
                **metrics,
            },
        )
        self.logger.info(
            "Network round %s/%s accuracy=%.4f loss=%.4f updates=%s duration=%.2fs",
            round_number,
            self.config.num_rounds,
            evaluation["accuracy"],
            evaluation["loss"],
            len(complete_updates),
            duration,
        )
        return metrics

    def run_training(self) -> None:
        set_global_seed(self.config.seed)
        started = time.monotonic()
        try:
            self.events.set_state("config", self.config.to_dict() | {"transport": "http"})
            self.events.log_event(
                "SYSTEM_STARTED", config=self.config.to_dict(), transport="http"
            )
            self.logger.info("Preparing coordinator test dataset")
            ensure_mnist_available(self.config)
            self.test_loader = load_test_data(self.config)
            self._wait_for_workers()
            last_metrics = None
            for round_number in range(1, self.config.num_rounds + 1):
                last_metrics = self._run_round(round_number)
            summary = {
                "status": "COMPLETED",
                "round": self.config.num_rounds,
                "total_rounds": self.config.num_rounds,
                "total_duration": time.monotonic() - started,
                "final_accuracy": last_metrics["accuracy"],
                "final_loss": last_metrics["loss"],
                "registered_clients": sorted(self.registered),
                "transport": "http",
                "database": str(self.config.database_path),
            }
            self.events.log_event("TRAINING_COMPLETED", **summary)
            self.events.set_state("current", summary)
            self.logger.info("Network training completed accuracy=%.4f", summary["final_accuracy"])
        except BaseException as exc:
            if not self.stop_event.is_set():
                self.events.log_event("TRAINING_FAILED", error=repr(exc), transport="http")
                self.events.set_state("current", {"status": "FAILED", "error": repr(exc)})
                self.logger.exception("Network federated training failed")


def create_app(coordinator: NetworkCoordinator) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        coordinator.start_training_thread()
        try:
            yield
        finally:
            coordinator.stop()

    app = FastAPI(
        title="Federated Learning LAN Coordinator",
        version="1.0.0",
        description="Authenticated LAN transport for independent federated workers.",
        lifespan=lifespan,
    )

    def authorize(authorization: str | None = Header(default=None)) -> None:
        expected = f"Bearer {coordinator.shared_token}"
        if not authorization or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="invalid shared token")

    def translate_value_error(exc: ValueError) -> None:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "transport": "http"}

    @app.get("/worker-config/{client_id}", dependencies=[Depends(authorize)])
    def worker_config(client_id: int) -> dict:
        try:
            return coordinator.public_worker_config(client_id)
        except ValueError as exc:
            translate_value_error(exc)

    @app.post("/register", dependencies=[Depends(authorize)])
    def register(payload: RegisterRequest, request: Request) -> dict:
        try:
            remote_addr = request.client.host if request.client else None
            return coordinator.register(payload, remote_addr)
        except ValueError as exc:
            translate_value_error(exc)

    @app.post("/heartbeat", dependencies=[Depends(authorize)])
    def heartbeat(payload: HeartbeatRequest) -> dict:
        try:
            return coordinator.heartbeat(payload)
        except ValueError as exc:
            translate_value_error(exc)

    @app.post("/telemetry", dependencies=[Depends(authorize)])
    def telemetry(payload: TelemetryRequest) -> dict:
        try:
            return coordinator.telemetry(payload)
        except ValueError as exc:
            translate_value_error(exc)

    @app.post("/event", dependencies=[Depends(authorize)])
    def client_event(payload: ClientEventRequest) -> dict:
        try:
            return coordinator.client_event(payload)
        except ValueError as exc:
            translate_value_error(exc)

    @app.get("/task/{client_id}", dependencies=[Depends(authorize)])
    def task(client_id: int) -> dict:
        try:
            return coordinator.get_task(client_id)
        except ValueError as exc:
            translate_value_error(exc)

    @app.post("/update", dependencies=[Depends(authorize)])
    def update(payload: UpdateRequest) -> dict:
        try:
            return coordinator.submit_update(payload)
        except ValueError as exc:
            translate_value_error(exc)

    @app.get("/status", dependencies=[Depends(authorize)])
    def status() -> dict:
        return coordinator.status()

    @app.get("/events", dependencies=[Depends(authorize)])
    def events(limit: int = 100) -> list[dict]:
        return coordinator.recent_events(limit)

    return app

