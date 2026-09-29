from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import queue
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from client.client import client_process
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


class FederatedCoordinator:
    def __init__(self, config):
        self.config = config
        self.config.ensure_directories()
        self.logger = self._create_logger()
        self.events = EventLogger(config.database_path)
        self.registry = WorkerRegistry(list(range(1, config.num_clients + 1)))
        self.scheduler = DynamicScheduler()
        self.detector = FaultDetector(config.heartbeat_timeout)
        self.checkpoints = CheckpointManager(config.checkpoint_dir)
        self.model = create_model(config.seed)
        self.test_loader = None
        self.ctx = mp.get_context("spawn")
        self.manager = None
        self.event_queue = None
        self.command_queues: dict[int, object] = {}
        self.processes: dict[int, mp.Process] = {}
        self.failure_events: list[dict] = []
        self.latest_run_checkpoint: Path | None = None

    def _create_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"fl.coordinator.{self.config.run_id}.{os.getpid()}")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            file_handler = logging.FileHandler(self.config.log_dir / "coordinator.log", encoding="utf-8")
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        logger.propagate = False
        return logger

    def _persist_worker(self, client_id: int) -> None:
        self.events.upsert_worker(client_id, self.registry.snapshot(client_id))

    def _handle_message(self, item: dict) -> tuple[str, dict] | None:
        kind = item.get("kind")
        client_id = item.get("client_id")
        if kind == "registered":
            self.registry.register(client_id, item["pid"])
            self._persist_worker(client_id)
            self.events.log_event(
                "CLIENT_CONNECTED", source=f"client-{client_id}", client_id=client_id,
                pid=item["pid"], sample_count=item["sample_count"], distribution=item["distribution"],
            )
        elif kind == "heartbeat":
            self.registry.heartbeat(client_id, item)
            self._persist_worker(client_id)
            self.events.log_event(
                "CLIENT_HEARTBEAT", source=f"client-{client_id}", client_id=client_id,
                round_number=item.get("round"), epoch=item.get("epoch"), batch=item.get("batch"),
                global_step=item.get("global_step"), status=item.get("status"), pid=item.get("pid"),
            )
        elif kind == "telemetry":
            telemetry = item["telemetry"]
            self.registry.telemetry(telemetry["client_id"], telemetry)
            self.events.record_telemetry(telemetry)
            self._persist_worker(telemetry["client_id"])
        elif kind == "event":
            # Any client-originated event proves liveness. In particular,
            # LOCAL_TRAINING_COMPLETED can arrive just before a large model
            # update is serialized onto the queue.
            if client_id is not None:
                self.registry.heartbeat(client_id, item)
                self._persist_worker(client_id)
            reserved = {"kind", "timestamp", "event_type", "source", "client_id", "round"}
            payload = {key: value for key, value in item.items() if key not in reserved}
            self.events.log_event(
                item["event_type"], source=item.get("source", "client"), client_id=client_id,
                round_number=item.get("round"), timestamp=item.get("timestamp"), **payload,
            )
        elif kind == "client_exception":
            self.logger.error("Client %s exception: %s", client_id, item.get("error"))
            self.events.log_event(
                "CLIENT_EXCEPTION", source=f"client-{client_id}", client_id=client_id,
                round_number=item.get("round"), error=item.get("error"), traceback=item.get("traceback"),
            )
        elif kind == "update":
            return "update", item
        return None

    def _drain_one(self, timeout: float = 0.1):
        try:
            item = self.event_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        return self._handle_message(item)

    def _start_clients(self) -> None:
        self.manager = self.ctx.Manager()
        self.event_queue = self.manager.Queue()
        for client_id in range(1, self.config.num_clients + 1):
            command_queue = self.ctx.Queue()
            process = self.ctx.Process(
                target=client_process,
                name=f"fl-client-{client_id}",
                args=(client_id, command_queue, self.event_queue, self.config),
            )
            process.start()
            self.command_queues[client_id] = command_queue
            self.processes[client_id] = process

        deadline = time.monotonic() + 180
        while any(record.status == "STARTING" for record in self.registry.clients.values()):
            self._drain_one(timeout=0.5)
            if time.monotonic() > deadline:
                missing = [record.client_id for record in self.registry.clients.values() if record.status == "STARTING"]
                raise TimeoutError(f"clients did not register: {missing}")
            dead = [client_id for client_id, process in self.processes.items() if not process.is_alive()]
            if dead:
                raise RuntimeError(f"clients exited before registration: {dead}")

    def _last_telemetry(self, client_id: int) -> dict:
        rows = self.events.fetch_all(
            "SELECT * FROM telemetry WHERE client_id=? ORDER BY id DESC LIMIT 1", (client_id,)
        )
        return rows[0] if rows else {
            "round": 0, "epoch": 0, "batch": 0, "global_step": 0,
            "loss": None, "accuracy": None, "timestamp": None,
        }

    def _recover_failure(self, client_id: int, round_number: int, reason: str) -> None:
        detected_epoch = time.time()
        detected_at = datetime.fromtimestamp(detected_epoch, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        record = self.registry.clients[client_id]
        detection_latency_ms = max(0.0, (detected_epoch - record.last_heartbeat) * 1000)
        telemetry = self._last_telemetry(client_id)
        recovery_started = time.time()
        recovery_started_at = utc_now()
        self.registry.mark_failed(client_id)
        self._persist_worker(client_id)

        checkpoint = self.latest_run_checkpoint
        checkpoint_name = checkpoint.name if checkpoint is not None else "initial_global_model"
        failure = {
            "event_id": f"evt_{round_number:03d}_{client_id:03d}_{telemetry.get('global_step', 0)}",
            "failure_type": reason.upper(),
            "epoch": telemetry.get("epoch", 0),
            "batch": telemetry.get("batch", 0),
            "global_step": telemetry.get("global_step", 0),
            "last_loss": telemetry.get("loss"),
            "last_accuracy": telemetry.get("accuracy"),
            "last_heartbeat": datetime.fromtimestamp(record.last_heartbeat, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "detected_at": detected_at,
            "detection_latency_ms": detection_latency_ms,
            "checkpoint_used": checkpoint_name,
            "recovery_started_at": recovery_started_at,
            "recovery_completed_at": None,
            "recovery_time_ms": None,
            "status": "RECOVERING",
            "incomplete_update_discarded": True,
            "process_exitcode": self.processes[client_id].exitcode,
        }
        failure_row_id = self.events.log_event(
            "WORKER_FAILURE", client_id=client_id, round_number=round_number,
            timestamp=detected_at, **failure,
        )
        if checkpoint is not None:
            restored_path, restored = self.checkpoints.restore(self.model, checkpoint)
            checkpoint_name = restored_path.name
            self.events.log_event(
                "CHECKPOINT_RESTORED", round_number=round_number, client_id=client_id,
                checkpoint=checkpoint_name, restored_round=restored["round"],
            )

        healthy = self.scheduler.select(list(self.registry.clients.values()))
        self.events.log_event(
            "WORKLOAD_REASSIGNED", round_number=round_number, client_id=client_id,
            failed_client=client_id, assigned_clients=healthy,
        )
        recovery_completed = time.time()
        failure.update(
            checkpoint_used=checkpoint_name,
            recovery_completed_at=utc_now(),
            recovery_time_ms=(recovery_completed - recovery_started) * 1000,
            status="RECOVERED",
            process_exitcode=self.processes[client_id].exitcode,
        )
        self.events.update_event_payload(failure_row_id, failure)
        self.failure_events.append({"client_id": client_id, "round": round_number, **failure})
        self.logger.error(
            "Worker %s failed at round=%s epoch=%s batch=%s step=%s; checkpoint=%s; recovered in %.1fms",
            client_id, round_number, failure["epoch"], failure["batch"], failure["global_step"],
            checkpoint_name, failure["recovery_time_ms"],
        )

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

    def _run_round(self, round_number: int) -> dict:
        selected = self.scheduler.select(list(self.registry.clients.values()))
        if not selected:
            raise RuntimeError("no healthy clients remain")
        started = time.monotonic()
        self.events.log_event("ROUND_STARTED", round_number=round_number, selected_clients=selected)
        global_state = cpu_state_dict(self.model)
        for client_id in selected:
            self.registry.clients[client_id].status = "TRAINING"
            self._persist_worker(client_id)
            self.events.log_event("CLIENT_SELECTED", client_id=client_id, round_number=round_number, score=self.scheduler.score(self.registry.clients[client_id]))
            self.command_queues[client_id].put({"kind": "train", "round": round_number, "state_dict": global_state})

        pending = set(selected)
        updates: list[dict] = []
        deadline = time.monotonic() + self.config.round_timeout
        while pending:
            handled_messages = []
            first = self._drain_one(timeout=0.1)
            if first is not None:
                handled_messages.append(first)
            # Drain the ready burst before checking heartbeat age. Otherwise a
            # completed update behind batch events could be falsely timed out.
            for _ in range(100):
                ready = self._drain_one(timeout=0)
                if ready is None:
                    break
                handled_messages.append(ready)
            for handled in handled_messages:
                if handled[0] != "update":
                    continue
                item = handled[1]
                client_id = item["client_id"]
                if item["round"] == round_number and client_id in pending:
                    update = item["update"]
                    updates.append(update)
                    pending.remove(client_id)
                    latency = float(update["metrics"]["duration"])
                    self.registry.mark_success(client_id, latency)
                    self._persist_worker(client_id)
                    self.events.log_event(
                        "MODEL_UPDATE_RECEIVED", client_id=client_id, round_number=round_number,
                        sample_count=update["sample_count"], complete=update["complete"], latency=latency,
                    )

            for client_id in self.detector.failed_clients(self.registry.clients, pending):
                self._recover_failure(client_id, round_number, self.config.failure_mode if client_id == self.config.failure_client_id else "heartbeat_timeout")
                pending.remove(client_id)

            if time.monotonic() > deadline:
                for client_id in list(pending):
                    self._recover_failure(client_id, round_number, "round_timeout")
                    pending.remove(client_id)

        if len(updates) < self.config.min_updates:
            raise RuntimeError(f"round {round_number} received only {len(updates)} complete updates")
        aggregated = weighted_fedavg(updates, cpu_state_dict(self.model))
        self.model.load_state_dict(aggregated)
        evaluation = self.evaluate()
        duration = time.monotonic() - started
        metrics = {
            **evaluation,
            "duration": duration,
            "active_clients": len(self.scheduler.select(list(self.registry.clients.values()))),
            "successful_updates": len(updates),
            "selected_clients": selected,
        }
        checkpoint = self.checkpoints.save(round_number, self.model, metrics)
        self.latest_run_checkpoint = checkpoint
        self.events.log_event("CHECKPOINT_SAVED", round_number=round_number, checkpoint=checkpoint.name, **evaluation)
        self.events.record_round(round_number, metrics)
        self.events.log_event("ROUND_COMPLETED", round_number=round_number, **metrics)
        self.events.set_state("current", {"status": "RUNNING", "round": round_number, "total_rounds": self.config.num_rounds, **metrics})
        self.logger.info(
            "Round %s/%s accuracy=%.4f loss=%.4f updates=%s duration=%.2fs",
            round_number, self.config.num_rounds, evaluation["accuracy"], evaluation["loss"], len(updates), duration,
        )
        return metrics

    def shutdown(self) -> None:
        for client_id, process in self.processes.items():
            if process.is_alive():
                try:
                    self.command_queues[client_id].put({"kind": "shutdown"})
                except (BrokenPipeError, EOFError):
                    pass
        for process in self.processes.values():
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        if self.manager is not None:
            self.manager.shutdown()

    def run(self) -> dict:
        set_global_seed(self.config.seed)
        started = time.monotonic()
        self.events.set_state("config", self.config.to_dict())
        self.events.set_state("current", {"status": "STARTING", "round": 0, "total_rounds": self.config.num_rounds})
        self.events.log_event("SYSTEM_STARTED", config=self.config.to_dict())
        self.logger.info("Preparing MNIST dataset")
        ensure_mnist_available(self.config)
        self.test_loader = load_test_data(self.config)
        last_metrics = None
        try:
            self._start_clients()
            for round_number in range(1, self.config.num_rounds + 1):
                last_metrics = self._run_round(round_number)
            summary = {
                "status": "COMPLETED",
                "round": self.config.num_rounds,
                "total_rounds": self.config.num_rounds,
                "total_duration": time.monotonic() - started,
                "final_accuracy": last_metrics["accuracy"],
                "final_loss": last_metrics["loss"],
                "failure_count": len(self.failure_events),
                "failures": self.failure_events,
                "database": str(self.config.database_path),
            }
            self.events.log_event("TRAINING_COMPLETED", **summary)
            self.events.set_state("current", summary)
            self.logger.info("Training completed accuracy=%.4f in %.2fs", summary["final_accuracy"], summary["total_duration"])
            return summary
        except BaseException as exc:
            self.events.log_event("TRAINING_FAILED", error=repr(exc))
            self.events.set_state("current", {"status": "FAILED", "error": repr(exc)})
            self.logger.exception("Federated training failed")
            raise
        finally:
            self.shutdown()


def run_federated(config) -> dict:
    return FederatedCoordinator(config).run()

