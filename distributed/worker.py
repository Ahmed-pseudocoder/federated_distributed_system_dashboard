from __future__ import annotations

import logging
import os
import platform
import socket
import threading
import time
from dataclasses import replace
from pathlib import Path

import torch

from client.trainer import train_local
from core.config import FLConfig, PROJECT_ROOT
from core.data import load_client_data, set_global_seed
from core.messages import message
from core.model import create_model
from distributed.codec import decode_state_dict, encode_state_dict
from distributed.http_client import CoordinatorHttpClient, HttpEventSink


class NetworkHeartbeatEmitter(threading.Thread):
    def __init__(self, client_id, sink, progress, lock, interval, stop_event, logger):
        super().__init__(name=f"network-heartbeat-{client_id}", daemon=True)
        self.client_id = client_id
        self.sink = sink
        self.progress = progress
        self.lock = lock
        self.interval = interval
        self.stop_event = stop_event
        self.logger = logger

    def run(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                snapshot = dict(self.progress)
            try:
                self.sink.put(
                    message(
                        "heartbeat",
                        client_id=self.client_id,
                        pid=os.getpid(),
                        **snapshot,
                    )
                )
            except ConnectionError as exc:
                self.logger.warning("Heartbeat delivery failed: %s", exc)
            self.stop_event.wait(self.interval)


def _create_logger(client_id: int, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"fl.network.worker.{client_id}.{os.getpid()}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(
            log_dir / f"network_client_{client_id}.log", encoding="utf-8"
        )
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    logger.propagate = False
    return logger


def run_network_worker(
    *,
    client_id: int,
    coordinator_url: str,
    token: str,
    data_dir: Path | None = None,
    log_dir: Path | None = None,
    poll_interval: float = 1.0,
) -> None:
    torch.set_num_threads(1)
    http = CoordinatorHttpClient(coordinator_url, token, timeout=180.0)
    remote_config = http.get(f"/worker-config/{client_id}")
    config = FLConfig(
        num_clients=int(remote_config["num_clients"]),
        local_epochs=int(remote_config["local_epochs"]),
        batch_size=int(remote_config["batch_size"]),
        learning_rate=float(remote_config["learning_rate"]),
        heartbeat_interval=float(remote_config["heartbeat_interval"]),
        telemetry_every_batches=int(remote_config["telemetry_every_batches"]),
        max_train_samples_per_client=int(remote_config["max_train_samples_per_client"]),
        seed=int(remote_config["seed"]),
        failure_mode="none",
        run_id=f"network_worker_{client_id}",
        data_dir=data_dir or PROJECT_ROOT / "data",
        log_dir=log_dir or PROJECT_ROOT / "logs",
    )
    config.ensure_directories()
    logger = _create_logger(client_id, config.log_dir)
    set_global_seed(config.seed + client_id)
    logger.info("Loading local MNIST partition for Client %s", client_id)
    loader, distribution = load_client_data(config, client_id)
    hostname = socket.gethostname()
    http.post(
        "/register",
        {
            "client_id": client_id,
            "pid": os.getpid(),
            "hostname": hostname,
            "platform": platform.platform(),
            "sample_count": len(loader.dataset),
            "distribution": {str(key): value for key, value in distribution.items()},
        },
    )
    logger.info(
        "Registered with %s as Client %s host=%s pid=%s samples=%s",
        coordinator_url,
        client_id,
        hostname,
        os.getpid(),
        len(loader.dataset),
    )

    sink = HttpEventSink(http)
    progress = {
        "round": 0,
        "epoch": 0,
        "batch": 0,
        "global_step": 0,
        "status": "healthy",
    }
    progress_lock = threading.Lock()
    heartbeat_stop = threading.Event()
    heartbeat = NetworkHeartbeatEmitter(
        client_id,
        sink,
        progress,
        progress_lock,
        config.heartbeat_interval,
        heartbeat_stop,
        logger,
    )
    heartbeat.start()
    completed_task_ids: set[str] = set()

    try:
        while True:
            task = http.get(f"/task/{client_id}")
            action = task.get("action")
            if action == "shutdown":
                logger.info("Coordinator reported training status %s", task.get("training_status"))
                return
            if action != "train":
                time.sleep(float(task.get("poll_after", poll_interval)))
                continue
            task_id = task["task_id"]
            if task_id in completed_task_ids:
                time.sleep(poll_interval)
                continue

            training = task["training"]
            task_config = replace(
                config,
                local_epochs=int(training["local_epochs"]),
                batch_size=int(training["batch_size"]),
                learning_rate=float(training["learning_rate"]),
                failure_mode=str(training["failure_mode"]),
                failure_client_id=int(training["failure_client_id"]),
                failure_round=int(training["failure_round"]),
                failure_epoch=int(training["failure_epoch"]),
                failure_batch=int(training["failure_batch"]),
            )
            round_number = int(task["round"])
            model = create_model(task_config.seed)
            model.load_state_dict(decode_state_dict(task["model_state"]))
            logger.info("Received task %s for round %s", task_id, round_number)
            update = train_local(
                client_id=client_id,
                round_number=round_number,
                model=model,
                loader=loader,
                config=task_config,
                event_queue=sink,
                progress=progress,
                progress_lock=progress_lock,
                heartbeat_stop=heartbeat_stop,
                logger=logger,
            )
            response = http.post(
                "/update",
                {
                    "client_id": client_id,
                    "round": round_number,
                    "task_id": task_id,
                    "sample_count": update["sample_count"],
                    "metrics": update["metrics"],
                    "state_dict": encode_state_dict(update["state_dict"]),
                    "complete": True,
                },
            )
            if response.get("accepted"):
                completed_task_ids.add(task_id)
                logger.info("Coordinator accepted round %s update", round_number)
    finally:
        heartbeat_stop.set()
        heartbeat.join(timeout=2)
