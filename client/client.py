from __future__ import annotations

import logging
import os
import threading
import traceback

import torch

from client.heartbeat import HeartbeatEmitter
from client.trainer import train_local
from core.data import load_client_data, set_global_seed
from core.messages import message
from core.model import create_model


def _logger_for(client_id: int, log_dir) -> logging.Logger:
    logger = logging.getLogger(f"fl.client.{client_id}.{os.getpid()}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_dir / f"client_{client_id}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def client_process(client_id: int, command_queue, event_queue, config) -> None:
    torch.set_num_threads(1)
    set_global_seed(config.seed + client_id)
    logger = _logger_for(client_id, config.log_dir)
    progress = {"round": 0, "epoch": 0, "batch": 0, "global_step": 0, "status": "starting"}
    progress_lock = threading.Lock()
    heartbeat_stop = threading.Event()
    try:
        loader, distribution = load_client_data(config, client_id)
        event_queue.put(
            message(
                "registered",
                client_id=client_id,
                pid=os.getpid(),
                sample_count=len(loader.dataset),
                distribution=distribution,
            )
        )
        with progress_lock:
            progress["status"] = "healthy"
        heartbeat = HeartbeatEmitter(
            client_id, event_queue, progress, progress_lock, config.heartbeat_interval, heartbeat_stop
        )
        heartbeat.start()
        logger.info("Client %s registered pid=%s samples=%s distribution=%s", client_id, os.getpid(), len(loader.dataset), distribution)

        while True:
            command = command_queue.get()
            if command["kind"] == "shutdown":
                break
            if command["kind"] != "train":
                logger.warning("Ignoring unknown command %s", command.get("kind"))
                continue
            round_number = int(command["round"])
            model = create_model(config.seed)
            model.load_state_dict(command["state_dict"])
            update = train_local(
                client_id=client_id,
                round_number=round_number,
                model=model,
                loader=loader,
                config=config,
                event_queue=event_queue,
                progress=progress,
                progress_lock=progress_lock,
                heartbeat_stop=heartbeat_stop,
                logger=logger,
            )
            event_queue.put(
                message("update", client_id=client_id, round=round_number, update=update)
            )
    except BaseException as exc:
        logger.exception("Unexpected client failure: %s", exc)
        event_queue.put(
            message(
                "client_exception",
                client_id=client_id,
                pid=os.getpid(),
                error=repr(exc),
                traceback=traceback.format_exc(),
                **progress,
            )
        )
    finally:
        heartbeat_stop.set()
        logger.info("Client %s exiting", client_id)
