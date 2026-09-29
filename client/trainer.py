from __future__ import annotations

import logging
import time

import torch
from torch import nn

from client.failure_simulator import should_fail, trigger_failure
from client.telemetry import make_telemetry
from core.messages import message
from core.serialization import cpu_state_dict


def train_local(
    *,
    client_id: int,
    round_number: int,
    model,
    loader,
    config,
    event_queue,
    progress,
    progress_lock,
    heartbeat_stop,
    logger: logging.Logger,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    started = time.monotonic()
    running_loss = 0.0
    correct = 0
    seen = 0
    completed_batches = 0

    with progress_lock:
        progress.update({"round": round_number, "epoch": 0, "batch": 0, "status": "training"})
    event_queue.put(message("event", event_type="LOCAL_TRAINING_STARTED", source=f"client-{client_id}", client_id=client_id, round=round_number))
    logger.info("Client %s Round %s local training started", client_id, round_number)

    for epoch_index in range(1, config.local_epochs + 1):
        logger.info("Client %s Round %s Epoch %s started", client_id, round_number, epoch_index)
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += float(loss.item()) * batch_size
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += batch_size
            completed_batches += 1
            with progress_lock:
                progress["epoch"] = epoch_index
                progress["batch"] = batch_index
                progress["global_step"] = int(progress.get("global_step", 0)) + 1
                snapshot = dict(progress)
            telemetry = make_telemetry(
                client_id,
                snapshot,
                loss=running_loss / seen,
                accuracy=correct / seen,
                status="training",
            )
            if batch_index % config.telemetry_every_batches == 0:
                event_queue.put(message("telemetry", telemetry=telemetry))
                event_queue.put(
                    message(
                        "event",
                        event_type="BATCH_COMPLETED",
                        source=f"client-{client_id}",
                        client_id=client_id,
                        round=round_number,
                        epoch=epoch_index,
                        batch=batch_index,
                        global_step=telemetry["global_step"],
                        loss=telemetry["loss"],
                        accuracy=telemetry["accuracy"],
                    )
                )
            logger.info(
                "Client %s Round %s Epoch %s Batch %s completed loss=%.5f accuracy=%.4f",
                client_id, round_number, epoch_index, batch_index, telemetry["loss"], telemetry["accuracy"],
            )

            if should_fail(config, client_id, round_number, epoch_index, batch_index):
                # Manager-backed queue puts are synchronous. This final telemetry and
                # heartbeat are committed before the process is genuinely terminated.
                event_queue.put(message("telemetry", telemetry=telemetry))
                event_queue.put(message("heartbeat", client_id=client_id, pid=telemetry["pid"], **snapshot))
                logger.critical(
                    "Injecting %s after completed Round %s Epoch %s Batch %s Step %s",
                    config.failure_mode, round_number, epoch_index, batch_index, telemetry["global_step"],
                )
                for handler in logger.handlers:
                    handler.flush()
                trigger_failure(config.failure_mode, heartbeat_stop)

    duration = time.monotonic() - started
    metrics = {
        "loss": running_loss / max(1, seen),
        "accuracy": correct / max(1, seen),
        "duration": duration,
        "completed_batches": completed_batches,
    }
    with progress_lock:
        progress["status"] = "healthy"
    event_queue.put(message("event", event_type="LOCAL_TRAINING_COMPLETED", source=f"client-{client_id}", client_id=client_id, round=round_number, **metrics))
    logger.info("Client %s Round %s local training completed in %.3fs", client_id, round_number, duration)
    return {
        "state_dict": cpu_state_dict(model),
        "sample_count": len(loader.dataset),
        "metrics": metrics,
        "complete": True,
    }
