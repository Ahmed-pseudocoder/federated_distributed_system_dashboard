from __future__ import annotations

import os
import time


def should_fail(config, client_id: int, round_number: int, epoch: int, batch: int) -> bool:
    return (
        config.failure_mode != "none"
        and client_id == config.failure_client_id
        and round_number == config.failure_round
        and epoch == config.failure_epoch
        and batch == config.failure_batch
    )


def trigger_failure(mode: str, heartbeat_stop) -> None:
    heartbeat_stop.set()
    if mode == "process_crash":
        os._exit(42)
    if mode in {"hang", "network_timeout"}:
        while True:
            time.sleep(60)
    raise ValueError(f"unsupported failure mode: {mode}")
