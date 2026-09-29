from __future__ import annotations

import os

from core.messages import Telemetry, utc_now


def make_telemetry(client_id: int, progress: dict, loss=None, accuracy=None, status="training") -> dict:
    return Telemetry(
        client_id=client_id,
        round=int(progress.get("round", 0)),
        epoch=int(progress.get("epoch", 0)),
        batch=int(progress.get("batch", 0)),
        global_step=int(progress.get("global_step", 0)),
        loss=None if loss is None else float(loss),
        accuracy=None if accuracy is None else float(accuracy),
        timestamp=utc_now(),
        status=status,
        pid=os.getpid(),
    ).to_dict()

