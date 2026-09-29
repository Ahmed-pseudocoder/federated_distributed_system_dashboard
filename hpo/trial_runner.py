from __future__ import annotations

import time
from pathlib import Path

from coordinator.server import run_federated
from core.config import FLConfig, PROJECT_ROOT


def run_trial(spec: dict) -> dict:
    trial_id = int(spec["trial_id"])
    trial_dir = Path(spec.get("base_dir", PROJECT_ROOT / "results")) / f"trial_{trial_id:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    config = FLConfig(
        num_clients=3,
        num_rounds=int(spec.get("rounds", 3)),
        local_epochs=int(spec["local_epochs"]),
        batch_size=int(spec["batch_size"]),
        learning_rate=float(spec["learning_rate"]),
        failure_mode="none",
        max_train_samples_per_client=int(spec.get("max_train_samples", 600)),
        max_test_samples=int(spec.get("max_test_samples", 1_000)),
        seed=int(spec.get("seed", 42)) + trial_id,
        run_id=f"trial_{trial_id:03d}",
        results_dir=trial_dir,
        checkpoint_dir=trial_dir / "checkpoints",
        log_dir=trial_dir / "logs",
    )
    started = time.monotonic()
    result = {
        "trial_id": trial_id,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "local_epochs": config.local_epochs,
        "result_dir": str(trial_dir),
    }
    try:
        summary = run_federated(config)
        result.update(
            accuracy=summary["final_accuracy"],
            duration=time.monotonic() - started,
            status="COMPLETED",
        )
    except BaseException as exc:
        result.update(accuracy=None, duration=time.monotonic() - started, status="FAILED", error=repr(exc))
    return result

