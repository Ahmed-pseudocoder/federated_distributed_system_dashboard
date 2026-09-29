from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

from coordinator.event_logger import EventLogger
from core.config import PROJECT_ROOT
from hpo.trial_runner import run_trial


def optimize(
    n_trials: int = 3,
    workers: int = 3,
    rounds: int = 3,
    max_train_samples: int = 600,
    base_dir: Path | None = None,
) -> dict:
    destination = Path(base_dir or PROJECT_ROOT / "results")
    destination.mkdir(parents=True, exist_ok=True)
    grid = list(product((0.01, 0.001, 0.0001), (32, 64), (1, 2)))
    specs = [
        {
            "trial_id": index + 1,
            "learning_rate": values[0],
            "batch_size": values[1],
            "local_epochs": values[2],
            "rounds": rounds,
            "max_train_samples": max_train_samples,
            "base_dir": str(destination),
        }
        for index, values in enumerate(grid[:n_trials])
    ]
    results = []
    with ProcessPoolExecutor(max_workers=min(workers, len(specs))) as executor:
        futures = [executor.submit(run_trial, spec) for spec in specs]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda trial: trial["trial_id"])
    logger = EventLogger(destination / "hpo.db")
    for result in results:
        logger.record_hpo_trial(result)
    completed = [trial for trial in results if trial["status"] == "COMPLETED"]
    best = max(completed, key=lambda trial: trial["accuracy"]) if completed else None
    logger.set_state("hpo_summary", {"trials": results, "best": best})
    return {"trials": results, "best": best}
