from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class FLConfig:
    num_clients: int = 3
    num_rounds: int = 10
    local_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 0.001
    heartbeat_interval: float = 0.5
    heartbeat_timeout: float = 3.0
    round_timeout: float = 180.0
    failure_mode: str = "process_crash"
    failure_client_id: int = 2
    failure_round: int = 4
    failure_epoch: int = 2
    failure_batch: int = 37
    seed: int = 42
    max_train_samples_per_client: int = 1_200
    max_test_samples: int = 2_000
    min_updates: int = 1
    telemetry_every_batches: int = 1
    num_workers: int = 0
    run_id: str = "default"
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    checkpoint_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "checkpoints")
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results")
    log_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.results_dir = Path(self.results_dir)
        self.log_dir = Path(self.log_dir)
        if self.num_clients < 1:
            raise ValueError("num_clients must be positive")
        if self.failure_mode not in {"none", "process_crash", "hang", "network_timeout"}:
            raise ValueError(f"unsupported failure_mode: {self.failure_mode}")
        if self.min_updates < 1:
            raise ValueError("min_updates must be positive")

    @property
    def database_path(self) -> Path:
        return self.results_dir / f"{self.run_id}.db"

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.checkpoint_dir, self.results_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key, value in list(result.items()):
            if isinstance(value, Path):
                result[key] = str(value)
        return result

