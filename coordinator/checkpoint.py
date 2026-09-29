from __future__ import annotations

import os
from pathlib import Path

import torch

from core.messages import utc_now


class CheckpointManager:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, round_number: int) -> Path:
        return self.directory / f"round_{round_number:03d}.pt"

    def save(self, round_number: int, model, metrics: dict, optimizer_state_dict=None) -> Path:
        path = self.path_for(round_number)
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "round": round_number,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_state_dict or {},
                "metrics": metrics,
                "timestamp": utc_now(),
            },
            temporary,
        )
        os.replace(temporary, path)
        return path

    def latest(self) -> Path | None:
        candidates = sorted(self.directory.glob("round_*.pt"))
        return candidates[-1] if candidates else None

    def load(self, path: Path | None = None) -> dict:
        selected = path or self.latest()
        if selected is None:
            raise FileNotFoundError("no valid checkpoint exists")
        return torch.load(selected, map_location="cpu", weights_only=False)

    def restore(self, model, path: Path | None = None) -> tuple[Path, dict]:
        selected = path or self.latest()
        checkpoint = self.load(selected)
        model.load_state_dict(checkpoint["model_state_dict"])
        return Path(selected), checkpoint
