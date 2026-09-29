from __future__ import annotations

from collections import OrderedDict

import torch


def cpu_state_dict(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((key, value.detach().cpu().clone()) for key, value in model.state_dict().items())


def validate_state_dict(candidate, reference) -> None:
    if list(candidate.keys()) != list(reference.keys()):
        raise ValueError("state dict keys do not match the global model")
    for key, tensor in candidate.items():
        if tensor.shape != reference[key].shape:
            raise ValueError(f"shape mismatch for {key}")
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            raise ValueError(f"non-finite values in {key}")

