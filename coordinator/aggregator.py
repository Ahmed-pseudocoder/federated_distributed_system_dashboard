from __future__ import annotations

from collections import OrderedDict

import torch

from core.serialization import validate_state_dict


def weighted_fedavg(updates: list[dict], reference_state: dict | None = None):
    if not updates:
        raise ValueError("at least one complete update is required")
    if any(not update.get("complete", False) for update in updates):
        raise ValueError("incomplete updates must not be aggregated")
    reference = reference_state or updates[0]["state_dict"]
    total_samples = sum(int(update["sample_count"]) for update in updates)
    if total_samples <= 0:
        raise ValueError("total sample count must be positive")
    for update in updates:
        validate_state_dict(update["state_dict"], reference)

    result = OrderedDict()
    for key, reference_tensor in reference.items():
        if torch.is_floating_point(reference_tensor):
            accumulator = torch.zeros_like(reference_tensor, dtype=torch.float64)
            for update in updates:
                weight = int(update["sample_count"]) / total_samples
                accumulator.add_(update["state_dict"][key].double(), alpha=weight)
            result[key] = accumulator.to(reference_tensor.dtype)
        else:
            # Integer buffers (for example batch counters) are copied from the
            # largest contributing client rather than averaged into invalid values.
            largest = max(updates, key=lambda item: int(item["sample_count"]))
            result[key] = largest["state_dict"][key].clone()
    return result

