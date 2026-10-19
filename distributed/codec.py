from __future__ import annotations

import base64
import io
from collections import OrderedDict

import torch


def encode_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    """Serialize tensors for transport over the JSON demo protocol."""
    buffer = io.BytesIO()
    cpu_state = OrderedDict(
        (name, tensor.detach().cpu().clone()) for name, tensor in state_dict.items()
    )
    torch.save(cpu_state, buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_state_dict(payload: str) -> OrderedDict[str, torch.Tensor]:
    raw = base64.b64decode(payload.encode("ascii"), validate=True)
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(state, (dict, OrderedDict)):
        raise ValueError("model payload is not a state dictionary")
    if not all(isinstance(name, str) and torch.is_tensor(tensor) for name, tensor in state.items()):
        raise ValueError("model payload contains unsupported values")
    return OrderedDict(state)
