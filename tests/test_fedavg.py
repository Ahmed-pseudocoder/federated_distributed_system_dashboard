from collections import OrderedDict

import pytest
import torch

from coordinator.aggregator import weighted_fedavg


def update(value, samples=10, complete=True):
    return {
        "state_dict": OrderedDict(weight=torch.tensor([float(value)])),
        "sample_count": samples,
        "complete": complete,
    }


def test_weighted_fedavg_equal_samples():
    result = weighted_fedavg([update(1), update(3)])
    assert result["weight"].item() == pytest.approx(2.0)


def test_weighted_fedavg_uses_sample_counts():
    result = weighted_fedavg([update(1, 1), update(3, 3)])
    assert result["weight"].item() == pytest.approx(2.5)


def test_incomplete_update_is_rejected():
    with pytest.raises(ValueError, match="incomplete"):
        weighted_fedavg([update(1), update(3, complete=False)])

