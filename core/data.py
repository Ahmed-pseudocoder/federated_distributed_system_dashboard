from __future__ import annotations

import random
from collections import Counter
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


PRIMARY_DIGITS = {
    1: (0, 1, 2, 3),
    2: (4, 5, 6),
    3: (7, 8, 9),
}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _targets(dataset: Dataset) -> list[int]:
    values = getattr(dataset, "targets", None)
    if values is None:
        return [int(dataset[index][1]) for index in range(len(dataset))]
    if torch.is_tensor(values):
        return [int(v) for v in values.tolist()]
    return [int(v) for v in values]


def non_iid_indices(
    targets: Sequence[int], client_id: int, max_samples: int, seed: int = 42, num_clients: int = 3
) -> list[int]:
    """Return one deterministic, disjoint, 80/20 primary/other partition.

    Every client independently reconstructs the same full allocation and keeps
    only its own indices, so no raw examples or labels need coordinator routing.
    """
    used: set[int] = set()
    partitions: dict[int, list[int]] = {}
    for allocated_client in range(1, num_clients + 1):
        primary = set(PRIMARY_DIGITS.get(allocated_client, tuple(range(10))))
        primary_indices = [
            index for index, label in enumerate(targets) if index not in used and label in primary
        ]
        other_indices = [
            index for index, label in enumerate(targets) if index not in used and label not in primary
        ]
        rng = random.Random(seed + allocated_client * 10_007)
        rng.shuffle(primary_indices)
        rng.shuffle(other_indices)
        primary_count = min(len(primary_indices), int(max_samples * 0.8))
        selected = primary_indices[:primary_count]
        selected.extend(other_indices[: min(len(other_indices), max_samples - len(selected))])
        if len(selected) < max_samples:
            remainder = [index for index in range(len(targets)) if index not in used and index not in selected]
            rng.shuffle(remainder)
            selected.extend(remainder[: max_samples - len(selected)])
        rng.shuffle(selected)
        partitions[allocated_client] = selected
        used.update(selected)
    return partitions[client_id]


def load_client_data(config, client_id: int) -> tuple[DataLoader, dict[int, int]]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST(config.data_dir, train=True, download=True, transform=transform)
    targets = _targets(dataset)
    indices = non_iid_indices(
        targets,
        client_id,
        min(config.max_train_samples_per_client, len(dataset) // config.num_clients),
        config.seed,
        config.num_clients,
    )
    subset = Subset(dataset, indices)
    generator = torch.Generator().manual_seed(config.seed + client_id)
    loader = DataLoader(
        subset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
    )
    return loader, dict(Counter(targets[i] for i in indices))


def load_test_data(config) -> DataLoader:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST(config.data_dir, train=False, download=True, transform=transform)
    count = min(config.max_test_samples, len(dataset))
    subset = Subset(dataset, list(range(count)))
    return DataLoader(subset, batch_size=256, shuffle=False, num_workers=config.num_workers)


def ensure_mnist_available(config) -> None:
    """Download files before processes start; no samples are sent to the coordinator."""
    datasets.MNIST(config.data_dir, train=True, download=True)
    datasets.MNIST(config.data_dir, train=False, download=True)
