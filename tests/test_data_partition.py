from collections import Counter

from core.data import PRIMARY_DIGITS, non_iid_indices


def test_non_iid_partition_is_deterministic_and_primary_heavy():
    targets = list(range(10)) * 100
    first = non_iid_indices(targets, client_id=2, max_samples=200, seed=42)
    second = non_iid_indices(targets, client_id=2, max_samples=200, seed=42)
    counts = Counter(targets[index] for index in first)
    primary_count = sum(counts[digit] for digit in PRIMARY_DIGITS[2])
    assert first == second
    assert len(first) == 200
    assert primary_count == 160


def test_client_partitions_are_disjoint():
    targets = list(range(10)) * 200
    partitions = [set(non_iid_indices(targets, client_id, 200, seed=42)) for client_id in (1, 2, 3)]
    assert partitions[0].isdisjoint(partitions[1])
    assert partitions[0].isdisjoint(partitions[2])
    assert partitions[1].isdisjoint(partitions[2])

