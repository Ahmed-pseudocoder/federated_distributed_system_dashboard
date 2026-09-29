from coordinator.registry import ClientRecord
from coordinator.scheduler import DynamicScheduler


def test_failed_clients_are_not_selected():
    records = [ClientRecord(1), ClientRecord(2, status="FAILED", available=False), ClientRecord(3)]
    assert DynamicScheduler().select(records) == [1, 3]


def test_reliability_affects_priority():
    records = [ClientRecord(1, reliability=0.5), ClientRecord(2, reliability=1.0)]
    assert DynamicScheduler().select(records)[0] == 2

