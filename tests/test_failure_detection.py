from coordinator.fault_detector import FaultDetector


def test_timeout_detection():
    detector = FaultDetector(3.0)
    assert detector.is_timed_out(last_heartbeat=10.0, now=13.001)
    assert not detector.is_timed_out(last_heartbeat=10.0, now=12.999)
