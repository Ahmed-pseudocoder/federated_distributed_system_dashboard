from coordinator.event_logger import EventLogger


def test_event_and_telemetry_are_persisted(tmp_path):
    logger = EventLogger(tmp_path / "events.db")
    event_id = logger.log_event("SYSTEM_STARTED", answer=42)
    logger.update_event_payload(event_id, {"answer": 43})
    logger.record_telemetry(
        {
            "timestamp": "2026-09-30T00:00:00.000Z",
            "client_id": 2,
            "round": 4,
            "epoch": 2,
            "batch": 37,
            "global_step": 225,
            "loss": 0.18,
            "accuracy": 0.94,
            "status": "training",
            "pid": 123,
        }
    )
    events = logger.fetch_all("SELECT * FROM events")
    telemetry = logger.fetch_all("SELECT * FROM telemetry")
    assert events[0]["event_type"] == "SYSTEM_STARTED"
    assert '"answer": 43' in events[0]["payload_json"]
    assert telemetry[0]["batch"] == 37
    assert telemetry[0]["global_step"] == 225

