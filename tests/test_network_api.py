from core.config import FLConfig
from distributed.coordinator import NetworkCoordinator, create_app
from distributed.schemas import HeartbeatRequest, RegisterRequest


def test_network_api_tracks_remote_worker_and_exposes_routes(tmp_path):
    config = FLConfig(
        num_clients=2,
        num_rounds=1,
        failure_mode="none",
        results_dir=tmp_path / "results",
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
        data_dir=tmp_path / "data",
        run_id="api_test",
    )
    coordinator = NetworkCoordinator(config, "secret")
    app = create_app(coordinator)
    routes = {route.path for route in app.routes}
    assert {"/register", "/heartbeat", "/telemetry", "/task/{client_id}", "/update", "/status"} <= routes
    assert coordinator.public_worker_config(1)["num_clients"] == 2

    registration = coordinator.register(
        RegisterRequest(
            client_id=1,
            pid=1234,
            hostname="laptop-one",
            platform="Windows",
            sample_count=100,
            distribution={"0": 80, "5": 20},
        ),
        "192.168.1.20",
    )
    assert registration["registered"] is True
    coordinator.heartbeat(
        HeartbeatRequest(
            client_id=1,
            pid=1234,
            round=1,
            epoch=1,
            batch=3,
            global_step=3,
            status="training",
        )
    )
    status = coordinator.status()
    assert status["registered_clients"] == [1]
    assert status["workers"][0]["hostname"] == "laptop-one"
    assert status["workers"][0]["remote_addr"] == "192.168.1.20"
    assert status["workers"][0]["current_batch"] == 3
