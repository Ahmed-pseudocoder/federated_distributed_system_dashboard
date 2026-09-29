from core.config import FLConfig
from coordinator.server import run_federated


if __name__ == "__main__":
    run_federated(FLConfig(failure_mode="process_crash", run_id="dynamic_allocation"))

