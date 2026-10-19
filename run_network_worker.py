from __future__ import annotations

import argparse
import os
from pathlib import Path

from distributed.worker import run_network_worker


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one federated worker on a LAN laptop")
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--coordinator", required=True, help="Example: http://192.168.1.10:8000")
    parser.add_argument("--token", default=os.getenv("FL_SHARED_TOKEN", "change-me"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    args = parser.parse_args()
    run_network_worker(
        client_id=args.client_id,
        coordinator_url=args.coordinator,
        token=args.token,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        poll_interval=args.poll_interval,
    )


if __name__ == "__main__":
    main()
