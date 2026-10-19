from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from core.config import FLConfig, PROJECT_ROOT
from distributed.coordinator import NetworkCoordinator, create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LAN federated-learning coordinator")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--token", default=os.getenv("FL_SHARED_TOKEN", "change-me"))
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--heartbeat-interval", type=float, default=0.5)
    parser.add_argument("--heartbeat-timeout", type=float, default=3.0)
    parser.add_argument("--round-timeout", type=float, default=300.0)
    parser.add_argument(
        "--failure-mode",
        choices=("none", "process_crash", "hang", "network_timeout"),
        default="process_crash",
    )
    parser.add_argument("--failure-client", type=int, default=2)
    parser.add_argument("--failure-round", type=int, default=4)
    parser.add_argument("--failure-epoch", type=int, default=2)
    parser.add_argument("--failure-batch", type=int, default=37)
    parser.add_argument("--max-train-samples", type=int, default=1200)
    parser.add_argument("--max-test-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default="lan_demo")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.token == "change-me":
        print("WARNING: using demo token 'change-me'; pass --token or set FL_SHARED_TOKEN on every laptop")
    kwargs = {
        "num_clients": args.clients,
        "num_rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "heartbeat_interval": args.heartbeat_interval,
        "heartbeat_timeout": args.heartbeat_timeout,
        "round_timeout": args.round_timeout,
        "failure_mode": args.failure_mode,
        "failure_client_id": args.failure_client,
        "failure_round": args.failure_round,
        "failure_epoch": args.failure_epoch,
        "failure_batch": args.failure_batch,
        "max_train_samples_per_client": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "seed": args.seed,
        "run_id": args.run_id,
    }
    if args.results_dir is not None:
        kwargs["results_dir"] = args.results_dir
    if args.checkpoint_dir is not None:
        kwargs["checkpoint_dir"] = args.checkpoint_dir
    else:
        kwargs["checkpoint_dir"] = PROJECT_ROOT / "checkpoints" / args.run_id
    if args.log_dir is not None:
        kwargs["log_dir"] = args.log_dir
    else:
        kwargs["log_dir"] = PROJECT_ROOT / "logs" / args.run_id
    config = FLConfig(**kwargs)
    coordinator = NetworkCoordinator(config, args.token)
    uvicorn.run(create_app(coordinator), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
