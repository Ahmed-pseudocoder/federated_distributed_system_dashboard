from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

from coordinator.server import run_federated
from core.config import FLConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fault-tolerant federated MNIST MVP")
    parser.add_argument("--mode", choices=("basic", "fault-tolerant"), default="fault-tolerant")
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--heartbeat-interval", type=float, default=0.5)
    parser.add_argument("--heartbeat-timeout", type=float, default=3.0)
    parser.add_argument("--round-timeout", type=float, default=180.0)
    parser.add_argument("--failure-mode", choices=("none", "process_crash", "hang", "network_timeout"), default=None)
    parser.add_argument("--failure-client", type=int, default=2)
    parser.add_argument("--failure-round", type=int, default=4)
    parser.add_argument("--failure-epoch", type=int, default=2)
    parser.add_argument("--failure-batch", type=int, default=37)
    parser.add_argument("--max-train-samples", type=int, default=1200)
    parser.add_argument("--max-test-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default="default")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> FLConfig:
    failure_mode = args.failure_mode or ("none" if args.mode == "basic" else "process_crash")
    kwargs = {
        "num_clients": args.clients,
        "num_rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "heartbeat_interval": args.heartbeat_interval,
        "heartbeat_timeout": args.heartbeat_timeout,
        "round_timeout": args.round_timeout,
        "failure_mode": failure_mode,
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
    if args.log_dir is not None:
        kwargs["log_dir"] = args.log_dir
    return FLConfig(**kwargs)


def main() -> None:
    args = build_parser().parse_args()
    summary = run_federated(config_from_args(args))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    mp.freeze_support()
    main()

