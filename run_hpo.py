from __future__ import annotations

import argparse
import json
import multiprocessing as mp

from hpo.optimizer import optimize


def main() -> None:
    parser = argparse.ArgumentParser(description="Run parallel federated-learning HPO trials")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=600)
    args = parser.parse_args()
    result = optimize(args.trials, args.workers, args.rounds, args.max_train_samples)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    mp.freeze_support()
    main()
