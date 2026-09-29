from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove generated run artifacts")
    parser.add_argument("--yes", action="store_true", help="confirm deletion")
    args = parser.parse_args()
    if not args.yes:
        raise SystemExit("No files removed. Re-run with --yes to confirm.")
    root = Path(__file__).resolve().parents[1]
    targets = []
    for directory, pattern in ((root / "checkpoints", "*.pt"), (root / "results", "*.db"), (root / "logs", "*.log")):
        if directory.exists():
            targets.extend(directory.glob(pattern))
    for target in targets:
        target.unlink()
        print(f"removed {target}")


if __name__ == "__main__":
    main()
