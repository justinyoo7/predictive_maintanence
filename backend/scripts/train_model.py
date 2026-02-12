#!/usr/bin/env python3
"""Train model artifacts from Oracle-mock CSV data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.modeling import train_and_save_model  # noqa: E402
from app.oracle_client import oracle_client  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train predictive parts models.")
    parser.add_argument(
        "--artifact_name",
        type=str,
        default=None,
        help="Optional explicit artifact id/name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    oracle_client.reload()
    result = train_and_save_model(oracle_client, artifact_name=args.artifact_name)
    print(
        f"Trained model artifact {result['artifact_id']} "
        f"from {result['n_rows']} rows at {result['path']}"
    )


if __name__ == "__main__":
    main()
