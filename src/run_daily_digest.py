"""Entrypoint for the daily-digest systemd timer.

Usage:
    python src/run_daily_digest.py        # send the digest
    python src/run_daily_digest.py --dry  # build and print only

Runs once per call (oneshot). Returns non-zero only if both build
AND alert dispatch failed completely; partial failures (e.g.
Pushover delivered, email failed) still return 0.
"""
from __future__ import annotations

import argparse
import logging
import sys

from common.daily_digest import build_digest, send_digest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true",
                     help="Build the digest and print to stdout, don't send")
    args = ap.parse_args()

    if args.dry:
        print(build_digest())
        return 0

    return 0 if send_digest() else 1


if __name__ == "__main__":
    sys.exit(main())
