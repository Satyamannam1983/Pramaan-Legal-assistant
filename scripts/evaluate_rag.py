"""Compatibility wrapper for answer-flow evaluation."""
from __future__ import annotations

import sys

from run_evaluation import main as run_main


def main() -> int:
    return run_main([*sys.argv[1:], "--answers-only"])


if __name__ == "__main__":
    raise SystemExit(main())
