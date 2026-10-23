"""Compatibility wrapper for retrieval-only evaluation."""
from __future__ import annotations

import sys

from run_evaluation import main as run_main


def main() -> int:
    return run_main([*sys.argv[1:], "--retrieval-only"])


if __name__ == "__main__":
    raise SystemExit(main())
