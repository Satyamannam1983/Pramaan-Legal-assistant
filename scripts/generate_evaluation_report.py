"""Compatibility wrapper that runs, rather than fabricates, an evaluation report."""
from __future__ import annotations

from run_evaluation import main


if __name__ == "__main__":
    raise SystemExit(main())
