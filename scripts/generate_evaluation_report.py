from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))

from app.evaluation.report import build_report, write_report


def main() -> None:
    report = build_report(
        status="NOT RUN",
        dataset_size=0,
        metrics={},
        limitations=["MongoDB-dependent benchmarks were not executed."],
    )
    write_report(report, Path("reports"))
    print("Wrote reports/evaluation_report.json and reports/evaluation_report.md")


if __name__ == "__main__":
    main()
