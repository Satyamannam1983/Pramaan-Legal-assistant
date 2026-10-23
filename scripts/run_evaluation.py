"""Run Pramaan's reproducible dataset-driven evaluation workflow."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))

from app.evaluation.runner import run_default_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieval-dataset",
        default="data/evaluation/retrieval_gold.json",
        help="Versioned dataset containing relevant chunk/document labels.",
    )
    parser.add_argument(
        "--answer-dataset",
        default="data/evaluation/rag_gold.json",
        help="Versioned answer dataset containing expected abstention/characteristics.",
    )
    parser.add_argument("--output-dir", default="reports")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--retrieval-only", action="store_true")
    mode.add_argument("--answers-only", action="store_true")
    return parser


async def _run(args: argparse.Namespace) -> int:
    report = await run_default_evaluation(
        args.retrieval_dataset,
        args.answer_dataset,
        args.output_dir,
        retrieval_only=args.retrieval_only,
        answers_only=args.answers_only,
    )
    print(f"Evaluation status: {report['status']}")
    print(f"Report JSON: {Path(args.output_dir) / 'evaluation_report.json'}")
    print(f"Report Markdown: {Path(args.output_dir) / 'evaluation_report.md'}")
    for name, stage in report["evaluations"].items():
        print(f"  {name}: {stage['status']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except (OSError, ValueError) as exc:
        print(f"Evaluation could not start: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
