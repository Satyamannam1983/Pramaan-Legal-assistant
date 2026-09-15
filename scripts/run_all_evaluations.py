"""Run evaluation entry points without converting unavailable services to zeros."""
from __future__ import annotations

import subprocess
import sys


def main() -> int:
    scripts = [
        "scripts/evaluate_retrieval.py",
        "scripts/evaluate_rag.py",
        "scripts/evaluate_documents.py",
        "scripts/generate_evaluation_report.py",
    ]
    print("=" * 40)
    print("LEGAL RAG EVALUATION")
    print("=" * 40)
    for script in scripts:
        print(f"\n>> {script}")
        result = subprocess.run([sys.executable, script], check=False)
        if result.returncode != 0 and script != "scripts/generate_evaluation_report.py":
            print("Status: NOT RUN or failed safely")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
