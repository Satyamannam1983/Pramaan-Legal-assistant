"""Evaluate document-scoped safety and comparison contracts."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))

from app.config import settings
from app.db.connection import check_connection


async def main() -> None:
    if not await check_connection():
        print("Document evaluation: NOT RUN (MongoDB unavailable)")
        return
    print(f"Document evaluation: READY (top_k={settings.document_analysis_top_k})")
    print("Run document-scope and comparison cases against indexed documents.")


if __name__ == "__main__":
    asyncio.run(main())
