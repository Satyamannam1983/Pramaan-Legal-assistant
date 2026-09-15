"""index_documents.py — Embed processed JSONL files and store in MongoDB.

Usage
-----
  python scripts/index_documents.py
  python scripts/index_documents.py --processed-dir data/processed
  python scripts/index_documents.py --no-progress
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))

from app.config import settings
from app.db.connection import init_db, check_connection
from app.rag.indexing.document_indexer import DocumentIndexer
from app.db.repository import DocumentRepository


def _progress_bar(current: int, total: int, width: int = 40) -> str:
    filled = int(width * current / total) if total else 0
    bar = "#" * filled + "-" * (width - filled)
    return f"\r[{bar}] {current}/{total}"


async def main(processed_dir: Path, show_progress: bool) -> None:
    print("=" * 60)
    print("Pramaan — Document Indexing (MongoDB)")
    print("=" * 60)

    print(f"\nDatabase URL : {settings.database_url}")
    print("\nConnecting to MongoDB…")
    if not await check_connection():
        print("[ERR] Cannot connect to MongoDB.")
        print("  Set DATABASE_URL in your .env file:")
        print("    Local:  mongodb://localhost:27017/pramaan")
        print("    Atlas:  mongodb+srv://user:pass@cluster.mongodb.net/pramaan")
        sys.exit(1)

    print("[OK] MongoDB connected")
    await init_db()
    print("[OK] Indexes ready")

    # Discover JSONL files
    jsonl_files = sorted(processed_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"\nNo processed JSONL files found in {processed_dir}")
        print("Run: python scripts/process_documents.py")
        sys.exit(1)

    print(f"\nProcessed directory : {processed_dir}")
    print(f"JSONL files found   : {len(jsonl_files)}")
    for f in jsonl_files:
        print(f"  • {f.name}")

    # DB state before
    repo = DocumentRepository()
    pre_docs = await repo.count_documents()
    pre_chunks = await repo.count_chunks()
    print(f"\nCurrent DB state    : {pre_docs} documents, {pre_chunks} chunks")

    print(f"\nEmbedding provider  : {settings.embedding_provider}")
    print(f"Embedding model     : {settings.embedding_model}")
    print(f"Batch size          : {settings.embedding_batch_size}")
    print("\nGenerating embeddings and indexing chunks…\n")

    start = time.perf_counter()

    def on_progress(current: int, total: int) -> None:
        if show_progress:
            print(_progress_bar(current, total), end="", flush=True)

    indexer = DocumentIndexer(processed_dir=processed_dir)
    stats = await indexer.index_all(progress_callback=on_progress)

    if show_progress:
        print()

    elapsed = time.perf_counter() - start

    post_docs = await repo.count_documents()
    post_chunks = await repo.count_chunks()

    print("\n" + "=" * 60)
    print("Indexing complete")
    print("=" * 60)
    print(f"  Documents found    : {stats.documents_found}")
    print(f"  Documents indexed  : {stats.documents_indexed}")
    print(f"  Chunks found       : {stats.chunks_found}")
    print(f"  Chunks indexed     : {stats.chunks_indexed}")
    print(f"  Time elapsed       : {elapsed:.1f}s")
    print(f"\n  DB total           : {post_docs} documents, {post_chunks} chunks")

    if stats.errors:
        print(f"\n  Errors ({len(stats.errors)}):")
        for err in stats.errors:
            print(f"    [ERR] {err}")

    if stats.chunks_indexed == 0:
        print("\n  [WARN]  No chunks were indexed.")
        print("     Check that JSONL files contain 'sample_chunks' or 'all_chunks'.")
        sys.exit(1)

    print("\nDatabase indexing complete. [OK]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default=settings.processed_dir)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(Path(args.processed_dir), show_progress=not args.no_progress))
