"""Indexing layer - loads processed JSONL and stores embeddings in MongoDB."""
from .document_indexer import DocumentIndexer, IndexingStats

__all__ = ["DocumentIndexer", "IndexingStats"]
