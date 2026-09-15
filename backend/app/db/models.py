"""MongoDB document schemas (Pydantic, not SQLAlchemy)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# MongoDB collection schemas — used for validation before insert
# ---------------------------------------------------------------------------

DOCUMENTS_COLLECTION = "documents"
CHUNKS_COLLECTION = "document_chunks"


# Document shape stored in `documents` collection
DOCUMENT_SCHEMA = {
    "document_id": str,   # unique
    "title": str,
    "document_type": str,
    "source": str,
    "source_url": str,
    "jurisdiction": str,
    "court": str,
    "case_name": str,
    "case_number": str,
    "citation": str,
    "date": str,
    "act_name": str,
    "language": str,
    "metadata": dict,
    "created_at": datetime,
}

# Chunk shape stored in `document_chunks` collection
CHUNK_SCHEMA = {
    "chunk_id": str,       # unique, format: "{document_id}:{index}"
    "document_id": str,    # references documents.document_id
    "content": str,
    "embedding": list,     # list[float] — stored as BSON array
    "chunk_index": int,
    "section": str,
    "subsection": str,
    "page_start": int,
    "page_end": int,
    "document_type": str,
    "court": str,
    "date": str,
    "source": str,
    "source_url": str,
    "metadata": dict,
    "created_at": datetime,
}
