"""Keyword retriever — lightweight BM25-style search over the indexed corpus.

BM25 is ideal for legal retrieval because it handles:
- Exact section numbers (Section 302, Article 14)
- Case citations (2024 SCC 123)
- Legal terminology (habeas corpus, certiorari)
- Alphanumeric identifiers (Crl. A. No. 1234/2024)

The index is built once and cached. It rebuilds only when the corpus changes.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter

from app.config import settings
from app.db.connection import check_connection
from app.db.repository import DocumentRepository
from app.rag.retrieval.schemas import KeywordResult

logger = logging.getLogger(__name__)

# Legal-aware tokeniser: preserves section numbers, case citations,
# article numbers, and alphanumeric identifiers while splitting on
# whitespace and punctuation.
_TOKEN_RE = re.compile(
    r"""
    [A-Z][A-Za-z]*[.:](?:\d+[A-Za-z]?(?:\.\d+)*)  # Section/Article patterns: Section 302, Art. 14(1)
    | [A-Z][a-z]+(?:\s+v\.?\s+[A-Z][a-z]+)+         # Case citations: State v. Accused
    | \b[A-Z][a-z]+\s+Act\b                          # Act names: Indian Contract Act
    | \b(?:Section|Section|Rule|Article|Chapter|Part|Clause|Regulation|Notification)\s+\d+[A-Za-z]?(?:\.\d+)*
    | [A-Za-z]+(?:\.\d+)+                             # Dotted identifiers: 302.1.a
    | \b[A-Z][A-Za-z]+\b                              # Capitalized words (legal terms)
    | \b[a-z]+\b                                      # Lowercase words
    | \d+[A-Za-z]?                                    # Numbers with optional suffix
    """,
    re.VERBOSE,
)


def tokenise(text: str) -> list[str]:
    """Tokenise text for BM25, preserving legal identifiers."""
    return _TOKEN_RE.findall(text.lower())


def _tokenise_preserve_case(text: str) -> list[str]:
    """Tokenise preserving case for exact legal term matching."""
    return _TOKEN_RE.findall(text)


class BM25Index:
    """In-memory BM25 index over document chunks.

    Parameters
    ----------
    k1 : float
        Term frequency saturation parameter (default 1.5).
    b : float
        Length normalisation parameter (default 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        # Chunk storage
        self._chunk_ids: list[str] = []
        self._contents: list[str] = []
        self._metadata: list[dict] = []
        # Index data
        self._doc_tokens: list[list[str]] = []
        self._doc_lengths: list[int] = []
        self._avg_doc_length: float = 0.0
        self._df: Counter[str] = Counter()  # document frequency per term
        self._tf: list[Counter[str]] = []  # term frequency per document
        self._n_docs: int = 0

    @property
    def is_empty(self) -> bool:
        return self._n_docs == 0

    def build(self, chunks: list[dict]) -> None:
        """Build the BM25 index from a list of chunk dicts.

        Each chunk dict must contain at minimum:
        - chunk_id, content, document_id
        Plus optional metadata fields.
        """
        self._chunk_ids = []
        self._contents = []
        self._metadata = []
        self._doc_tokens = []
        self._doc_lengths = []
        self._df = Counter()
        self._tf = []

        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            content = chunk["content"]
            tokens = tokenise(content)

            self._chunk_ids.append(chunk_id)
            self._contents.append(content)
            self._metadata.append({
                "document_id": chunk.get("document_id", ""),
                "title": chunk.get("title"),
                "section": chunk.get("section"),
                "subsection": chunk.get("subsection"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "document_type": chunk.get("document_type"),
                "court": chunk.get("court"),
                "date": chunk.get("date"),
                "source": chunk.get("source"),
                "source_url": chunk.get("source_url"),
            })

            tf = Counter(tokens)
            self._tf.append(tf)
            self._doc_tokens.append(tokens)
            self._doc_lengths.append(len(tokens))

            # Update document frequency (unique terms in this document)
            for term in set(tokens):
                self._df[term] += 1

        self._n_docs = len(chunks)
        if self._n_docs > 0:
            self._avg_doc_length = sum(self._doc_lengths) / self._n_docs
        else:
            self._avg_doc_length = 0.0

        logger.info(
            "BM25 index built: %d chunks, %.1f avg tokens/doc, %d unique terms",
            self._n_docs,
            self._avg_doc_length,
            len(self._df),
        )

    def search(self, query: str, top_k: int = 20, document_id: str | None = None) -> list[KeywordResult]:
        """Search the index using BM25 scoring.

        Returns top_k results sorted by descending BM25 score.
        """
        if self._n_docs == 0:
            return []

        query_tokens = tokenise(query)
        if not query_tokens:
            return []

        scores: list[tuple[int, float]] = []

        for doc_idx in range(self._n_docs):
            if document_id and self._metadata[doc_idx].get("document_id") != document_id:
                continue
            score = 0.0
            doc_len = self._doc_lengths[doc_idx]
            tf = self._tf[doc_idx]

            for qt in query_tokens:
                if qt not in self._df:
                    continue

                term_freq = tf.get(qt, 0)
                if term_freq == 0:
                    continue

                # IDF component
                n_containing = self._df[qt]
                idf = math.log(
                    (self._n_docs - n_containing + 0.5) / (n_containing + 0.5) + 1.0
                )

                # TF component with length normalisation
                tf_norm = (
                    term_freq * (self.k1 + 1)
                ) / (
                    term_freq
                    + self.k1
                    * (1 - self.b + self.b * doc_len / self._avg_doc_length)
                )

                score += idf * tf_norm

            if score > 0:
                scores.append((doc_idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        results: list[KeywordResult] = []
        for rank, (doc_idx, bm25_score) in enumerate(scores[:top_k], start=1):
            meta = self._metadata[doc_idx]
            results.append(
                KeywordResult(
                    chunk_id=self._chunk_ids[doc_idx],
                    content=self._contents[doc_idx],
                    document_id=meta["document_id"],
                    title=meta["title"],
                    section=meta["section"],
                    subsection=meta.get("subsection"),
                    page_start=meta["page_start"],
                    page_end=meta["page_end"],
                    document_type=meta["document_type"],
                    court=meta["court"],
                    date=meta["date"],
                    source=meta["source"],
                    source_url=meta["source_url"],
                    keyword_score=bm25_score,
                )
            )

        return results


class KeywordRetriever:
    """BM25 keyword retriever over the MongoDB document_chunks collection.

    The index is built once on first query and cached in memory.
    It rebuilds when the corpus size changes (new documents indexed).
    """

    def __init__(self, top_k: int | None = None) -> None:
        self._default_top_k = top_k or settings.keyword_top_k
        self._index: BM25Index | None = None
        self._indexed_chunk_count: int = -1

    async def _ensure_index(self) -> BM25Index:
        """Build or return the cached BM25 index."""
        if self._index is not None:
            return self._index

        if not await check_connection():
            raise RuntimeError("MongoDB is unavailable for keyword index build.")

        repo = DocumentRepository()
        total_chunks = await repo.count_chunks()

        if total_chunks == 0:
            self._index = BM25Index()
            return self._index

        # Fetch all chunks with their document titles
        from app.db.connection import get_db  # noqa: PLC0415
        from app.db.models import CHUNKS_COLLECTION, DOCUMENTS_COLLECTION  # noqa: PLC0415

        db = get_db()
        cursor = db[CHUNKS_COLLECTION].find(
            {},
            {
                "_id": 0,
                "chunk_id": 1,
                "content": 1,
                "document_id": 1,
                "section": 1,
                "subsection": 1,
                "page_start": 1,
                "page_end": 1,
                "document_type": 1,
                "court": 1,
                "date": 1,
                "source": 1,
                "source_url": 1,
            },
        )
        chunks = await cursor.to_list(length=100_000)

        # Fetch document titles
        doc_ids = list({c["document_id"] for c in chunks})
        doc_cursor = db[DOCUMENTS_COLLECTION].find(
            {"document_id": {"$in": doc_ids}},
            {"_id": 0, "document_id": 1, "title": 1},
        )
        docs = await doc_cursor.to_list(length=None)
        title_map: dict[str, str | None] = {d["document_id"]: d.get("title") for d in docs}

        # Enrich chunks with titles
        for chunk in chunks:
            chunk["title"] = title_map.get(chunk["document_id"])

        self._index = BM25Index()
        self._index.build(chunks)
        self._indexed_chunk_count = total_chunks

        return self._index

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> list[KeywordResult]:
        """Return top_k chunks matching the query via BM25.

        Raises
        ------
        ValueError
            If the query is empty.
        RuntimeError
            If MongoDB is unavailable.
        """
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")

        k = top_k if top_k is not None else self._default_top_k

        index = await self._ensure_index()
        return index.search(query, top_k=k, document_id=document_id)

    async def invalidate(self) -> None:
        """Force rebuild on next query."""
        self._index = None
        self._indexed_chunk_count = -1
