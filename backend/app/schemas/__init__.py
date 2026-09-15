from pydantic import BaseModel, Field
from datetime import datetime


# ---------------------------------------------------------------------------
# Document models
# ---------------------------------------------------------------------------

class DocumentMetadata(BaseModel):
    document_id: str
    filename: str | None = None
    title: str | None = None
    document_type: str | None = None
    source: str | None = None
    source_url: str | None = None
    jurisdiction: str | None = None
    court: str | None = None
    case_name: str | None = None
    case_number: str | None = None
    citation: str | None = None
    date: str | None = None
    act_name: str | None = None
    section: str | None = None
    language: str | None = None
    content: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    content_type: str | None = None
    size: int = 0
    source_type: str | None = None
    created_at: datetime


class DocumentPage(BaseModel):
    page_number: int
    text: str


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    text: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    subsection: str | None = None
    document_type: str | None = None
    court: str | None = None
    date: str | None = None
    source: str | None = None
    source_url: str | None = None
    chunk_index: int


class DocumentProcessResponse(BaseModel):
    metadata: DocumentMetadata
    pages: list[DocumentPage]
    chunk_count: int
    sample_chunks: list[DocumentChunk]
    all_chunks: list[DocumentChunk] | None = None


# ---------------------------------------------------------------------------
# Search — hybrid retrieval (Step 4)
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, description="Natural-language legal query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to return")
    document_id: str | None = None
    document_type: str | None = None
    court: str | None = None
    date_from: str | None = None
    date_to: str | None = None


class SearchResultItem(BaseModel):
    """A single search result with multi-signal scores."""
    chunk_id: str
    content: str
    document_id: str
    title: str | None = None
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    document_type: str | None = None
    court: str | None = None
    date: str | None = None
    source: str | None = None
    source_url: str | None = None
    vector_score: float = 0.0
    keyword_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_rank: int | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]


class SearchDebugInfo(BaseModel):
    """Detailed retrieval stage information for debugging."""
    query: str
    vector_results: list[SearchResultItem] = []
    keyword_results: list[SearchResultItem] = []
    fused_results: list[SearchResultItem] = []
    reranked_results: list[SearchResultItem] = []
    final_results: list[SearchResultItem] = []
    vector_time_ms: float = 0.0
    keyword_time_ms: float = 0.0
    fusion_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    total_time_ms: float = 0.0
    reranker_available: bool = True


# ---------------------------------------------------------------------------
# Legacy semantic search schemas (backward compat)
# ---------------------------------------------------------------------------

class SemanticSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, description="Natural-language legal query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to return")


class SemanticSearchResult(BaseModel):
    """A single retrieved chunk with full provenance."""
    chunk_id: str
    content: str
    document_id: str
    title: str | None = None
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    document_type: str | None = None
    court: str | None = None
    date: str | None = None
    source: str | None = None
    source_url: str | None = None
    similarity_score: float


class SemanticSearchResponse(BaseModel):
    query: str
    results: list[SemanticSearchResult]


# ---------------------------------------------------------------------------
# Chat / RAG (Step 5)
# ---------------------------------------------------------------------------

class CitationRef(BaseModel):
    """A citation referenced in the answer."""
    citation_id: str
    chunk_id: str = ""
    document_id: str = ""
    title: str | None = None
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    source_url: str | None = None
    relevant_quote: str | None = None


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000, description="Legal question")
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    key_points: list[str] = []
    citations: list[CitationRef] = []
    limitations: list[str] = []
    confidence: str = "low"
    conversation_id: str = ""
    retrieval_attempts: int = 0
    strategies: list[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    document_id: str | None = None
    filename: str
    content_type: str | None
    size: int
    status: str = "accepted"
