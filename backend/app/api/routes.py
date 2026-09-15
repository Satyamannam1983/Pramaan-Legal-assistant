from pathlib import Path
import logging
from uuid import uuid4
from fastapi import APIRouter, File, HTTPException, UploadFile
from app.config import settings
from app.schemas import (
    ChatRequest,
    ChatResponse,
    CitationRef,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SearchDebugInfo,
    SemanticSearchRequest,
    SemanticSearchResponse,
    SemanticSearchResult,
    UploadResponse,
    DocumentProcessResponse,
)
from app.rag.loaders import LoaderError
from app.services.document_processing import DocumentProcessingService
from app.db.repository import DocumentRepository, DocumentData, ChunkData
from app.rag.document_analysis import DocumentAnalyzer, ComparisonService
from app.rag.document_analysis.schemas import DocumentQuestion, CompareRequest, DocumentSummary

router = APIRouter()
logger = logging.getLogger(__name__)

# Singleton services
document_processing_service = DocumentProcessingService()


async def _read_upload(file: UploadFile) -> bytes:
    """Read an upload in bounded chunks so oversized files never accumulate unchecked."""
    limit = settings.max_upload_size_mb * 1024 * 1024
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail="Uploaded document exceeds the configured size limit.")
        chunks.append(chunk)
    return b"".join(chunks)


def _get_retrieval_service():
    """Lazy-initialise the LegalRetrievalService singleton."""
    if not hasattr(router, "_retrieval_service"):
        from app.rag.services.retrieval_service import LegalRetrievalService  # noqa: PLC0415
        router._retrieval_service = LegalRetrievalService()
    return router._retrieval_service


def _get_rag_service():
    """Lazy-initialise the LegalRAGService singleton."""
    if not hasattr(router, "_rag_service"):
        from app.rag.services.rag_service import LegalRAGService  # noqa: PLC0415
        router._rag_service = LegalRAGService()
    return router._rag_service


def _get_research_graph():
    """Lazy-compile the bounded LangGraph workflow once per process."""
    if not hasattr(router, "_research_graph"):
        from app.rag.graph import build_legal_research_graph  # noqa: PLC0415
        router._research_graph = build_legal_research_graph()
    return router._research_graph


@router.get("/health")
async def health() -> dict:
    from app.db.connection import check_connection  # noqa: PLC0415
    db_ok = await check_connection()
    return {
        "status": "ok",
        "environment": settings.app_env,
        "database": "connected" if db_ok else "unavailable (set DATABASE_URL in .env)",
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "reranker_provider": settings.reranker_provider or "none",
        "reranker_model": settings.reranker_model or "none",
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
    }


@router.get("/live")
async def liveness() -> dict:
    return {"status": "ok"}


@router.get("/ready")
async def readiness() -> dict:
    from app.db.connection import check_connection  # noqa: PLC0415
    if not await check_connection():
        raise HTTPException(status_code=503, detail="Database is unavailable.")
    return {"status": "ready"}


@router.get("/evaluation")
async def evaluation_report() -> dict:
    """Return the local evaluation report without exposing secrets."""
    report_path = Path("reports/evaluation_report.json")
    if not report_path.exists():
        return {"status": "NOT RUN", "dataset_size": 0, "metrics": {}, "limitations": ["No report generated yet."]}
    try:
        import json
        return json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail="Evaluation report unavailable.") from exc


# ---------------------------------------------------------------------------
# Hybrid search endpoint (primary)
# ---------------------------------------------------------------------------

@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """Hybrid search: vector + keyword retrieval with RRF fusion and reranking."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be empty.")
    try:
        service = _get_retrieval_service()
        results = await service.search(query, top_k=request.top_k, document_id=request.document_id)
        if request.document_type:
            results = [r for r in results if (r.document_type or "").upper() == request.document_type.upper()]
        if request.court:
            results = [r for r in results if request.court.lower() in (r.court or "").lower()]
        if request.date_from:
            results = [r for r in results if r.date and r.date >= request.date_from]
        if request.date_to:
            results = [r for r in results if r.date and r.date <= request.date_to]
        return SearchResponse(
            query=query,
            results=[
                SearchResultItem(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    document_id=r.document_id,
                    title=r.title,
                    section=r.section,
                    page_start=r.page_start,
                    page_end=r.page_end,
                    document_type=r.document_type,
                    court=r.court,
                    date=r.date,
                    source=r.source,
                    source_url=r.source_url,
                    vector_score=r.vector_score,
                    keyword_score=r.keyword_score,
                    rrf_score=r.rrf_score,
                    rerank_score=r.rerank_score,
                    final_rank=r.final_rank,
                )
                for r in results
            ],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Debug endpoint — exposes retrieval stages
# ---------------------------------------------------------------------------

@router.post("/search/debug", response_model=SearchDebugInfo)
async def search_debug(request: SearchRequest) -> SearchDebugInfo:
    """Debug endpoint exposing all retrieval stages. Dev-only."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be empty.")

    if settings.app_env == "production":
        raise HTTPException(status_code=404, detail="Debug endpoint not available in production.")

    try:
        service = _get_retrieval_service()
        debug = await service.search_debug(query, top_k=request.top_k)

        def _to_item(r) -> SearchResultItem:
            return SearchResultItem(
                chunk_id=r.chunk_id,
                content=r.content[:500] + "..." if len(r.content) > 500 else r.content,
                document_id=r.document_id,
                title=r.title,
                section=r.section,
                page_start=r.page_start,
                page_end=r.page_end,
                document_type=r.document_type,
                court=r.court,
                date=r.date,
                source=r.source,
                source_url=r.source_url,
                vector_score=r.vector_score,
                keyword_score=r.keyword_score,
                rrf_score=r.rrf_score,
                rerank_score=getattr(r, "rerank_score", 0.0),
                final_rank=getattr(r, "final_rank", None),
            )

        return SearchDebugInfo(
            query=debug.query,
            vector_results=[_to_item(r) for r in debug.vector_results],
            keyword_results=[_to_item(r) for r in debug.keyword_results],
            fused_results=[_to_item(r) for r in debug.fused_results],
            reranked_results=[_to_item(r) for r in debug.reranked_results],
            final_results=[_to_item(r) for r in debug.final_results],
            vector_time_ms=debug.vector_time_ms,
            keyword_time_ms=debug.keyword_time_ms,
            fusion_time_ms=debug.fusion_time_ms,
            rerank_time_ms=debug.rerank_time_ms,
            total_time_ms=debug.total_time_ms,
            reranker_available=debug.reranker_available,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Chat / RAG endpoint (Step 5)
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Bounded LangGraph workflow for grounded legal research."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be empty.")
    try:
        graph = _get_research_graph()
        state = await graph.ainvoke({"query": query, "original_query": query})
        answer = state.get("answer", {})
        return ChatResponse(
            answer=answer.get("answer", ""),
            key_points=answer.get("key_points", []),
            citations=[
                CitationRef(
                    citation_id=c.citation_id,
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    title=c.title,
                    section=c.section,
                    page_start=c.page_start,
                    page_end=c.page_end,
                    source_url=c.source_url,
                    relevant_quote=c.relevant_quote,
                )
                for c in [CitationRef.model_validate(c) for c in state.get("citations", [])]
            ],
            limitations=state.get("limitations", answer.get("limitations", [])),
            confidence=answer.get("confidence", "low"),
            conversation_id=request.conversation_id or "",
            retrieval_attempts=state.get("retrieval_attempt", 0),
            strategies=state.get("strategies", []),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/chat/debug")
async def chat_debug(request: ChatRequest) -> dict:
    """Development-only graph execution summary without prompts or credentials."""
    if settings.app_env == "production":
        raise HTTPException(status_code=404, detail="Debug endpoint not available in production.")
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be empty.")
    try:
        state = await _get_research_graph().ainvoke(
            {"query": query, "original_query": query}
        )
        answer = state.get("answer", {})
        return {
            "query": query,
            "query_type": state.get("query_type", "UNKNOWN"),
            "retrieval_attempts": state.get("retrieval_attempt", 0),
            "strategies": state.get("strategies", []),
            "evidence_sufficient": state.get("evidence_sufficient", False),
            "nodes_executed": state.get("nodes_executed", []),
            "answer": answer.get("answer", ""),
            "citations": state.get("citations", []),
        }
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Legacy semantic search endpoint (backward compat)
# ---------------------------------------------------------------------------

@router.post("/search/semantic", response_model=SemanticSearchResponse)
async def search_semantic(request: SemanticSearchRequest) -> SemanticSearchResponse:
    """Legacy vector-only semantic search. Prefer /search for hybrid retrieval."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be empty.")
    try:
        from app.rag.retrieval.vector_retriever import VectorRetriever  # noqa: PLC0415
        retriever = VectorRetriever()
        results = await retriever.retrieve(query, top_k=request.top_k)
        return SemanticSearchResponse(
            query=query,
            results=[
                SemanticSearchResult(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    document_id=r.document_id,
                    title=r.title,
                    section=r.section,
                    page_start=r.page_start,
                    page_end=r.page_end,
                    document_type=r.document_type,
                    court=r.court,
                    date=r.date,
                    source=r.source,
                    source_url=r.source_url,
                    similarity_score=r.vector_score,
                )
                for r in results
            ],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Document endpoints
# ---------------------------------------------------------------------------

@router.post("/documents/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    safe_name = Path(file.filename or "upload.bin").name
    extension = Path(safe_name).suffix.lower()
    if extension not in {".pdf", ".txt", ".html", ".htm", ".json"}:
        raise HTTPException(status_code=415, detail="Unsupported document type.")
    content = await _read_upload(file)
    target = Path(settings.upload_dir)
    target.mkdir(parents=True, exist_ok=True)
    storage_name = f"{uuid4().hex}{extension}"
    (target / storage_name).write_bytes(content)
    processed = await document_processing_service.process(safe_name, content, include_all_chunks=True)
    try:
        from app.rag.embeddings.factory import get_embedding_service
        from app.db.connection import check_connection
        if await check_connection():
            metadata = processed.metadata
            repo = DocumentRepository()
            await repo.upsert_document(DocumentData(
                document_id=metadata.document_id, title=metadata.title or safe_name,
                document_type=metadata.document_type, source=metadata.source,
                source_url=metadata.source_url, jurisdiction=metadata.jurisdiction,
                court=metadata.court, case_name=metadata.case_name,
                case_number=metadata.case_number, citation=metadata.citation,
                date=metadata.date, act_name=metadata.act_name, language=metadata.language,
                metadata=metadata.metadata,
            ))
            chunks = processed.all_chunks or []
            embeddings = get_embedding_service().embed_documents([c.content for c in chunks])
            await repo.upsert_chunks([
                ChunkData(
                    chunk_id=c.chunk_id, document_id=c.document_id, content=c.content,
                    embedding=e, chunk_index=c.chunk_index, section=c.section,
                    subsection=c.subsection, page_start=c.page_start, page_end=c.page_end,
                    document_type=c.document_type, court=c.court, date=c.date,
                    source=c.source, source_url=c.source_url,
                ) for c, e in zip(chunks, embeddings)
            ])
    except Exception as exc:
        logger.warning("Upload stored but indexing was unavailable: %s", exc)
    logger.info("Accepted document upload: %s (%d bytes)", safe_name, len(content))
    return UploadResponse(document_id=processed.metadata.document_id, filename=safe_name, content_type=file.content_type, size=len(content))


@router.post("/documents/process", response_model=DocumentProcessResponse)
async def process_document(file: UploadFile = File(...)) -> DocumentProcessResponse:
    safe_name = Path(file.filename or "upload.txt").name
    if Path(safe_name).suffix.lower() not in {".pdf", ".txt", ".html", ".htm", ".json"}:
        raise HTTPException(status_code=415, detail="Unsupported document type.")
    content = await _read_upload(file)
    try:
        return await document_processing_service.process(
            safe_name, content
        )
    except LoaderError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents() -> list[DocumentSummary]:
    try:
        rows = await DocumentRepository().list_documents()
        result = []
        repo = DocumentRepository()
        for row in rows:
            result.append(DocumentSummary(
                document_id=row["document_id"], title=row.get("title"),
                document_type=row.get("document_type"), court=row.get("court"),
                date=row.get("date"), chunk_count=await repo.count_chunks(row["document_id"]),
                created_at=row.get("created_at").isoformat() if row.get("created_at") else None,
            ))
        return result
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Document store unavailable.") from exc


@router.get("/documents/{document_id}")
async def get_document(document_id: str) -> dict:
    if not document_id or "/" in document_id or "\\" in document_id:
        raise HTTPException(status_code=422, detail="Invalid document ID.")
    row = await DocumentRepository().get_document(document_id)
    if not row:
        raise HTTPException(status_code=404, detail="Document not found.")
    row["chunk_count"] = await DocumentRepository().count_chunks(document_id)
    return row


@router.get("/documents/{document_id}/analysis")
async def analyze_document(document_id: str) -> dict:
    try:
        return (await DocumentAnalyzer().analyze(document_id)).model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Document analysis unavailable.") from exc


@router.post("/documents/{document_id}/ask", response_model=ChatResponse)
async def ask_document(document_id: str, request: DocumentQuestion) -> ChatResponse:
    if not await DocumentRepository().document_exists(document_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    try:
        results = await _get_retrieval_service().search(
            request.query.strip(), top_k=settings.final_top_k, document_id=document_id
        )
        if not results:
            return ChatResponse(
                answer="I couldn't find sufficient evidence in this document to answer reliably.",
                limitations=["No relevant indexed evidence was found in the requested document."],
            )
        answer = await _get_rag_service().answer(request.query, search_results=results)
        return ChatResponse(
            answer=answer.answer, key_points=answer.key_points,
            citations=[CitationRef.model_validate(c.model_dump()) for c in answer.citations],
            limitations=answer.limitations, confidence=answer.confidence,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/documents/compare")
async def compare_documents(request: CompareRequest) -> dict:
    if len(request.document_ids) != settings.max_comparison_documents:
        raise HTTPException(status_code=422, detail="Exactly two document IDs are required.")
    try:
        return (await ComparisonService().compare(request.document_ids)).model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Document comparison unavailable.") from exc
