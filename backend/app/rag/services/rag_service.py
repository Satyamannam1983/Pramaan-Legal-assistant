"""Legal RAG service — orchestrates the full grounded retrieval pipeline.

Pipeline:
  query
    ↓
  Hybrid Retrieval
    ↓
  Cross-Encoder Reranking
    ↓
  Evidence Selection
    ↓
  Context Construction
    ↓
  LLM Generation
    ↓
  Citation Extraction
    ↓
  Grounding Validation
    ↓
  LegalAnswer
"""
from __future__ import annotations

import json
import logging
import os
import time
from uuid import uuid4

from pydantic import ValidationError

from app.config import settings
from app.rag.generation.context_builder import ContextBuilder
from app.rag.generation.evidence_selector import EvidenceBundle, EvidenceSelector
from app.rag.generation.factory import get_llm_provider
from app.rag.generation.schemas import CitationRef, LegalAnswer
from app.rag.citations.citation_service import CitationService
from app.rag.reranking.base import RerankResult
from app.rag.services.retrieval_service import SearchResultItem
from app.rag.validation.grounding_validator import GroundingValidator

logger = logging.getLogger(__name__)


class LegalRAGService:
    """High-level service coordinating retrieval, generation, and validation.

    Usage::

        service = LegalRAGService()
        answer = await service.answer("What is consideration in contract law?")
    """

    def __init__(self) -> None:
        self._retrieval_service = None
        self._evidence_selector = EvidenceSelector()
        self._context_builder = ContextBuilder()
        self._grounding_validator = GroundingValidator()

    def _get_retrieval_service(self):
        if self._retrieval_service is None:
            from app.rag.services.retrieval_service import LegalRetrievalService  # noqa: PLC0415
            self._retrieval_service = LegalRetrievalService()
        return self._retrieval_service

    async def answer(
        self,
        query: str,
        conversation_id: str | None = None,
        search_results: list[SearchResultItem] | None = None,
    ) -> LegalAnswer:
        """Run the full grounded RAG pipeline.

        Returns a LegalAnswer with grounded response, citations, and metadata.
        """
        query = query.strip()
        if not query:
            return LegalAnswer(
                answer="Please provide a valid legal question.",
                confidence="low",
                limitations=["Empty query received."],
            )

        conv_id = conversation_id or str(uuid4())
        start_time = time.monotonic()

        # ── Stage 1: Hybrid Retrieval + Reranking ──────────────────────
        if search_results is None:
            try:
                retrieval = self._get_retrieval_service()
                search_results = await retrieval.search(query, top_k=settings.rerank_top_k)
            except Exception as exc:
                logger.warning("RAG retrieval failed (%s).", type(exc).__name__)
                return LegalAnswer(
                    answer=(
                        "I couldn't retrieve legal sources at this time. "
                        "The search service may be temporarily unavailable."
                    ),
                    confidence="low",
                    limitations=["Legal source retrieval is temporarily unavailable."],
                    abstained=True,
                )

        if not search_results:
            return LegalAnswer(
                answer=(
                    "I couldn't find sufficient evidence in the indexed legal sources "
                    "to answer this question. The query may be outside the scope of "
                    "the currently indexed documents."
                ),
                confidence="low",
                limitations=["No relevant evidence found in indexed sources."],
                abstained=True,
            )

        # ── Stage 2: Convert to RerankResult for evidence selector ─────
        candidates = [
            RerankResult(
                chunk_id=r.chunk_id,
                rerank_score=r.rerank_score or r.rrf_score,
                content=r.content,
                document_id=r.document_id,
                title=r.title,
                section=r.section,
                subsection=r.subsection,
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
            )
            for r in search_results
        ]

        # ── Stage 3: Evidence Selection ────────────────────────────────
        evidence_bundle = self._evidence_selector.select(candidates, query)

        if not evidence_bundle.items:
            return LegalAnswer(
                answer=(
                    "I found some potentially relevant passages, but none met "
                    "the quality threshold for generating a reliable answer. "
                    "Try rephrasing your question or using more specific legal terms."
                ),
                confidence="low",
                limitations=["Evidence below quality threshold."],
                abstained=True,
            )

        # ── Stage 4: Context Construction ──────────────────────────────
        context = self._context_builder.build_context(evidence_bundle)
        user_prompt = self._context_builder.build_user_prompt(query, context)

        # ── Stage 5: LLM Generation ───────────────────────────────────
        try:
            llm = get_llm_provider()
            llm_output = await llm.generate(
                system_prompt=ContextBuilder.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
        except Exception as exc:
            logger.warning("LLM generation failed (%s).", type(exc).__name__)
            provider = os.getenv("LLM_PROVIDER", settings.llm_provider).strip().lower()
            if provider == "openrouter":
                return LegalAnswer(
                    answer=(
                        "OpenRouter answer generation is unavailable. "
                        "No generated legal answer was accepted."
                    ),
                    confidence="low",
                    limitations=[
                        "The configured OpenRouter request failed; no answer was generated."
                    ],
                    generation_mode="failed",
                    abstained=True,
                )
            # Return evidence-based answer without LLM
            return self._build_evidence_only_answer(evidence_bundle, conv_id)

        # ── Stage 6: Parse Structured Output ───────────────────────────
        legal_answer = self._parse_llm_output(llm_output)
        if legal_answer.generation_mode == "failed":
            return legal_answer
        legal_answer.generation_mode = getattr(llm, "provider_name", "configured")

        # ── Stage 7: Citation Validation ───────────────────────────────
        citation_service = CitationService()
        citation_service.load_evidence(evidence_bundle)
        claim_text = "\n".join([legal_answer.answer, *legal_answer.key_points])
        validation = self._grounding_validator.validate(
            legal_answer.answer,
            citation_service,
            has_evidence=True,
            key_points=legal_answer.key_points,
        )

        if not validation.is_grounded:
            logger.warning("Grounding validation failed: %s", validation.summary)
            legal_answer.limitations.append(
                f"Citation validation warning: {validation.summary}"
            )

        # ── Stage 8: Map citations to real metadata ────────────────────
        validated_citations = citation_service.extract_citations(claim_text)
        if (
            not validation.is_grounded
            or not validated_citations
            or not citation_service.validate_citations(validated_citations)
        ):
            logger.warning(
                "Rejecting generated answer because citation validation failed: %s",
                validation.summary,
            )
            return LegalAnswer(
                answer=(
                    "I couldn't verify the generated answer against the retrieved legal "
                    "sources, so I can't provide it reliably."
                ),
                confidence="low",
                limitations=[
                    "Generated answer rejected because its citations were missing or invalid."
                ],
                generation_mode=legal_answer.generation_mode,
                abstained=True,
            )

        legal_answer.citations = [
            CitationRef(
                citation_id=c.citation_id,
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                title=c.title,
                section=c.section,
                subsection=c.subsection,
                page_start=c.page_start,
                page_end=c.page_end,
                document_type=c.document_type,
                court=c.court,
                date=c.date,
                source=c.source,
                source_url=c.source_url,
                relevant_quote=c.relevant_quote,
            )
            for c in validated_citations
        ]
        legal_answer.grounding_valid = True

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "RAG pipeline completed: evidence=%d, citations=%d, "
            "confidence=%s, generation_mode=%s, latency_ms=%.1f",
            len(evidence_bundle.items),
            len(legal_answer.citations),
            legal_answer.confidence,
            legal_answer.generation_mode,
            elapsed_ms,
        )

        return legal_answer

    def _parse_llm_output(self, llm_output: str) -> LegalAnswer:
        """Parse the LLM's JSON output into a LegalAnswer."""
        try:
            data = json.loads(llm_output)
            if not isinstance(data, dict) or not isinstance(data.get("answer"), str) or not data["answer"].strip():
                raise ValueError("LLM response must contain a non-empty answer.")
            return LegalAnswer(
                answer=data["answer"],
                key_points=data.get("key_points", []),
                citations=[],  # Will be populated by citation service
                limitations=data.get("limitations", []),
                confidence=data.get("confidence", "medium"),
            )
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            logger.warning("LLM response parsing failed (%s).", type(exc).__name__)
            return LegalAnswer(
                answer="I couldn't produce a structured, verifiable answer from the retrieved sources.",
                confidence="low",
                limitations=["Generated response was invalid or incomplete; no answer was accepted."],
                generation_mode="failed",
                abstained=True,
            )

    def _build_evidence_only_answer(
        self, bundle: EvidenceBundle, conv_id: str
    ) -> LegalAnswer:
        """Build a response using evidence only, when LLM is unavailable."""
        evidence_texts = []
        fallback_items = bundle.items[:3]
        for item in fallback_items:
            text = f"[{item.citation_id}]"
            if item.title:
                text += f" {item.title}"
            if item.section:
                text += f", {item.section}"
            text += f"\n{item.content[:500]}"
            evidence_texts.append(text)

        answer = (
            "The LLM service is not available. Here are the most relevant "
            "legal passages retrieved for your query:\n\n"
            + "\n\n".join(evidence_texts)
        )

        return LegalAnswer(
            answer=answer,
            key_points=[
                f"LLM unavailable — showing raw evidence. [{fallback_items[0].citation_id}]"
            ],
            citations=[
                CitationRef(
                    citation_id=item.citation_id,
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    title=item.title,
                    section=item.section,
                    subsection=item.subsection,
                    page_start=item.page_start,
                    page_end=item.page_end,
                    document_type=item.document_type,
                    court=item.court,
                    date=item.date,
                    source=item.source,
                    source_url=item.source_url,
                )
                for item in fallback_items
            ],
            limitations=["LLM service unavailable. Answer is based on raw evidence only."],
            confidence="low",
            grounding_valid=True,
            generation_mode="evidence_only",
        )
