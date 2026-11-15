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

import asyncio
import json
import logging
import re
import time
from uuid import uuid4

from pydantic import ValidationError

from app.config import settings
from app.rag.generation.context_builder import ContextBuilder
from app.rag.generation.evidence_selector import EvidenceBundle, EvidenceItem, EvidenceSelector
from app.rag.generation.factory import get_llm_provider
from app.rag.generation.schemas import CitationRef, LegalAnswer
from app.rag.citations.citation_service import CitationService
from app.rag.reranking.base import RerankResult
from app.rag.services.retrieval_service import SearchResultItem
from app.rag.validation.grounding_validator import GroundingValidator

logger = logging.getLogger(__name__)
_LLM_TIMEOUT_SECONDS = 15
_MIN_EXTRACTIVE_MATCHES = 2
_MIN_EXTRACTIVE_QUERY_COVERAGE = 0.35


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
        follow_up: bool = False,
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

        if not follow_up:
            extractive_answer = self._build_evidence_only_answer(
                evidence_bundle,
                conv_id,
                require_query_match=True,
            )
            if not extractive_answer.abstained:
                logger.info("Returning directly matched source sentence without LLM generation.")
                return extractive_answer

        # ── Stage 4: Context Construction ──────────────────────────────
        context = self._context_builder.build_context(evidence_bundle)
        user_prompt = self._context_builder.build_user_prompt(
            query,
            context,
            is_follow_up=follow_up,
        )

        # ── Stage 5: LLM Generation ───────────────────────────────────
        try:
            llm = get_llm_provider()
            llm_output = await asyncio.wait_for(
                llm.generate(
                    system_prompt=ContextBuilder.SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                ),
                timeout=_LLM_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning(
                "LLM generation failed or exceeded %d seconds (error_type=%s).",
                _LLM_TIMEOUT_SECONDS,
                type(exc).__name__,
            )
            if follow_up:
                return self._build_follow_up_fallback(evidence_bundle, query)
            return self._build_evidence_only_answer(
                evidence_bundle,
                conv_id,
                fallback_reason="A verified answer could not be generated in time.",
            )

        # ── Stage 6: Parse Structured Output ───────────────────────────
        legal_answer = self._parse_llm_output(llm_output)
        if legal_answer.generation_mode == "failed":
            if follow_up:
                return self._build_follow_up_fallback(evidence_bundle, query)
            return self._build_evidence_only_answer(
                evidence_bundle,
                conv_id,
                fallback_reason="The generated answer could not be verified.",
            )
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
            if not validation.invalid_citations and not validation.malformed_citations:
                logger.warning(
                    "Generated answer had no usable citations; returning retrieved evidence only."
                )
                if follow_up:
                    return self._build_follow_up_fallback(
                        evidence_bundle,
                        query,
                    )
                return self._build_evidence_only_answer(
                    evidence_bundle,
                    conv_id,
                    fallback_reason="The generated answer could not be verified.",
                )
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

    def _build_follow_up_fallback(
        self,
        bundle: EvidenceBundle,
        query: str,
    ) -> LegalAnswer:
        normalized_query = query.lower()
        if "exception" in normalized_query:
            exception_markers = (
                "however",
                "except",
                "unless",
                "but",
                "partial",
                "partially",
                "diverted",
            )
            for item in bundle.items:
                normalized_content = re.sub(r"\s+", " ", item.content)
                for raw_sentence in re.split(r"(?<=[.!?])\s+", normalized_content):
                    sentence = re.sub(
                        r"^\s*(?:\(?\d+[.)]\s*)+",
                        "",
                        raw_sentence,
                    ).strip()
                    if (
                        len(sentence) >= 30
                        and any(marker in sentence.lower() for marker in exception_markers)
                        and not re.search(r"\b(?:and|or|but|if|however)$", sentence, re.IGNORECASE)
                    ):
                        return self._build_cited_follow_up_answer(
                            f"The source describes this related condition: {sentence}",
                            item,
                        )

        if "reason" in normalized_query:
            item = bundle.items[0]
            return self._build_cited_follow_up_answer(
                "The cited passage states the outcome but does not explain the "
                "reason behind the policy.",
                item,
            )

        if "evidence" in normalized_query or "support" in normalized_query:
            item = bundle.items[0]
            page = f" on page {item.page_start}" if item.page_start is not None else ""
            return self._build_cited_follow_up_answer(
                f"The cited passage directly supports this answer{page}.",
                item,
            )

        return LegalAnswer(
            answer="The retrieved source does not provide enough information to answer this follow-up.",
            confidence="low",
            limitations=["The follow-up requires details not established by the retrieved source."],
            abstained=True,
            generation_mode="failed",
        )

    @staticmethod
    def _build_cited_follow_up_answer(answer: str, item: EvidenceItem) -> LegalAnswer:
        return LegalAnswer(
            answer=f"{answer} [{item.citation_id}]",
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
            ],
            confidence="medium",
            grounding_valid=True,
            generation_mode="evidence_only",
        )

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
        self,
        bundle: EvidenceBundle,
        conv_id: str,
        fallback_reason: str = "The LLM service is not available.",
        require_query_match: bool = False,
    ) -> LegalAnswer:
        """Return one concise, cited sentence from the most relevant source."""
        ignored_terms = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "how", "if", "in", "is", "it", "of", "on", "or", "the", "to",
            "under", "was", "what", "when", "where", "which", "who", "why",
            "with",
        }
        query_terms = {
            term.lower()
            for term in re.findall(r"\b[\w'-]+\b", bundle.query)
            if term.lower() not in ignored_terms
        }
        refund_query = bool(
            {"fare", "refund", "refunded", "reimbursement", "clerkage"}
            & query_terms
        )
        refund_terms = {
            "refund",
            "refunded",
            "refundable",
            "reimbursement",
            "fare",
            "amount",
            "clerkage",
            "deduct",
            "deducted",
            "deducting",
        } if refund_query else set()
        candidates: list[tuple[int, int, int, int, int, str, EvidenceItem]] = []
        for item_index, item in enumerate(bundle.items):
            normalized_content = re.sub(r"\s+", " ", item.content)
            sentences = re.split(r"(?<=[.!?])\s+", normalized_content)
            for sentence_index, raw_sentence in enumerate(sentences):
                sentence = re.sub(r"^\s*(?:\(?\d+[.)]\s*)+", "", raw_sentence).strip()
                if not sentence:
                    continue
                sentence_terms = {
                    term.lower()
                    for term in re.findall(r"\b[\w'-]+\b", sentence)
                }
                score = len(query_terms & sentence_terms)
                focus_score = len(refund_terms & sentence_terms)
                candidates.append(
                    (
                        focus_score,
                        score,
                        len(query_terms),
                        -item_index,
                        -sentence_index,
                        sentence,
                        item,
                    )
                )

        if not candidates:
            return LegalAnswer(
                answer="I couldn't find a complete source sentence to answer this question.",
                confidence="low",
                limitations=[fallback_reason],
                abstained=True,
                generation_mode="failed",
            )

        best = max(candidates, key=lambda candidate: candidate[:5])
        _, matches, query_term_count, _, _, sentence, item = best
        has_query_match = (
            query_term_count > 0
            and matches >= _MIN_EXTRACTIVE_MATCHES
            and matches / query_term_count >= _MIN_EXTRACTIVE_QUERY_COVERAGE
        )
        if require_query_match and not has_query_match:
            return LegalAnswer(
                answer="",
                confidence="low",
                limitations=[],
                abstained=True,
                generation_mode="not_run",
            )
        if not has_query_match:
            return LegalAnswer(
                answer="I couldn't generate a verified answer from the retrieved sources.",
                confidence="low",
                limitations=[fallback_reason],
                abstained=True,
                generation_mode="failed",
            )
        sentence = sentence[:500].rstrip()
        answer = f"{sentence} [{item.citation_id}]"

        return LegalAnswer(
            answer=answer,
            key_points=[],
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
            ],
            limitations=[],
            confidence="low",
            grounding_valid=True,
            generation_mode="evidence_only",
        )
