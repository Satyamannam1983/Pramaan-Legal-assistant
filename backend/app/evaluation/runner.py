from __future__ import annotations

import logging
import math
import os
import re
import statistics
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.db.connection import check_connection, get_db, init_db
from app.db.models import CHUNKS_COLLECTION, DOCUMENTS_COLLECTION
from app.evaluation.dataset import dataset_sha256, load_retrieval_dataset
from app.evaluation.retrieval_evaluator import category_metrics, evaluate_rankings
from app.evaluation.report import write_report
from app.rag.generation.factory import get_llm_provider
from app.rag.citations.citation_service import CitationService
from app.rag.graph.graph import build_legal_research_graph
from app.rag.services.rag_service import LegalRAGService
from app.rag.services.retrieval_service import LegalRetrievalService, SearchResultItem

logger = logging.getLogger(__name__)
_SOURCE_ID = re.compile(r"^SOURCE [1-9]\d*$")


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return round(ordered[lower], 3)
    weight = index - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean_ms": round(statistics.fmean(values), 3) if values else None,
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
    }


def _status(reason: str) -> dict[str, Any]:
    return {"status": "NOT RUN", "reason": reason, "metrics": {}, "diagnostics": {}}


def _build_configuration(provider_name: str | None = None, model_name: str | None = None) -> dict[str, Any]:
    provider = os.getenv("LLM_PROVIDER", settings.llm_provider).strip().lower()
    model = os.getenv("LLM_MODEL", settings.llm_model).strip()
    if provider == "openrouter":
        api_key_configured = bool(
            os.getenv("OPENROUTER_API_KEY") or settings.openrouter_api_key
        )
    elif provider in {"openai", "openai-compatible", "openai_compatible"}:
        api_key_configured = bool(
            os.getenv("OPENAI_API_KEY") or settings.openai_api_key
        )
    else:
        api_key_configured = False
    return {
        "llm_provider": provider_name or provider,
        "llm_model": model_name or model,
        "answer_execution_path": "LangGraph question-answering workflow",
        "llm_api_key_configured": api_key_configured,
        "llm_temperature": settings.llm_temperature,
        "llm_max_tokens": settings.llm_max_tokens,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "reranker_provider": settings.reranker_provider or None,
        "reranker_model": settings.reranker_model or None,
        "vector_top_k": settings.vector_top_k,
        "keyword_top_k": settings.keyword_top_k,
        "rerank_top_k": settings.rerank_top_k,
        "final_top_k": settings.final_top_k,
        "retrieval_evaluation_top_k": max(settings.final_top_k, 10),
        "rrf_k": settings.rrf_k,
        "vector_weight": settings.vector_weight,
        "keyword_weight": settings.keyword_weight,
        "max_retrieval_attempts": settings.max_retrieval_attempts,
    }


async def _load_label_inventory(datasets: Iterable[Any]) -> tuple[set[str], dict[str, str], set[str]]:
    chunk_ids: set[str] = set()
    document_ids: set[str] = set()
    for dataset in datasets:
        for example in dataset.examples:
            chunk_ids.update(example.relevant_chunk_ids)
            document_ids.update(example.relevant_document_ids)
            if example.expected_answer:
                document_ids.update(example.expected_answer.expected_citation_document_ids)
    db = get_db()
    existing_chunks: dict[str, str] = {}
    if chunk_ids:
        rows = await db[CHUNKS_COLLECTION].find(
            {"chunk_id": {"$in": sorted(chunk_ids)}},
            {"_id": 0, "chunk_id": 1, "document_id": 1},
        ).to_list(length=len(chunk_ids))
        existing_chunks = {
            row["chunk_id"]: row["document_id"]
            for row in rows
            if row.get("chunk_id") and row.get("document_id")
        }
    if existing_chunks:
        document_ids.update(existing_chunks.values())
    existing_docs: set[str] = set()
    if document_ids:
        rows = await db[DOCUMENTS_COLLECTION].find(
            {"document_id": {"$in": sorted(document_ids)}},
            {"_id": 0, "document_id": 1},
        ).to_list(length=len(document_ids))
        existing_docs = {row["document_id"] for row in rows if row.get("document_id")}
    return existing_docs, existing_chunks, document_ids


def _validated_relevance(
    example: Any,
    existing_docs: set[str],
    existing_chunks: dict[str, str],
) -> tuple[set[str] | None, str | None]:
    if example.relevant_chunk_ids:
        missing = sorted(set(example.relevant_chunk_ids) - existing_chunks.keys())
        orphaned = sorted(
            chunk_id
            for chunk_id in example.relevant_chunk_ids
            if chunk_id in existing_chunks
            and existing_chunks[chunk_id] not in existing_docs
        )
        expected_docs_missing = sorted(set(example.relevant_document_ids) - existing_docs)
        mismatched = sorted(
            chunk_id
            for chunk_id in example.relevant_chunk_ids
            if chunk_id in existing_chunks
            and example.relevant_document_ids
            and existing_chunks[chunk_id] not in example.relevant_document_ids
        )
        if missing or orphaned or expected_docs_missing or mismatched:
            return None, "Relevance labels do not resolve to matching stored source documents and chunks."
        return set(example.relevant_chunk_ids), None
    if example.relevant_document_ids:
        if not set(example.relevant_document_ids).issubset(existing_docs):
            return None, "One or more relevance document IDs are absent from the indexed corpus."
        return set(example.relevant_document_ids), None
    return None, "No retrieval relevance labels were supplied."


def _citation_integrity(answer: Any, retrieved: list[Any]) -> dict[str, Any]:
    if isinstance(answer, dict):
        answer_payload = answer.get("answer", {})
        returned = answer.get("citations", [])
        answer_text = (
            answer_payload.get("answer", "")
            if isinstance(answer_payload, dict) else ""
        )
    else:
        returned = answer.citations
        answer_text = answer.answer
    available_sources: set[tuple[str, str]] = set()
    for row in retrieved:
        if isinstance(row, dict):
            document_id = row.get("document_id")
            chunk_id = row.get("chunk_id")
            content = row.get("content")
        else:
            document_id = row.document_id
            chunk_id = row.chunk_id
            content = row.content
        if (
            isinstance(document_id, str) and document_id.strip()
            and isinstance(chunk_id, str) and chunk_id.strip()
            and isinstance(content, str) and content.strip()
        ):
            available_sources.add((document_id, chunk_id))
    inline_numbers, malformed = CitationService.parse_references(answer_text)
    inline_ids = [f"SOURCE {source_id}" for source_id in inline_numbers]
    returned_ids = [
        citation.get("citation_id") if isinstance(citation, dict)
        else citation.citation_id
        for citation in returned
    ]
    seen: set[str] = set()
    valid_identifier_count = 0
    valid_provenance_count = 0
    valid_record_count = 0
    for citation, citation_id in zip(returned, returned_ids):
        identity = (
            (citation.get("document_id", ""), citation.get("chunk_id", ""))
            if isinstance(citation, dict)
            else (citation.document_id, citation.chunk_id)
        )
        unique = isinstance(citation_id, str) and citation_id not in seen
        valid_identifier = (
            isinstance(citation_id, str)
            and bool(_SOURCE_ID.fullmatch(citation_id))
            and citation_id in inline_ids
            and unique
        )
        valid_provenance = identity in available_sources
        valid_identifier_count += int(valid_identifier)
        valid_provenance_count += int(valid_provenance)
        valid_record_count += int(valid_identifier and valid_provenance)
        if isinstance(citation_id, str):
            seen.add(citation_id)
    unresolved_inline = sum(
        1 for citation_id in inline_ids
        if citation_id not in returned_ids
    )
    return {
        "returned_citation_count": len(returned),
        "citation_identifier_validity": (
            valid_identifier_count / len(returned) if returned else None
        ),
        "valid_identifier_count": valid_identifier_count,
        "inline_source_reference_count": len(inline_ids),
        "unresolved_inline_reference_count": unresolved_inline,
        "malformed_inline_reference_count": len(malformed),
        "provenance_validity": (
            valid_provenance_count / len(returned) if returned else None
        ),
        "valid_provenance_count": valid_provenance_count,
        "citation_record_integrity": (
            valid_record_count / len(returned) if returned else None
        ),
        "valid_record_count": valid_record_count,
        "semantic_claim_support_evaluated": False,
    }


async def run_evaluation(
    retrieval_dataset_path: str | Path,
    answer_dataset_path: str | Path,
    output_directory: str | Path,
    *,
    retrieval_only: bool = False,
    answers_only: bool = False,
    retrieval_service: Any | None = None,
    rag_service: Any | None = None,
    database_available: bool | None = None,
) -> dict[str, Any]:
    retrieval_dataset = load_retrieval_dataset(retrieval_dataset_path)
    answer_dataset = load_retrieval_dataset(answer_dataset_path)
    datasets = [answer_dataset] if answers_only else [retrieval_dataset]
    if not retrieval_only and not answers_only:
        datasets.append(answer_dataset)

    provider_name: str | None = None
    model_name: str | None = None
    provider = None
    if not retrieval_only:
        try:
            provider = get_llm_provider()
            provider_name = getattr(provider, "provider_name", None)
            model_name = getattr(provider, "model_name", None)
        except Exception as exc:
            logger.warning("Could not initialize configured LLM provider (%s).", type(exc).__name__)

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "NOT RUN",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_size": (
            len(answer_dataset.examples) if answers_only else
            len(retrieval_dataset.examples) if retrieval_only else
            len(retrieval_dataset.examples) + len(answer_dataset.examples)
        ),
        "datasets": {
            "retrieval": {
                "name": retrieval_dataset.name,
                "version": retrieval_dataset.version,
                "provenance": retrieval_dataset.provenance,
                "example_count": len(retrieval_dataset.examples),
                "sha256": dataset_sha256(retrieval_dataset_path),
            },
            "answers": {
                "name": answer_dataset.name,
                "version": answer_dataset.version,
                "provenance": answer_dataset.provenance,
                "example_count": len(answer_dataset.examples),
                "sha256": dataset_sha256(answer_dataset_path),
            },
        },
        "configuration": _build_configuration(provider_name, model_name),
        "evaluations": {},
        "limitations": [
            "No semantic claim-evidence entailment metric is implemented.",
            "No human answer-quality evaluator or judged answer reference set is configured.",
            "Document analysis and document comparison evaluations are not executed by this workflow.",
            "Retrieval results depend on the indexed MongoDB corpus and its current contents.",
            "Provider output, approximate vector search, model hardware, and corpus changes can introduce nondeterminism.",
        ],
    }

    if retrieval_only:
        report["evaluations"]["answer_generation"] = _status("Skipped by --retrieval-only.")
    if answers_only:
        report["evaluations"]["retrieval"] = _status("Skipped by --answers-only.")
    if not retrieval_dataset.examples and not answers_only:
        report["evaluations"]["retrieval"] = _status("Retrieval dataset is empty.")
    if not answer_dataset.examples and not retrieval_only:
        report["evaluations"]["answer_generation"] = _status("Answer dataset is empty.")
    if not answers_only:
        report["evaluations"]["document_analysis"] = _status(
            "Not measured: no answer-reference evaluation is wired for document analysis."
        )
        report["evaluations"]["document_comparison"] = _status(
            "Not measured: no answer-reference evaluation is wired for document comparison."
        )

    connected = await check_connection() if database_available is None else database_available
    if connected:
        try:
            await init_db()
        except Exception as exc:
            logger.warning("Evaluation database initialization failed (%s).", type(exc).__name__)
            connected = False

    cache: dict[str, list[SearchResultItem] | None] = {}
    retrieval_latencies: list[float] = []
    retrieval_failures = 0
    empty_retrievals = 0
    retrieval_service = retrieval_service or LegalRetrievalService()

    async def retrieve(query: str) -> list[SearchResultItem] | None:
        nonlocal retrieval_failures, empty_retrievals
        key = query.strip()
        if key in cache:
            return cache[key]
        start = time.monotonic()
        try:
            results = await retrieval_service.search(
                key,
                top_k=max(settings.final_top_k, 10),
            )
            retrieval_latencies.append((time.monotonic() - start) * 1000)
            if not results:
                empty_retrievals += 1
            cache[key] = results
        except Exception as exc:
            retrieval_latencies.append((time.monotonic() - start) * 1000)
            retrieval_failures += 1
            logger.warning("Evaluation retrieval failed (%s).", type(exc).__name__)
            cache[key] = None
        return cache[key]

    if not connected:
        if "retrieval" not in report["evaluations"]:
            report["evaluations"]["retrieval"] = _status("MongoDB is unavailable.")
        if "answer_generation" not in report["evaluations"]:
            report["evaluations"]["answer_generation"] = _status("MongoDB is unavailable.")
    else:
        existing_docs, existing_chunks, _ = await _load_label_inventory(datasets)

        if not answers_only and retrieval_dataset.examples:
            retrieval_rows: list[tuple[list[str], set[str]]] = []
            categorized_rows: list[tuple[str, list[str], set[str]]] = []
            skipped_labels: list[dict[str, str]] = []
            for example in retrieval_dataset.examples:
                relevant, reason = _validated_relevance(
                    example, existing_docs, existing_chunks
                )
                if relevant is None:
                    skipped_labels.append({"id": example.id, "reason": reason or "No valid labels."})
                    continue
                results = await retrieve(example.query)
                if results is None:
                    continue
                if example.relevant_chunk_ids:
                    ranked_ids = [result.chunk_id for result in results]
                else:
                    ranked_ids = [result.document_id for result in results]
                retrieval_rows.append((ranked_ids, relevant))
                categorized_rows.append((example.category, ranked_ids, relevant))
            metrics = (
                evaluate_rankings(retrieval_rows).model_dump()
                if retrieval_rows
                else {}
            )
            status = "COMPLETED"
            if skipped_labels or retrieval_failures:
                status = "PARTIAL" if metrics else "BLOCKED"
            if not metrics:
                status = "BLOCKED"
            report["evaluations"]["retrieval"] = {
                "status": status,
                "reason": None if metrics else "No valid, indexed relevance labels were available.",
                "label_granularity": (
                    "chunk" if any(e.relevant_chunk_ids for e in retrieval_dataset.examples)
                    else "document"
                ),
                "valid_labeled_example_count": len(retrieval_rows),
                "skipped_labels": skipped_labels,
                "metrics": metrics,
                "category_metrics": {
                    category: values.model_dump()
                    for category, values in category_metrics(categorized_rows).items()
                } if categorized_rows else {},
                "diagnostics": {
                    "unique_retrieval_requests": len(cache),
                    "failed_requests": retrieval_failures,
                    "empty_results": empty_retrievals,
                    "latency_ms": _latency_summary(retrieval_latencies),
                },
            }

        if not retrieval_only and answer_dataset.examples:
            provider_label = (
                str(provider_name or os.getenv("LLM_PROVIDER", settings.llm_provider)).lower()
            )
            rag_service = rag_service or LegalRAGService()
            graph = build_legal_research_graph(retrieval_service, rag_service)
            answer_latencies: list[float] = []
            generation_failures = 0
            answer_retrieval_failures = 0
            fallback_responses = 0
            answers_observed = 0
            citation_records = 0
            valid_citation_records = 0
            valid_citation_provenance_records = 0
            valid_citation_record_integrity = 0
            inline_references = 0
            unresolved_inline = 0
            malformed_references = 0
            abstention_cases = 0
            correct_abstentions = 0
            skipped_abstention_cases = 0
            required_terms = 0
            matched_terms = 0
            expected_citation_documents = 0
            matched_citation_documents = 0
            unresolved_expected_source_cases = 0
            per_case: list[dict[str, Any]] = []

            for example in answer_dataset.examples:
                start = time.monotonic()
                try:
                    state = await graph.ainvoke(
                        {"query": example.query, "original_query": example.query}
                    )
                except Exception as exc:
                    generation_failures += 1
                    logger.warning("Evaluation answer generation failed (%s).", type(exc).__name__)
                    per_case.append({"id": example.id, "status": "FAILED"})
                    continue
                answer_latencies.append((time.monotonic() - start) * 1000)
                answer = state.get("answer", {})
                if not isinstance(answer, dict):
                    generation_failures += 1
                    per_case.append({"id": example.id, "status": "INVALID_RESPONSE"})
                    continue
                results = state.get("retrieved_results", [])
                if not isinstance(results, list):
                    results = []
                if state.get("retrieval_error"):
                    answer_retrieval_failures += 1
                answers_observed += 1
                mode = str(answer.get("generation_mode", state.get("generation_mode", ""))).lower()
                if mode in {"evidence_only", "failed"}:
                    fallback_responses += 1
                if mode == "failed" or state.get("generation_error"):
                    generation_failures += 1
                integrity = _citation_integrity(state, results)
                citation_records += integrity["returned_citation_count"]
                valid_citation_records += (
                    integrity["valid_identifier_count"]
                    if integrity["citation_identifier_validity"] is not None else 0
                )
                valid_citation_provenance_records += (
                    integrity["valid_provenance_count"]
                    if integrity["provenance_validity"] is not None else 0
                )
                valid_citation_record_integrity += (
                    integrity["valid_record_count"]
                    if integrity["citation_record_integrity"] is not None else 0
                )
                inline_references += integrity["inline_source_reference_count"]
                unresolved_inline += integrity["unresolved_inline_reference_count"]
                malformed_references += integrity["malformed_inline_reference_count"]
                expected = example.expected_answer
                expected_sources = (
                    set(expected.expected_citation_document_ids)
                    if expected else set()
                )
                expected_sources_available = expected_sources.issubset(existing_docs)
                if expected_sources and not expected_sources_available:
                    unresolved_expected_source_cases += 1
                if expected and expected.expected_abstention is not None:
                    if expected_sources and not expected_sources_available:
                        skipped_abstention_cases += 1
                    else:
                        abstention_cases += 1
                        correct_abstentions += int(
                            bool(answer.get("abstained", False))
                            == expected.expected_abstention
                        )
                if expected:
                    for term in expected.required_terms:
                        required_terms += 1
                        matched_terms += int(
                            term.casefold() in str(answer.get("answer", "")).casefold()
                        )
                    cited_documents = {
                        citation.get("document_id")
                        for citation in state.get("citations", [])
                        if isinstance(citation, dict)
                        and isinstance(citation.get("document_id"), str)
                    }
                    if expected_sources and expected_sources_available:
                        expected_citation_documents += len(expected_sources)
                        matched_citation_documents += len(expected_sources & cited_documents)
                per_case.append({
                    "id": example.id,
                    "status": "COMPLETED",
                    "generation_mode": mode,
                    "retrieval_result_count": len(results),
                    "retrieval_attempts": state.get("retrieval_attempt", 0),
                    "retrieval_failed": bool(state.get("retrieval_error")),
                    "citation_record_count": integrity["returned_citation_count"],
                    "citation_identifier_validity": integrity["citation_identifier_validity"],
                    "citation_provenance_validity": integrity["provenance_validity"],
                    "citation_record_integrity": integrity["citation_record_integrity"],
                    "inline_reference_resolution": (
                        1.0 - integrity["unresolved_inline_reference_count"]
                        / integrity["inline_source_reference_count"]
                        if integrity["inline_source_reference_count"] else None
                    ),
                    "abstained": bool(answer.get("abstained", False)),
                    "expected_source_status": (
                        "UNRESOLVED"
                        if expected_sources and not expected_sources_available
                        else "RESOLVED" if expected_sources else None
                    ),
                    "abstention_label_scored": bool(
                        expected
                        and expected.expected_abstention is not None
                        and (not expected_sources or expected_sources_available)
                    ),
                })

            citation_validity = (
                valid_citation_records / citation_records if citation_records else None
            )
            inline_resolution = (
                1.0 - unresolved_inline / inline_references if inline_references else None
            )
            abstention_accuracy = (
                correct_abstentions / abstention_cases if abstention_cases else None
            )
            characteristic_coverage = (
                matched_terms / required_terms if required_terms else None
            )
            expected_citation_coverage = (
                matched_citation_documents / expected_citation_documents
                if expected_citation_documents else None
            )
            answer_quality_status = "BLOCKED"
            answer_quality_reason = (
                "No semantic evaluator or human-judged answer references are configured."
            )
            if provider_label in {"stub", "mock"} or provider_name == "stub":
                answer_quality_reason = (
                    "Configured LLM is a deterministic stub; answer-quality evaluation is blocked."
                )
            report["evaluations"]["answer_generation"] = {
                "status": "PARTIAL" if answers_observed else "BLOCKED",
                "reason": (
                    "Operational observations completed; semantic answer-quality evaluation remains blocked."
                    if answers_observed else "No answer-generation observations completed."
                ),
                "provider_mode": provider_name or provider_label,
                "quality_evaluation": {
                    "status": answer_quality_status,
                    "reason": answer_quality_reason,
                    "evaluator": None,
                    "metrics": {},
                },
                "operational_metrics": {
                    "citation_identifier_validity": citation_validity,
                    "citation_valid_identifier_records": valid_citation_records,
                    "citation_provenance_validity": (
                        valid_citation_provenance_records / citation_records
                        if citation_records else None
                    ),
                    "citation_valid_provenance_records": valid_citation_provenance_records,
                    "citation_record_integrity": (
                        valid_citation_record_integrity / citation_records
                        if citation_records else None
                    ),
                    "citation_integrity_valid_records": valid_citation_record_integrity,
                    "citation_records": citation_records,
                    "valid_citation_records": valid_citation_records,
                    "inline_reference_resolution": inline_resolution,
                    "inline_references": inline_references,
                    "unresolved_inline_references": unresolved_inline,
                    "malformed_inline_references": malformed_references,
                    "citation_semantic_support_evaluated": False,
                    "abstention_accuracy": abstention_accuracy,
                    "abstention_cases": abstention_cases,
                    "correct_abstentions": correct_abstentions,
                    "skipped_abstention_cases": skipped_abstention_cases,
                    "required_term_coverage": characteristic_coverage,
                    "required_terms_checked": required_terms,
                    "required_terms_matched": matched_terms,
                    "expected_citation_document_coverage": expected_citation_coverage,
                    "expected_citation_documents": expected_citation_documents,
                    "matched_expected_citation_documents": matched_citation_documents,
                    "unresolved_expected_source_cases": unresolved_expected_source_cases,
                },
                "diagnostics": {
                    "answers_observed": answers_observed,
                    "failed_requests": generation_failures,
                    "retrieval_failures": answer_retrieval_failures,
                    "fallback_responses": fallback_responses,
                    "empty_retrievals": sum(
                        1 for item in per_case
                        if item.get("retrieval_result_count") == 0
                    ),
                    "latency_ms": _latency_summary(answer_latencies),
                },
                "cases": per_case,
                "limitations": [
                    "Citation identifier validity verifies source-ID syntax and retrieved document/chunk membership only.",
                    "Required-term checks are literal substring checks and do not establish answer correctness.",
                    "No semantic claim-evidence entailment evaluator is available.",
                ],
            }

    stages = list(report["evaluations"].values())
    runnable_statuses = [stage.get("status") for stage in stages]
    if not stages or all(status == "NOT RUN" for status in runnable_statuses):
        report["status"] = "NOT RUN"
    elif all(status in {"COMPLETED", "SKIPPED"} for status in runnable_statuses):
        report["status"] = "COMPLETED"
    elif any(status in {"COMPLETED", "PARTIAL"} for status in runnable_statuses):
        report["status"] = "PARTIAL"
    else:
        report["status"] = "BLOCKED"

    write_report(report, output_directory)
    return report


async def run_default_evaluation(
    retrieval_dataset_path: str | Path,
    answer_dataset_path: str | Path,
    output_directory: str | Path,
    *,
    retrieval_only: bool = False,
    answers_only: bool = False,
) -> dict[str, Any]:
    return await run_evaluation(
        retrieval_dataset_path,
        answer_dataset_path,
        output_directory,
        retrieval_only=retrieval_only,
        answers_only=answers_only,
    )
