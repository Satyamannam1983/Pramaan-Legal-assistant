"""Tests for the LLM providers and RAG service."""
from __future__ import annotations

import asyncio
import json
import pytest

from app.rag.generation.mock_provider import MockLLM
from app.rag.generation.factory import get_llm_provider, reset_llm_provider
from app.rag.services.retrieval_service import SearchResultItem


class TestMockLLM:
    @pytest.mark.asyncio
    async def test_returns_valid_json(self):
        llm = MockLLM()
        output = await llm.generate("system", "user")
        data = json.loads(output)
        assert "answer" in data
        assert "key_points" in data
        assert "citations" in data
        assert "limitations" in data
        assert "confidence" in data

    @pytest.mark.asyncio
    async def test_model_name(self):
        llm = MockLLM(model="test-model")
        assert llm.model_name == "test-model"

    @pytest.mark.asyncio
    async def test_citations_count_matches_sources(self):
        llm = MockLLM()
        user_prompt = "Question\n[SOURCE 1]\n[SOURCE 2]\n[SOURCE 3]\n[/SOURCES]"
        output = await llm.generate("system", user_prompt)
        data = json.loads(output)
        assert len(data["citations"]) == 3
        assert "[SOURCE 1]" in data["answer"]


class TestLLMFactory:
    def test_mock_provider(self):
        import os
        os.environ["LLM_PROVIDER"] = "mock"
        os.environ["LLM_MODEL"] = "test"
        reset_llm_provider()
        llm = get_llm_provider()
        assert isinstance(llm, MockLLM)
        reset_llm_provider()

    def test_stub_provider(self):
        import os
        os.environ["LLM_PROVIDER"] = "stub"
        reset_llm_provider()
        llm = get_llm_provider()
        assert isinstance(llm, MockLLM)
        reset_llm_provider()

    def test_unknown_provider_raises(self):
        import os
        original = os.environ.get("LLM_PROVIDER", "mock")
        os.environ["LLM_PROVIDER"] = "unknown_provider_xyz"
        reset_llm_provider()
        try:
            with pytest.raises(ValueError, match="Unknown LLM provider"):
                get_llm_provider()
        finally:
            os.environ["LLM_PROVIDER"] = original
            reset_llm_provider()

    def test_settings_provider_is_used_when_environment_override_is_absent(self, monkeypatch):
        from app.config import settings

        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.setattr(settings, "llm_provider", "mock")
        monkeypatch.setattr(settings, "llm_model", "settings-model")
        reset_llm_provider()
        try:
            provider = get_llm_provider()
            assert provider.model_name == "settings-model"
        finally:
            reset_llm_provider()

    def test_openrouter_factory_uses_router_endpoint_and_separate_key(self, monkeypatch):
        from app.config import settings
        from app.rag.generation.openai_provider import OpenAICompatibleLLM

        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        monkeypatch.setenv("LLM_MODEL", "openrouter/free")
        monkeypatch.setattr(settings, "openrouter_api_key", "test-openrouter-key")
        reset_llm_provider()
        try:
            llm = get_llm_provider()
            assert isinstance(llm, OpenAICompatibleLLM)
            assert llm.provider_name == "openrouter"
            assert llm.model_name == "openrouter/free"
            assert llm._api_key == "test-openrouter-key"
            assert llm._base_url == "https://openrouter.ai/api/v1"
        finally:
            reset_llm_provider()

    def test_openrouter_does_not_fall_back_to_configured_openai_key(self, monkeypatch):
        from app.config import settings

        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        monkeypatch.setattr(settings, "openrouter_api_key", None)
        monkeypatch.setattr(settings, "openai_api_key", "openai-key-must-not-be-used")
        reset_llm_provider()
        try:
            llm = get_llm_provider()
            with pytest.raises(RuntimeError, match="No API key configured"):
                llm._get_client()
        finally:
            reset_llm_provider()

    @pytest.mark.asyncio
    async def test_openrouter_request_failure_is_not_replaced_with_stub(self, monkeypatch):
        from app.rag.generation.openai_provider import OpenAICompatibleLLM

        class FailedCompletions:
            async def create(self, **kwargs):
                raise RuntimeError("router unavailable")

        class FailedClient:
            chat = type("Chat", (), {"completions": FailedCompletions()})()

        llm = OpenAICompatibleLLM(
            model="openrouter/free",
            api_key="test-openrouter-key",
            base_url="https://openrouter.ai/api/v1",
            provider_name="openrouter",
        )
        monkeypatch.setattr(llm, "_get_client", lambda: FailedClient())
        with pytest.raises(RuntimeError, match="LLM generation failed"):
            await llm.generate("system", "user")


class TestRAGService:
    @pytest.mark.asyncio
    async def test_empty_query_returns_message(self):
        from app.rag.services.rag_service import LegalRAGService
        service = LegalRAGService()
        result = await service.answer("")
        assert "valid legal question" in result.answer.lower()
        assert result.confidence == "low"

    @pytest.mark.asyncio
    async def test_matching_source_sentence_returns_without_calling_llm(self, monkeypatch):
        from app.rag.services import rag_service as rag_module

        class SlowLLM:
            async def generate(self, **kwargs):
                raise AssertionError("LLM should not be called for a direct source match.")

        monkeypatch.setattr(rag_module, "get_llm_provider", lambda: SlowLLM())
        result = await rag_module.LegalRAGService().answer(
            "If a train is cancelled for its entire run, how is the ticket refund handled?",
            search_results=[
                SearchResultItem(
                    chunk_id="chunk-1",
                    content=(
                        "A ticket may be booked online. "
                        "In case of train cancellation on its entire run, "
                        "full refund is granted automatically by the system."
                    ),
                    document_id="doc-1",
                    rrf_score=0.05,
                )
            ],
        )

        assert result.answer == (
            "In case of train cancellation on its entire run, full refund is granted "
            "automatically by the system. [SOURCE 1]"
        )
        assert result.generation_mode == "evidence_only"
        assert result.citations[0].chunk_id == "chunk-1"

    @pytest.mark.asyncio
    async def test_fare_question_prefers_refund_sentence_and_keeps_wrapped_text(self, monkeypatch):
        from app.rag.services import rag_service as rag_module

        class NoCallLLM:
            async def generate(self, **kwargs):
                raise AssertionError("A direct source match should not call the LLM.")

        monkeypatch.setattr(rag_module, "get_llm_provider", lambda: NoCallLLM())
        result = await rag_module.LegalRAGService().answer(
            "What happens to the fare when an e-ticket is fully waitlisted?",
            search_results=[
                SearchResultItem(
                    chunk_id="chunk-1",
                    content=(
                        "PNRs having fully waitlisted status will be dropped and automatic "
                        "refund of the ticket amount after deducting the applicable CLERKAGE "
                        "by Railway\nshall be credited to the account used for payment. "
                        "Passengers having fully waitlisted e-ticket are not allowed to "
                        "board the train."
                    ),
                    document_id="doc-1",
                    rrf_score=0.05,
                )
            ],
        )

        assert "automatic refund" in result.answer.lower()
        assert "CLERKAGE" in result.answer
        assert "shall be credited" in result.answer
        assert "not allowed to board" not in result.answer.lower()
        assert result.answer.endswith("[SOURCE 1]")

    @pytest.mark.asyncio
    async def test_follow_up_uses_focused_generation_instead_of_extractive_answer(self, monkeypatch):
        from app.rag.services import rag_service as rag_module

        captured_prompt = ""

        class FollowUpLLM:
            async def generate(self, **kwargs):
                nonlocal captured_prompt
                captured_prompt = kwargs["user_prompt"]
                return json.dumps({
                    "answer": "The source supports this by expressly stating that the refund is automatic [SOURCE 1].",
                    "key_points": [],
                    "limitations": [],
                    "confidence": "medium",
                })

        monkeypatch.setattr(rag_module, "get_llm_provider", lambda: FollowUpLLM())
        result = await rag_module.LegalRAGService().answer(
            'What evidence supports this answer: "In case of train cancellation on its entire run, full refund is granted automatically by the system."?',
            search_results=[
                SearchResultItem(
                    chunk_id="chunk-1",
                    content=(
                        "In case of train cancellation on its entire run, "
                        "full refund is granted automatically by the system."
                    ),
                    document_id="doc-1",
                    rrf_score=0.05,
                )
            ],
            follow_up=True,
        )

        assert "do not repeat the prior answer" in captured_prompt
        assert result.generation_mode != "evidence_only"
        assert result.answer.startswith("The source supports this")
        assert result.abstained is False

    @pytest.mark.asyncio
    async def test_follow_up_does_not_repeat_extract_when_generation_fails(self, monkeypatch):
        from app.rag.services import rag_service as rag_module

        class FailedLLM:
            async def generate(self, **kwargs):
                raise RuntimeError("provider unavailable")

        monkeypatch.setattr(rag_module, "get_llm_provider", lambda: FailedLLM())
        result = await rag_module.LegalRAGService().answer(
            "Does the source mention an exception to the refund?",
            search_results=[
                SearchResultItem(
                    chunk_id="chunk-1",
                    content="The full refund is granted automatically by the system.",
                    document_id="doc-1",
                    rrf_score=0.05,
                )
            ],
            follow_up=True,
        )

        assert result.abstained is True
        assert "does not provide enough information" in result.answer.lower()
        assert "full refund is granted automatically" not in result.answer.lower()

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            (
                'What evidence supports this answer: "The refund is automatic."?',
                "directly supports this answer",
            ),
            (
                'What reasoning supports this answer: "The refund is automatic."?',
                "does not explain the reason",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_evidence_and_reasoning_follow_ups_do_not_repeat_the_answer(
        self,
        monkeypatch,
        query,
        expected,
    ):
        from app.rag.services import rag_service as rag_module

        class FailedLLM:
            async def generate(self, **kwargs):
                raise RuntimeError("provider unavailable")

        monkeypatch.setattr(rag_module, "get_llm_provider", lambda: FailedLLM())
        result = await rag_module.LegalRAGService().answer(
            query,
            search_results=[
                SearchResultItem(
                    chunk_id="chunk-1",
                    content="The refund is automatic.",
                    document_id="doc-1",
                    page_start=2,
                    rrf_score=0.05,
                )
            ],
            follow_up=True,
        )

        assert result.abstained is False
        assert expected in result.answer.lower()
        assert "the refund is automatic" not in result.answer.lower()
        assert result.answer.endswith("[SOURCE 1]")

    @pytest.mark.asyncio
    async def test_exception_follow_up_returns_a_distinct_cited_condition_on_generation_failure(
        self,
        monkeypatch,
    ):
        from app.rag.services import rag_service as rag_module

        class FailedLLM:
            async def generate(self, **kwargs):
                raise RuntimeError("provider unavailable")

        monkeypatch.setattr(rag_module, "get_llm_provider", lambda: FailedLLM())
        result = await rag_module.LegalRAGService().answer(
            "Does the source describe any exception to this answer?",
            search_results=[
                SearchResultItem(
                    chunk_id="chunk-1",
                    content=(
                        "A full refund is granted automatically. "
                        "However, if the train is cancelled partially on its run or\n"
                        "diverted and not touching the boarding or destination station, "
                        "passengers must file a TDR within 72 hours."
                    ),
                    document_id="doc-1",
                    rrf_score=0.05,
                )
            ],
            follow_up=True,
        )

        assert result.abstained is False
        assert "cancelled partially" in result.answer
        assert "TDR within 72 hours" in result.answer
        assert not result.answer.endswith("from [SOURCE 1]")
        assert "full refund is granted automatically" not in result.answer.lower()
        assert result.answer.endswith("[SOURCE 1]")
        assert result.citations[0].chunk_id == "chunk-1"

    @pytest.mark.asyncio
    async def test_rag_rejects_answer_with_invalid_source_reference(self, monkeypatch):
        from app.rag.services import rag_service as rag_module

        class InvalidCitationLLM:
            async def generate(self, **kwargs):
                return json.dumps({
                    "answer": "Unsupported claim [SOURCE 99].",
                    "key_points": [],
                    "limitations": [],
                    "confidence": "high",
                })

        monkeypatch.setattr(rag_module, "get_llm_provider", lambda: InvalidCitationLLM())
        result = await rag_module.LegalRAGService().answer(
            "Question",
            search_results=[
                SearchResultItem(
                    chunk_id="chunk-1",
                    content="Retrieved source text.",
                    document_id="doc-1",
                    rrf_score=0.05,
                )
            ],
        )

        assert result.grounding_valid is False
        assert result.citations == []
        assert "couldn't verify" in result.answer.lower()

    @pytest.mark.asyncio
    async def test_rag_rejects_citation_metadata_not_used_in_answer(self, monkeypatch):
        from app.rag.services import rag_service as rag_module

        class MetadataOnlyCitationLLM:
            async def generate(self, **kwargs):
                return json.dumps({
                    "answer": "A legal claim without an inline citation.",
                    "key_points": [],
                    "citations": [{"citation_id": "SOURCE 1"}],
                    "limitations": [],
                    "confidence": "high",
                })

        monkeypatch.setattr(
            rag_module, "get_llm_provider", lambda: MetadataOnlyCitationLLM()
        )
        result = await rag_module.LegalRAGService().answer(
            "Question",
            search_results=[
                SearchResultItem(
                    chunk_id="chunk-1",
                    content="Retrieved source text.",
                    document_id="doc-1",
                    rrf_score=0.05,
                )
            ],
        )

        assert result.grounding_valid is False
        assert result.citations == []
        assert "couldn't generate a verified answer" in result.answer.lower()

    @pytest.mark.asyncio
    async def test_rag_malformed_llm_output_falls_back_to_cited_evidence(self, monkeypatch):
        from app.rag.services import rag_service as rag_module

        class MalformedLLM:
            async def generate(self, **kwargs):
                return "Private malformed provider payload [SOURCE 1]"

        monkeypatch.setattr(rag_module, "get_llm_provider", lambda: MalformedLLM())
        result = await rag_module.LegalRAGService().answer(
            "How is the refund handled if train cancellation occurs for its entire run?",
            search_results=[
                SearchResultItem(
                    chunk_id="chunk-1",
                    content=(
                        "Unrelated booking text. "
                        "In case of train cancellation on its entire run, "
                        "full refund is granted automatically by the system. "
                        "Contact support for unrelated information."
                    ),
                    document_id="doc-1",
                    rrf_score=0.05,
                )
            ],
        )

        assert result.grounding_valid is True
        assert result.generation_mode == "evidence_only"
        assert result.citations[0].chunk_id == "chunk-1"
        assert "Private malformed" not in result.answer
        assert result.answer == (
            "In case of train cancellation on its entire run, full refund is granted "
            "automatically by the system. [SOURCE 1]"
        )

    @pytest.mark.asyncio
    async def test_generation_failure_returns_explicit_evidence_only_fallback(self, monkeypatch):
        from app.rag.services import rag_service as rag_module

        def unavailable_provider():
            raise RuntimeError("private provider detail")

        monkeypatch.setattr(rag_module, "get_llm_provider", unavailable_provider)
        result = await rag_module.LegalRAGService().answer(
            "How is the refund handled if train cancellation occurs for its entire run?",
            search_results=[
                SearchResultItem(
                    chunk_id="chunk-1",
                    content=(
                        "In case of train cancellation on its entire run, "
                        "full refund is granted automatically by the system."
                    ),
                    document_id="doc-1",
                    rrf_score=0.05,
                )
            ],
        )

        assert result.generation_mode == "evidence_only"
        assert result.grounding_valid is True
        assert result.citations[0].chunk_id == "chunk-1"
        assert result.answer == (
            "In case of train cancellation on its entire run, full refund is granted "
            "automatically by the system. [SOURCE 1]"
        )
        assert result.key_points == []
        assert result.limitations == []

    @pytest.mark.asyncio
    async def test_openrouter_failure_returns_cited_evidence_without_generated_answer(self, monkeypatch):
        from app.rag.services import rag_service as rag_module

        class UnavailableOpenRouter:
            provider_name = "openrouter"

            async def generate(self, **kwargs):
                raise RuntimeError("private router error")

        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        monkeypatch.setattr(
            rag_module, "get_llm_provider", lambda: UnavailableOpenRouter()
        )
        result = await rag_module.LegalRAGService().answer(
            "How is the refund handled if train cancellation occurs for its entire run?",
            search_results=[
                SearchResultItem(
                    chunk_id="chunk-1",
                    content=(
                        "In case of train cancellation on its entire run, "
                        "full refund is granted automatically by the system."
                    ),
                    document_id="doc-1",
                    rrf_score=0.05,
                )
            ],
        )

        assert result.generation_mode == "evidence_only"
        assert result.grounding_valid is True
        assert result.citations[0].chunk_id == "chunk-1"
        assert result.answer == (
            "In case of train cancellation on its entire run, full refund is granted "
            "automatically by the system. [SOURCE 1]"
        )
        assert result.key_points == []
        assert result.limitations == []
        assert "private router error" not in result.answer

    @pytest.mark.asyncio
    async def test_slow_llm_falls_back_to_relevant_sentence(self, monkeypatch):
        from app.rag.services import rag_service as rag_module

        class SlowLLM:
            async def generate(self, **kwargs):
                await asyncio.sleep(1)

        monkeypatch.setattr(rag_module, "_LLM_TIMEOUT_SECONDS", 0.01)
        monkeypatch.setattr(rag_module, "get_llm_provider", lambda: SlowLLM())
        result = await rag_module.LegalRAGService().answer(
            "What is the contract termination notice requirement?",
            search_results=[
                SearchResultItem(
                    chunk_id="chunk-1",
                    content=(
                        "A contract may be renewed by mutual consent. "
                        "Termination requires written notice to the other party."
                    ),
                    document_id="doc-1",
                    rrf_score=0.05,
                )
            ],
        )

        assert result.answer == "Termination requires written notice to the other party. [SOURCE 1]"
        assert result.generation_mode == "evidence_only"
        assert result.citations[0].chunk_id == "chunk-1"

    @pytest.mark.asyncio
    async def test_stub_generation_is_labeled_and_not_presented_as_real(self, monkeypatch):
        from app.rag.services import rag_service as rag_module

        monkeypatch.setattr(rag_module, "get_llm_provider", lambda: MockLLM())
        result = await rag_module.LegalRAGService().answer(
            "Question",
            search_results=[
                SearchResultItem(
                    chunk_id="chunk-1",
                    content="Retrieved source text.",
                    document_id="doc-1",
                    rrf_score=0.05,
                )
            ],
        )

        assert result.generation_mode == "stub"
        assert "mock response" in result.answer.lower()
        assert result.grounding_valid is True
        assert result.citations[0].chunk_id == "chunk-1"

    @pytest.mark.asyncio
    async def test_rag_maps_valid_source_reference_to_retrieved_metadata(self, monkeypatch):
        from app.rag.services import rag_service as rag_module

        class CitedLLM:
            async def generate(self, **kwargs):
                return json.dumps({
                    "answer": "Supported claim [SOURCE 1].",
                    "key_points": [],
                    "limitations": [],
                    "confidence": "medium",
                })

        monkeypatch.setattr(rag_module, "get_llm_provider", lambda: CitedLLM())
        result = await rag_module.LegalRAGService().answer(
            "Question",
            search_results=[
                SearchResultItem(
                    chunk_id="chunk-1",
                    content="Retrieved source text.",
                    document_id="doc-1",
                    title="Evidence title",
                    section="Part II",
                    subsection="Section 4(a)",
                    page_start=12,
                    page_end=13,
                    document_type="judgment",
                    court="Supreme Court",
                    date="2024-01-02",
                    source="court repository",
                    source_url="https://example.test/source",
                    rrf_score=0.05,
                )
            ],
        )
        assert result.grounding_valid is True
        assert len(result.citations) == 1
        citation = result.citations[0]
        assert citation.chunk_id == "chunk-1"
        assert citation.document_id == "doc-1"
        assert citation.section == "Part II"
        assert citation.subsection == "Section 4(a)"
        assert citation.page_start == 12
        assert citation.page_end == 13
        assert citation.document_type == "judgment"
        assert citation.court == "Supreme Court"
        assert citation.date == "2024-01-02"
        assert citation.source == "court repository"
        assert citation.source_url == "https://example.test/source"

    @pytest.mark.asyncio
    async def test_shared_rag_service_keeps_citations_request_local(self, monkeypatch):
        from app.rag.services import rag_service as rag_module

        class CitedLLM:
            async def generate(self, **kwargs):
                await asyncio.sleep(0)
                return json.dumps({
                    "answer": "Supported claim [SOURCE 1].",
                    "key_points": [],
                    "limitations": [],
                    "confidence": "medium",
                })

        monkeypatch.setattr(rag_module, "get_llm_provider", lambda: CitedLLM())
        service = rag_module.LegalRAGService()

        async def answer_for(document_id: str):
            return await service.answer(
                "Question",
                search_results=[
                    SearchResultItem(
                        chunk_id=f"{document_id}:chunk",
                        content="Retrieved source text.",
                        document_id=document_id,
                        title=document_id,
                        rrf_score=0.05,
                    )
                ],
            )

        first, second = await asyncio.gather(answer_for("doc-1"), answer_for("doc-2"))
        assert first.citations[0].document_id == "doc-1"
        assert second.citations[0].document_id == "doc-2"

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_message(self):
        from app.rag.services.rag_service import LegalRAGService
        service = LegalRAGService()
        result = await service.answer("   ")
        assert "valid legal question" in result.answer.lower()
