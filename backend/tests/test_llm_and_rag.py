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
        assert "couldn't verify" in result.answer.lower()

    @pytest.mark.asyncio
    async def test_rag_malformed_llm_output_is_rejected_without_echoing_it(self, monkeypatch):
        from app.rag.services import rag_service as rag_module

        class MalformedLLM:
            async def generate(self, **kwargs):
                return "Private malformed provider payload [SOURCE 1]"

        monkeypatch.setattr(rag_module, "get_llm_provider", lambda: MalformedLLM())
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
        assert "Private malformed" not in result.answer
        assert result.generation_mode == "failed"

    @pytest.mark.asyncio
    async def test_generation_failure_returns_explicit_evidence_only_fallback(self, monkeypatch):
        from app.rag.services import rag_service as rag_module

        def unavailable_provider():
            raise RuntimeError("private provider detail")

        monkeypatch.setattr(rag_module, "get_llm_provider", unavailable_provider)
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

        assert result.generation_mode == "evidence_only"
        assert result.grounding_valid is True
        assert result.citations[0].chunk_id == "chunk-1"
        assert "LLM service is not available" in result.answer

    @pytest.mark.asyncio
    async def test_openrouter_failure_abstains_without_evidence_only_fallback(self, monkeypatch):
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

        assert result.generation_mode == "failed"
        assert result.abstained is True
        assert result.citations == []
        assert "OpenRouter answer generation is unavailable" in result.answer
        assert "private router error" not in result.answer

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
