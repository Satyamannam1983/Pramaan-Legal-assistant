"""Tests for the LLM providers and RAG service."""
from __future__ import annotations

import json
import pytest

from app.rag.generation.mock_provider import MockLLM
from app.rag.generation.factory import get_llm_provider, reset_llm_provider


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


class TestRAGService:
    @pytest.mark.asyncio
    async def test_empty_query_returns_message(self):
        from app.rag.services.rag_service import LegalRAGService
        service = LegalRAGService()
        result = await service.answer("")
        assert "valid legal question" in result.answer.lower()
        assert result.confidence == "low"

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_message(self):
        from app.rag.services.rag_service import LegalRAGService
        service = LegalRAGService()
        result = await service.answer("   ")
        assert "valid legal question" in result.answer.lower()
