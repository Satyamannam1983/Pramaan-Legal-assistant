"""Mock LLM provider for testing.

Returns deterministic, structured responses without calling any external API.
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


class MockLLM:
    """Deterministic mock LLM for unit tests and development.

    Always returns a valid JSON LegalAnswer structure.
    """

    provider_name = "stub"

    def __init__(self, model: str = "mock-model") -> None:
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """Return a mock structured legal answer."""
        logger.debug("MockLLM.generate called (would use %d tokens)", max_tokens)

        # Extract source count from user prompt to make citations realistic
        source_count = user_prompt.count("[SOURCE ")
        citations = []
        for i in range(1, min(source_count + 1, 4)):
            citations.append({
                "citation_id": f"SOURCE {i}",
                "chunk_id": f"mock_chunk_{i}",
                "document_id": f"mock_doc_{i}",
                "title": f"Mock Document {i}",
                "section": f"Section {i}",
                "page_start": i,
                "page_end": i,
                "source_url": None,
            })

        source_refs = " ".join(
            f"[SOURCE {i}]" for i in range(1, min(source_count + 1, 4))
        )
        answer = {
            "answer": (
                f"Based on the retrieved legal sources {source_refs}, this is a mock response. "
                "Configure a real LLM provider (OPENAI_API_KEY) for actual legal analysis."
            ),
            "key_points": [
                f"This is a mock response from the test LLM provider. {source_refs}".strip(),
                f"Set LLM_PROVIDER=openai and provide OPENAI_API_KEY for real answers. {source_refs}".strip(),
            ],
            "citations": citations,
            "limitations": [
                "Mock response — not based on actual legal analysis.",
            ],
            "confidence": "low",
        }

        return json.dumps(answer)
