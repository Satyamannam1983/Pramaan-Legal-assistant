"""LLM provider factory — lazy singleton based on configuration."""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_llm_instance = None


def _get_provider_config() -> tuple[str, str]:
    """Read LLM config from env vars (avoids stale Settings singleton in tests)."""
    provider = os.getenv("LLM_PROVIDER", "stub").lower().strip()
    model = os.getenv("LLM_MODEL", "placeholder")
    return provider, model


def get_llm_provider():
    """Return the configured LLM provider (singleton).

    Providers:
        - "openai" / "openai-compatible" → OpenAICompatibleLLM
        - "mock" / "stub" → MockLLM
    """
    global _llm_instance  # noqa: PLW0603
    if _llm_instance is not None:
        return _llm_instance

    provider, model = _get_provider_config()

    if provider in ("openai", "openai-compatible", "openai_compatible"):
        from app.rag.generation.openai_provider import OpenAICompatibleLLM  # noqa: PLC0415
        _llm_instance = OpenAICompatibleLLM()
        logger.info("LLM provider: OpenAI (model=%s)", model)
    elif provider in ("mock", "stub"):
        from app.rag.generation.mock_provider import MockLLM  # noqa: PLC0415
        _llm_instance = MockLLM(model=model)
        logger.info("LLM provider: Mock (model=%s)", model)
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            "Supported: openai, mock/stub."
        )

    return _llm_instance


def reset_llm_provider() -> None:
    """Reset the singleton (for testing)."""
    global _llm_instance  # noqa: PLW0603
    _llm_instance = None
