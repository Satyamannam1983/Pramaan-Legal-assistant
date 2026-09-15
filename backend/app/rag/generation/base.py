"""LLM abstraction — provider-agnostic interface for language model generation."""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class BaseLLM(Protocol):
    """Protocol every LLM provider must satisfy."""

    @property
    def model_name(self) -> str:
        """Name of the underlying model."""
        ...

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a completion from system + user prompts.

        Returns the raw model output as a string.
        """
        ...
