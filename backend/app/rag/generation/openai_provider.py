"""OpenAI-compatible LLM provider.

Works with OpenAI, Azure OpenAI, and any OpenAI-compatible API
(Together, Groq, Fireworks, Ollama, etc.).
"""
from __future__ import annotations

import logging

from app.config import settings

logger = logging.getLogger(__name__)


class OpenAICompatibleLLM:
    """LLM provider using the OpenAI chat completions API.

    Requires `openai` package and a valid API key.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model or settings.llm_model
        self._api_key = api_key or settings.openai_api_key
        self._base_url = base_url or settings.llm_base_url
        self._client = None

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self):
        if self._client is not None:
            return self._client

        if not self._api_key:
            raise RuntimeError(
                "No API key configured. Set OPENAI_API_KEY in .env."
            )

        try:
            from openai import AsyncOpenAI  # noqa: PLC0415

            kwargs: dict = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url

            self._client = AsyncOpenAI(**kwargs)
            logger.info("OpenAI client initialised (model=%s)", self._model)
        except ImportError as exc:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai"
            ) from exc

        return self._client

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a completion using the OpenAI chat API."""
        client = self._get_client()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            return content
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            raise RuntimeError(f"LLM generation failed: {exc}") from exc
