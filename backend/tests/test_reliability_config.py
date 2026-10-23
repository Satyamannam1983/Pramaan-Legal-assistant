import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.main import RequestSizeLimitMiddleware


def test_settings_accept_local_defaults():
    settings = Settings(_env_file=None)
    assert settings.app_env == "development"
    assert settings.cors_origins == "http://localhost:3000"
    assert settings.atlas_vector_search is False


def test_settings_accept_atlas_vector_search_configuration():
    settings = Settings(_env_file=None, atlas_vector_search=True)
    assert settings.atlas_vector_search is True


@pytest.mark.asyncio
async def test_index_command_redacts_database_credentials(monkeypatch, capsys):
    import importlib.util

    script = Path(__file__).parents[2] / "scripts" / "index_documents.py"
    module_spec = importlib.util.spec_from_file_location("index_documents", script)
    assert module_spec and module_spec.loader
    index_documents = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(index_documents)

    secret_uri = "mongodb+srv://private-user:private-password@cluster.example/pramaan"
    monkeypatch.setattr(index_documents.settings, "database_url", secret_uri)
    monkeypatch.setattr(index_documents, "check_connection", AsyncMock(return_value=False))

    exit_code = await index_documents.main(Path("processed"), show_progress=False)

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "MongoDB Atlas" in output
    assert secret_uri not in output
    assert "private-password" not in output


def test_test_database_override_never_falls_back_to_application_database():
    configured_test_database = os.getenv("TEST_DATABASE_URL")
    assert os.environ["DATABASE_URL"] == (
        configured_test_database
        or "mongodb://127.0.0.1:59999/test_nonexistent"
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"app_env": "staging"}, "APP_ENV"),
        ({"database_url": "https://example.test/db"}, "DATABASE_URL"),
        ({"cors_origins": "*"}, "CORS_ORIGINS"),
        ({"cors_origins": "https://example.test/path"}, "CORS_ORIGINS"),
        ({"max_upload_size_mb": 0}, "positive integers"),
        ({"max_upload_size_mb": 31}, "MAX_REQUEST_BODY_MB"),
        ({"max_comparison_documents": 3}, "MAX_COMPARISON_DOCUMENTS"),
        ({"chunk_overlap": 1200}, "CHUNK_OVERLAP"),
        ({"vector_weight": 0, "keyword_weight": 0}, "retrieval weight"),
        ({"llm_provider": "unknown"}, "LLM_PROVIDER"),
        (
            {"llm_provider": "openai", "openai_api_key": None},
            "OPENAI_API_KEY",
        ),
        (
            {"llm_provider": "openrouter", "openrouter_api_key": None},
            "OPENROUTER_API_KEY",
        ),
    ],
)
def test_settings_reject_invalid_configuration(overrides, message):
    with pytest.raises(ValidationError, match=message):
        Settings(_env_file=None, **overrides)


def test_production_requires_non_local_cors_origin_and_real_provider_key():
    with pytest.raises(ValidationError, match="CORS_ORIGINS"):
        Settings(_env_file=None, app_env="production")


def test_openrouter_configuration_requires_its_own_key():
    settings = Settings(
        _env_file=None,
        llm_provider="openrouter",
        llm_model="openrouter/free",
        openrouter_api_key="test-openrouter-key",
    )
    assert settings.llm_provider == "openrouter"


def test_request_size_middleware_rejects_chunked_body_without_content_length():
    async def run():
        app_called = False
        messages = [
            {"type": "http.request", "body": b"123", "more_body": True},
            {"type": "http.request", "body": b"4", "more_body": False},
        ]

        async def receive():
            return messages.pop(0)

        sent = []

        async def send(message):
            sent.append(message)

        async def app(scope, receive, send):
            nonlocal app_called
            app_called = True
            while True:
                message = await receive()
                if not message.get("more_body", False):
                    break

        middleware = RequestSizeLimitMiddleware(app, max_body_bytes=3)
        scope = {
            "type": "http",
            "method": "POST",
            "headers": [],
        }
        await middleware(scope, receive, send)
        assert app_called
        assert sent[0]["status"] == 413

    asyncio.run(run())


def test_request_size_middleware_rejects_malformed_content_length():
    async def run():
        sent = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        async def app(scope, receive, send):
            raise AssertionError("Application must not receive invalid request metadata.")

        middleware = RequestSizeLimitMiddleware(app, max_body_bytes=3)
        scope = {
            "type": "http",
            "method": "POST",
            "headers": [(b"content-length", b"not-a-number")],
        }
        await middleware(scope, receive, send)
        assert sent[0]["status"] == 400

    asyncio.run(run())
