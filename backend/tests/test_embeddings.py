"""Tests for the embedding service layer.

These tests do NOT require PostgreSQL or torch.
The stub embedding service is used throughout.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.rag.embeddings.stub import StubEmbeddingService
from app.rag.embeddings.factory import get_embedding_service
from app.rag.embeddings.base import BaseEmbeddingService


# ---------------------------------------------------------------------------
# Stub embedding service
# ---------------------------------------------------------------------------

class TestStubEmbeddingService:
    def setup_method(self):
        self.svc = StubEmbeddingService(dimension=128)

    def test_embedding_dimension_property(self):
        assert self.svc.embedding_dimension == 128

    def test_embed_text_returns_correct_length(self):
        vec = self.svc.embed_text("What is a contract?")
        assert len(vec) == 128

    def test_embed_text_returns_floats(self):
        vec = self.svc.embed_text("legal text")
        assert all(isinstance(v, float) for v in vec)

    def test_embed_text_is_unit_length(self):
        vec = self.svc.embed_text("test query")
        mag = sum(v * v for v in vec) ** 0.5
        assert abs(mag - 1.0) < 1e-4, f"Expected unit vector, got magnitude {mag}"

    def test_embed_documents_batch(self):
        texts = ["contract formation", "breach of contract", "consideration", "agency"]
        vecs = self.svc.embed_documents(texts)
        assert len(vecs) == 4
        for vec in vecs:
            assert len(vec) == 128

    def test_embed_documents_empty(self):
        assert self.svc.embed_documents([]) == []

    def test_same_text_same_vector(self):
        """Stub must be deterministic — same text → same vector."""
        text = "habeas corpus petition"
        v1 = self.svc.embed_text(text)
        v2 = self.svc.embed_text(text)
        assert v1 == v2

    def test_different_texts_different_vectors(self):
        """Different texts should produce different vectors."""
        v1 = self.svc.embed_text("contract law")
        v2 = self.svc.embed_text("criminal law")
        assert v1 != v2

    def test_satisfies_protocol(self):
        """StubEmbeddingService must satisfy BaseEmbeddingService protocol."""
        assert isinstance(self.svc, BaseEmbeddingService)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestEmbeddingFactory:
    def test_stub_provider_returns_stub_service(self, monkeypatch):
        import app.rag.embeddings.factory as factory_mod  # noqa: PLC0415
        monkeypatch.setattr(factory_mod, "_instance", None)
        monkeypatch.setattr("app.config.settings.embedding_provider", "stub")
        monkeypatch.setattr("app.config.settings.embedding_dimension", 64)
        svc = factory_mod.get_embedding_service(force_new=True)
        assert isinstance(svc, StubEmbeddingService)

    def test_unknown_provider_raises(self, monkeypatch):
        import app.rag.embeddings.factory as factory_mod  # noqa: PLC0415
        monkeypatch.setattr(factory_mod, "_instance", None)
        monkeypatch.setattr("app.config.settings.embedding_provider", "nonexistent_provider")
        with pytest.raises(ValueError, match="Unknown EMBEDDING_PROVIDER"):
            factory_mod.get_embedding_service(force_new=True)

    def test_singleton_returns_same_instance(self, monkeypatch):
        import app.rag.embeddings.factory as factory_mod  # noqa: PLC0415
        monkeypatch.setattr(factory_mod, "_instance", None)
        monkeypatch.setattr("app.config.settings.embedding_provider", "stub")
        monkeypatch.setattr("app.config.settings.embedding_dimension", 64)
        svc1 = factory_mod.get_embedding_service(force_new=True)
        svc2 = factory_mod.get_embedding_service()
        assert svc1 is svc2
