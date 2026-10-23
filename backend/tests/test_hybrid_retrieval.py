"""Tests for hybrid retrieval and RRF fusion."""
from __future__ import annotations

import pytest

from app.rag.retrieval.hybrid_retriever import (
    HybridRetriever,
    reciprocal_rank_fusion,
)
from app.rag.retrieval.schemas import FusedResult, KeywordResult, RetrievalResult


# ---------------------------------------------------------------------------
# RRF fusion tests
# ---------------------------------------------------------------------------

class TestReciprocalRankFusion:
    def _make_vresult(self, chunk_id: str, score: float = 0.5, rank: int = 1) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk_id,
            content=f"content of {chunk_id}",
            document_id="doc1",
            vector_score=score,
            vector_rank=rank,
        )

    def _make_kresult(self, chunk_id: str, score: float = 5.0, rank: int = 1) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk_id,
            content=f"content of {chunk_id}",
            document_id="doc1",
            keyword_score=score,
        )

    def test_single_list_passthrough(self):
        vresults = [
            self._make_vresult("c1", 0.9, 1),
            self._make_vresult("c2", 0.8, 2),
        ]
        fused = reciprocal_rank_fusion([vresults])
        assert len(fused) == 2
        assert fused[0].chunk_id == "c1"
        assert fused[0].rrf_score > fused[1].rrf_score

    def test_merge_duplicates(self):
        # Same chunk in both lists
        vresults = [self._make_vresult("c1", 0.9, 1)]
        kresults = [self._make_kresult("c1", 10.0)]
        fused = reciprocal_rank_fusion([vresults, kresults])
        assert len(fused) == 1
        # RRF score should be sum from both lists
        expected = 0.6 / (60 + 1) + 0.4 / (60 + 1)
        assert abs(fused[0].rrf_score - expected) < 1e-6

    def test_ranking_by_rrf(self):
        # c1 is rank 1 in list A, rank 2 in list B
        # c2 is rank 2 in list A, rank 1 in list B
        list_a = [
            self._make_vresult("c1", 0.9, 1),
            self._make_vresult("c2", 0.8, 2),
        ]
        list_b = [
            self._make_kresult("c2", 10.0),
            self._make_kresult("c1", 8.0),
        ]
        fused = reciprocal_rank_fusion([list_a, list_b])
        assert len(fused) == 2
        expected_c1 = 0.6 / 61 + 0.4 / 62
        expected_c2 = 0.6 / 62 + 0.4 / 61
        by_id = {result.chunk_id: result for result in fused}
        assert by_id["c1"].rrf_score == pytest.approx(expected_c1)
        assert by_id["c2"].rrf_score == pytest.approx(expected_c2)

    def test_empty_lists(self):
        fused = reciprocal_rank_fusion([[], []])
        assert fused == []

    def test_one_empty_list(self):
        vresults = [self._make_vresult("c1", 0.9, 1)]
        fused = reciprocal_rank_fusion([vresults, []])
        assert len(fused) == 1
        assert fused[0].chunk_id == "c1"

    def test_rrf_k_constant(self):
        vresults = [self._make_vresult("c1", 0.9, 1)]
        # With k=1, first rank gets 1/2 = 0.5
        fused = reciprocal_rank_fusion([vresults], k=1)
        assert abs(fused[0].rrf_score - 0.5) < 1e-6

    def test_metadata_preserved(self):
        vresults = [
            RetrievalResult(
                chunk_id="c1",
                content="hello",
                document_id="doc1",
                title="Title",
                section="Section 1",
                vector_score=0.9,
                vector_rank=1,
            )
        ]
        fused = reciprocal_rank_fusion([vresults])
        assert fused[0].title == "Title"
        assert fused[0].section == "Section 1"
        assert fused[0].vector_score == 0.9
        assert fused[0].vector_rank == 1

    def test_duplicate_within_one_ranked_list_contributes_only_once(self):
        results = [
            self._make_vresult("c1", 0.9, 1),
            self._make_vresult("c1", 0.8, 2),
        ]
        fused = reciprocal_rank_fusion([results])
        assert len(fused) == 1
        assert fused[0].rrf_score == pytest.approx(1 / 61)

    def test_ties_have_deterministic_document_and_chunk_order(self):
        first_list = [
            RetrievalResult(chunk_id="z", content="", document_id="doc-b"),
            RetrievalResult(chunk_id="a", content="", document_id="doc-a"),
        ]
        second_list = [
            RetrievalResult(chunk_id="a", content="", document_id="doc-a"),
            RetrievalResult(chunk_id="z", content="", document_id="doc-b"),
        ]
        forward = reciprocal_rank_fusion([first_list, second_list], weights=[1, 1])
        reverse = reciprocal_rank_fusion([first_list[::-1], second_list[::-1]], weights=[1, 1])
        assert [(result.document_id, result.chunk_id) for result in forward] == [
            ("doc-a", "a"),
            ("doc-b", "z"),
        ]
        assert [(result.document_id, result.chunk_id) for result in reverse] == [
            ("doc-a", "a"),
            ("doc-b", "z"),
        ]

    def test_same_chunk_identifier_from_different_documents_is_not_merged(self):
        results = [
            RetrievalResult(chunk_id="chunk-1", content="a", document_id="doc-a"),
            RetrievalResult(chunk_id="chunk-1", content="b", document_id="doc-b"),
        ]
        fused = reciprocal_rank_fusion([results], weights=[1])
        assert len(fused) == 2
        assert {result.document_id for result in fused} == {"doc-a", "doc-b"}

    def test_configurable_weights_and_invalid_rrf_parameters(self):
        vector = [self._make_vresult("vector", 0.8, 1)]
        keyword = [self._make_kresult("keyword", 9.0)]
        fused = reciprocal_rank_fusion([vector, keyword], k=10, weights=[0.2, 0.8])
        by_id = {result.chunk_id: result for result in fused}
        assert by_id["vector"].rrf_score == pytest.approx(0.2 / 11)
        assert by_id["keyword"].rrf_score == pytest.approx(0.8 / 11)
        with pytest.raises(ValueError, match="positive"):
            reciprocal_rank_fusion([vector], k=0)

    def test_scores_preserved(self):
        vresults = [self._make_vresult("c1", 0.9, 1)]
        kresults = [self._make_kresult("c1", 10.0)]
        fused = reciprocal_rank_fusion([vresults, kresults])
        assert fused[0].vector_score == 0.9
        assert fused[0].keyword_score == 10.0
        assert fused[0].vector_rank == 1

    def test_many_chunks_sorted_correctly(self):
        # Create 10 chunks with known ranking
        vresults = [
            self._make_vresult(f"c{i}", 0.9 - i * 0.05, i) for i in range(1, 11)
        ]
        fused = reciprocal_rank_fusion([vresults])
        assert len(fused) == 10
        # Verify sorted by RRF score descending
        for i in range(len(fused) - 1):
            assert fused[i].rrf_score >= fused[i + 1].rrf_score


class TestHybridRetriever:
    @pytest.mark.asyncio
    async def test_semantic_and_exact_term_results_share_query_and_provenance(self):
        class Vector:
            async def retrieve(self, query, top_k, document_id):
                assert query == "Section 302 IPC"
                assert document_id == "ipc"
                return [
                    RetrievalResult(
                        chunk_id="ipc:302",
                        content="Section 302 punishment",
                        document_id="ipc",
                        section="Section 302",
                        source="official",
                        source_url="https://example.test/ipc",
                        vector_score=0.91,
                    )
                ]

        class Keyword:
            async def retrieve(self, query, top_k, document_id):
                assert query == "Section 302 IPC"
                assert document_id == "ipc"
                return [
                    KeywordResult(
                        chunk_id="ipc:302",
                        content="Section 302 punishment",
                        document_id="ipc",
                        section="Section 302",
                        source="official",
                        source_url="https://example.test/ipc",
                        keyword_score=4.2,
                    )
                ]

        retriever = HybridRetriever(Vector(), Keyword())
        results = await retriever.retrieve("  Section 302 IPC  ", document_id="ipc")

        assert len(results) == 1
        assert results[0].document_id == "ipc"
        assert results[0].chunk_id == "ipc:302"
        assert results[0].section == "Section 302"
        assert results[0].source_url == "https://example.test/ipc"
        assert results[0].vector_score == 0.91
        assert results[0].keyword_score == 4.2
        assert results[0].vector_rank == results[0].keyword_rank == 1

    @pytest.mark.asyncio
    async def test_no_results_returns_empty(self):
        class Empty:
            async def retrieve(self, query, top_k, document_id):
                return []

        assert await HybridRetriever(Empty(), Empty()).retrieve("consideration") == []

    @pytest.mark.asyncio
    async def test_both_retrieval_errors_are_reported(self):
        class Broken:
            async def retrieve(self, query, top_k, document_id):
                raise RuntimeError("private connection details")

        with pytest.raises(RuntimeError, match="[Bb]oth dense and lexical"):
            await HybridRetriever(Broken(), Broken()).retrieve("consideration")

    @pytest.mark.asyncio
    async def test_one_failed_retriever_keeps_successful_results(self):
        class Broken:
            async def retrieve(self, query, top_k, document_id):
                raise RuntimeError("unavailable")

        class Semantic:
            async def retrieve(self, query, top_k, document_id):
                return [RetrievalResult(
                    chunk_id="doc:1",
                    content="Relevant legal passage",
                    document_id="doc",
                    vector_score=0.8,
                )]

        results = await HybridRetriever(Semantic(), Broken()).retrieve("breach")
        assert [result.chunk_id for result in results] == ["doc:1"]
