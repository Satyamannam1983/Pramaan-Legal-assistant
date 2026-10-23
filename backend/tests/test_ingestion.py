import io
import json
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[2]))

from pypdf import PdfWriter
from app.rag.loaders import JSONLoader, PDFLoader, TXTLoader
from app.rag.processors import clean_text, process_pages
from app.rag.legal_chunker import LegalChunker
from app.rag.loaders.base import LoadedPage
from app.rag.loaders import LoaderError
from app.services.document_processing import DocumentProcessingService


def test_txt_and_json_loaders():
    assert TXTLoader().load(b"Section 1.\nText", "a.txt").pages[0].text.startswith("Section")
    loaded = JSONLoader().load(b'{"title":"Synthetic","content":"Section 1."}', "a.json")
    assert '"title": "Synthetic"' in loaded.pages[0].text


def test_loaders_reject_malformed_text_and_json():
    with pytest.raises(LoaderError, match="Malformed text"):
        TXTLoader().load(b"\xff", "bad.txt")
    with pytest.raises(LoaderError, match="Malformed JSON"):
        JSONLoader().load(b'{"title":', "bad.json")


def test_pdf_loader_preserves_page_numbers():
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buffer = io.BytesIO()
    writer.write(buffer)
    loaded = PDFLoader().load(buffer.getvalue(), "blank.pdf")
    assert loaded.pages[0].page_number == 1


def test_cleaning_normalizes_and_removes_repeated_headers():
    pages = process_pages([LoadedPage(1, "HEADER\nA   B"), LoadedPage(2, "HEADER\nC\r\nD")])
    assert pages[0].text == "A B"
    assert "HEADER" not in pages[1].text
    assert "\ufffd" not in clean_text("bad\ufffdtext")


def test_chunking_preserves_overlap_sections_and_traceability():
    pages = [LoadedPage(1, "Chapter 1\nSection 1. " + ("legal text " * 20)), LoadedPage(2, "Section 2. " + ("more text " * 20))]
    chunks = LegalChunker(chunk_size=100, chunk_overlap=20).chunk(pages)
    assert len(chunks) > 1
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert chunks[0].section
    assert any(chunk.section and chunk.section.startswith("Section 1.") for chunk in chunks)


@pytest.mark.asyncio
async def test_document_processing_rejects_empty_and_unsupported_documents():
    service = DocumentProcessingService()
    with pytest.raises(LoaderError, match="empty"):
        await service.process("empty.txt", b" \n")
    with pytest.raises(LoaderError, match="Unsupported"):
        await service.process("notice.docx", b"content")


@pytest.mark.asyncio
async def test_processing_preserves_source_identity_and_document_metadata():
    processed = await DocumentProcessingService().process(
        "notice.json",
        b'{"title":"Notice","court":"High Court","date":"2024-01-01","source_url":"https://example.test"}',
        document_id="notice-2024",
        include_all_chunks=True,
        metadata_overrides={"dataset_metadata": {"case": "known", "year": 2024}},
    )

    assert processed.metadata.document_id == "notice-2024"
    assert processed.metadata.court == "High Court"
    assert processed.metadata.date == "2024-01-01"
    assert processed.all_chunks
    chunk = processed.all_chunks[0]
    assert chunk.chunk_id == "notice-2024:0"
    assert chunk.document_id == processed.metadata.document_id
    assert chunk.court == processed.metadata.court
    assert chunk.date == processed.metadata.date
    assert chunk.source_url == processed.metadata.source_url
    assert json.loads(processed.metadata.metadata["dataset_metadata"]) == {
        "case": "known",
        "year": 2024,
    }


@pytest.mark.asyncio
async def test_offline_processing_uses_the_shared_processing_service(tmp_path):
    from scripts.process_documents import process_file, stable_document_id

    source = tmp_path / "notice.txt"
    source.write_text("Section 1. Legal notice text.", encoding="utf-8")
    offline = await process_file(source, tmp_path)
    api = await DocumentProcessingService().process(
        source.name,
        source.read_bytes(),
        include_all_chunks=True,
        document_id=stable_document_id(source, tmp_path),
        source="raw_file",
    )

    assert offline["metadata"]["document_id"] == api.metadata.document_id
    assert offline["pages"] == [page.model_dump(mode="json") for page in api.pages]
    assert offline["all_chunks"] == [
        chunk.model_dump(mode="json") for chunk in api.all_chunks
    ]


@pytest.mark.asyncio
async def test_offline_processing_reports_unsupported_files(tmp_path):
    from scripts.process_documents import process_file

    source = tmp_path / "notice.docx"
    source.write_bytes(b"content")
    result = await process_file(source, tmp_path)

    assert result["filename"] == "notice.docx"
    assert "Unsupported" in result["error"]


@pytest.mark.asyncio
async def test_offline_processing_reports_unreadable_files(tmp_path, monkeypatch):
    from scripts.process_documents import process_file

    source = tmp_path / "notice.txt"

    def deny_read(_path):
        raise PermissionError("permission denied")

    monkeypatch.setattr(Path, "read_bytes", deny_read)
    result = await process_file(source, tmp_path)

    assert result == {"filename": "notice.txt", "error": "The file could not be read."}


def test_offline_document_ids_are_repeatable_and_distinguish_nested_sources(tmp_path):
    from scripts.process_documents import stable_document_id

    first = tmp_path / "court-a" / "notice.txt"
    second = tmp_path / "court-b" / "notice.txt"

    first_id = stable_document_id(first, tmp_path)
    assert first_id == stable_document_id(first, tmp_path)
    assert first_id != stable_document_id(second, tmp_path)


def test_supreme_court_dataset_sidecar_preserves_available_metadata(tmp_path):
    from scripts.process_documents import _dataset_metadata

    sidecar = tmp_path / "judgment.json"
    sidecar.write_text(
        json.dumps(
            {
                "title": "MAMMAN KHAN versus STATE OF HARYANA",
                "petitioner": "MAMMAN KHAN",
                "respondent": "STATE OF HARYANA",
                "decision_date": "12-09-2025",
                "raw_html": (
                    "MAMMAN KHAN versus STATE OF HARYANA - [2025] 10 S.C.R. 131 "
                    "Decision Date : 12-09-2025 | "
                    "Case No : CRIMINAL APPEAL No. 4002/2025"
                ),
                "citation_year": 2025,
                "nc_display": "2025INSC1113",
                "path": "2025_10_131_151",
            }
        ),
        encoding="utf-8",
    )

    metadata = _dataset_metadata(sidecar, "2025_10_131_151_EN.pdf")

    assert metadata["title"] == "MAMMAN KHAN versus STATE OF HARYANA"
    assert metadata["case_name"] == "MAMMAN KHAN v. STATE OF HARYANA"
    assert metadata["case_number"] == "CRIMINAL APPEAL No. 4002/2025"
    assert metadata["citation"] == "2025 INSC 1113"
    assert metadata["date"] == "12-09-2025"
    assert metadata["source_url"].endswith(
        "/data/pdf/year=2025/english/2025_10_131_151_EN.pdf"
    )
    assert metadata["dataset_metadata"]["path"] == "2025_10_131_151"
    assert "CC BY 4.0" in metadata["dataset_attribution"]


@pytest.mark.asyncio
async def test_document_indexer_record_reuses_processed_metadata_and_chunks(monkeypatch):
    from app.rag.indexing import document_indexer

    class FakeEmbeddingService:
        def embed_documents(self, texts):
            return [[float(index)] for index, _ in enumerate(texts)]

    class FakeRepository:
        def __init__(self):
            self.document = None
            self.chunks = {}
            self.stale_cleanup = []

        async def upsert_document(self, document):
            self.document = document

        async def upsert_chunks(self, chunks):
            self.chunks.update({chunk.chunk_id: chunk for chunk in chunks})

        async def delete_stale_chunks(self, document_id, keep_chunk_ids):
            self.stale_cleanup.append((document_id, list(keep_chunk_ids)))
            self.chunks = {
                chunk_id: chunk
                for chunk_id, chunk in self.chunks.items()
                if chunk.document_id != document_id or chunk_id in keep_chunk_ids
            }

    repository = FakeRepository()
    monkeypatch.setattr(
        document_indexer,
        "DocumentRepository",
        lambda: repository,
    )
    indexer = document_indexer.DocumentIndexer(
        embedding_service=FakeEmbeddingService()
    )

    indexed = await indexer.index_record({
        "metadata": {
            "document_id": "upload-1",
            "title": "Uploaded notice",
            "document_type": "TXT",
            "source": "user_upload",
            "metadata": {},
        },
        "all_chunks": [
            {
                "chunk_id": "upload-1:0",
                "content": "Notice text",
                "chunk_index": 0,
                "document_type": "TXT",
                "source": "user_upload",
            }
        ],
    })

    assert indexed == 1
    assert repository.document.document_id == "upload-1"
    assert repository.chunks["upload-1:0"].embedding == [0.0]
    assert repository.stale_cleanup == [("upload-1", ["upload-1:0"])]


@pytest.mark.asyncio
async def test_reindexing_removes_obsolete_chunks(monkeypatch):
    from app.rag.indexing import document_indexer

    class FakeEmbeddingService:
        def embed_documents(self, texts):
            return [[0.1] for _ in texts]

    class FakeRepository:
        def __init__(self):
            self.chunks = {}

        async def upsert_document(self, document):
            pass

        async def upsert_chunks(self, chunks):
            self.chunks.update({chunk.chunk_id: chunk for chunk in chunks})

        async def delete_stale_chunks(self, document_id, keep_chunk_ids):
            self.chunks = {
                chunk_id: chunk
                for chunk_id, chunk in self.chunks.items()
                if chunk.document_id != document_id or chunk_id in keep_chunk_ids
            }

    repository = FakeRepository()
    monkeypatch.setattr(document_indexer, "DocumentRepository", lambda: repository)
    indexer = document_indexer.DocumentIndexer(embedding_service=FakeEmbeddingService())

    def record(contents):
        return {
            "metadata": {"document_id": "source-1", "metadata": {}},
            "all_chunks": [
                {"chunk_id": f"source-1:{i}", "content": content, "chunk_index": i}
                for i, content in enumerate(contents)
            ],
        }

    await indexer.index_record(record(["first", "second"]))
    await indexer.index_record(record(["updated"]))

    assert list(repository.chunks) == ["source-1:0"]
    assert repository.chunks["source-1:0"].content == "updated"


@pytest.mark.asyncio
async def test_indexing_failure_is_propagated_without_stale_cleanup(monkeypatch):
    from app.rag.indexing import document_indexer

    class FakeEmbeddingService:
        def embed_documents(self, texts):
            return [[0.1] for _ in texts]

    class FailingRepository:
        async def upsert_document(self, document):
            pass

        async def upsert_chunks(self, chunks):
            raise RuntimeError("database write failed")

        async def delete_stale_chunks(self, document_id, keep_chunk_ids):
            pytest.fail("Stale chunks must not be removed after an upsert failure.")

    monkeypatch.setattr(
        document_indexer,
        "DocumentRepository",
        lambda: FailingRepository(),
    )
    indexer = document_indexer.DocumentIndexer(embedding_service=FakeEmbeddingService())
    record = {
        "metadata": {"document_id": "source-1", "metadata": {}},
        "all_chunks": [{"chunk_id": "source-1:0", "content": "text", "chunk_index": 0}],
    }

    with pytest.raises(RuntimeError, match="database write failed"):
        await indexer.index_record(record)
