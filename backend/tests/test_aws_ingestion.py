import io
import asyncio
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts import ingest_aws_judgments


def _status():
    return {
        "archives_seen": 0,
        "pdfs_total": 0,
        "pdfs_extracted": 0,
        "pending_documents": 0,
        "storage_limited": False,
        "failed_documents": 0,
        "recent_failures": [],
    }


def test_archive_extraction_is_safe_validated_and_restartable(tmp_path, monkeypatch):
    data_root = tmp_path / "data" / "raw" / "aws_open_data"
    monkeypatch.setattr(ingest_aws_judgments, "DATA_ROOT", data_root)
    archive_path = tmp_path / "judgments.tar"
    pdf_content = b"%PDF-1.7\nsample"
    unsafe_content = b"%PDF-1.7\nunsafe"

    with tarfile.open(archive_path, "w") as archive:
        for name, content in (
            ("case-1.pdf", pdf_content),
            ("../outside.pdf", unsafe_content),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    archive_path.with_suffix(".index.json").write_text(
        json.dumps({"file_count": 2}),
        encoding="utf-8",
    )

    destination = data_root / "supreme_court" / "extracted"
    first_status = _status()
    extracted = ingest_aws_judgments._extract_archive(
        archive_path, destination, first_status
    )

    assert extracted == 1
    assert (destination / "case-1.pdf").read_bytes() == pdf_content
    assert not (tmp_path / "outside.pdf").exists()
    assert first_status["failed_documents"] == 1

    resumed_status = _status()
    resumed = ingest_aws_judgments._extract_archive(
        archive_path, destination, resumed_status
    )

    assert resumed == 0
    assert resumed_status["pdfs_extracted"] == 1
    assert resumed_status["failed_documents"] == 1


def test_index_only_preserves_extraction_checkpoint(tmp_path, monkeypatch):
    status_path = tmp_path / "ingestion_status.json"
    status_path.write_text(
        json.dumps({
            "archives_seen": 18,
            "pdfs_total": 27598,
            "pdfs_extracted": 27598,
            "pending_documents": 0,
            "failed_documents": 2,
            "storage_limited": False,
            "recent_failures": [{"source": "existing.pdf"}],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(ingest_aws_judgments, "STATUS_PATH", status_path)

    progress = ingest_aws_judgments._read_extraction_checkpoint(index_only=True)

    assert progress == {
        "archives_seen": 18,
        "pdfs_total": 27598,
        "pdfs_extracted": 27598,
        "pending_documents": 0,
        "failed_documents": 2,
        "storage_limited": False,
        "recent_failures": [{"source": "existing.pdf"}],
    }


def test_index_only_refuses_to_replace_invalid_checkpoint(tmp_path, monkeypatch):
    status_path = tmp_path / "ingestion_status.json"
    status_path.write_text("{", encoding="utf-8")
    monkeypatch.setattr(ingest_aws_judgments, "STATUS_PATH", status_path)

    try:
        ingest_aws_judgments._read_extraction_checkpoint(index_only=True)
    except RuntimeError as exc:
        assert "refusing to replace it" in str(exc)
    else:
        raise AssertionError("An invalid ingestion checkpoint must not be discarded.")


def _index_status():
    return {
        "documents_processed": 0,
        "documents_indexed": 0,
        "chunks_indexed": 0,
        "duplicates_skipped": 0,
        "failed_documents": 0,
        "recent_failures": [],
    }


def test_manifest_indexing_stops_and_checkpoints_on_first_failure(tmp_path, monkeypatch):
    async def connected():
        return True

    async def initialized():
        return None

    class Repository:
        async def document_exists(self, _document_id):
            return False

    class Processor:
        async def process(self, *_args, **_kwargs):
            raise OSError("processing failed")

    monkeypatch.setattr(ingest_aws_judgments, "check_connection", connected)
    monkeypatch.setattr(ingest_aws_judgments, "init_db", initialized)
    monkeypatch.setattr(ingest_aws_judgments, "DocumentRepository", Repository)
    monkeypatch.setattr(ingest_aws_judgments, "DocumentProcessingService", Processor)
    monkeypatch.setattr(ingest_aws_judgments, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(ingest_aws_judgments, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(ingest_aws_judgments, "FAILURES_PATH", tmp_path / "failures.jsonl")
    status = _index_status()
    pdf = tmp_path / "first.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")

    try:
        asyncio.run(
            ingest_aws_judgments._index_documents(
                [(pdf, "first", {}), (pdf, "second", {})],
                status,
                stop_on_error=True,
                stop_on_duplicate=True,
            )
        )
    except OSError:
        pass
    else:
        raise AssertionError("A manifest batch must stop on its first indexing failure.")

    assert status["documents_processed"] == 1
    assert status["documents_indexed"] == 0
    assert status["failed_documents"] == 1
    assert len((tmp_path / "failures.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))[
        "documents_processed"
    ] == 1


def test_manifest_indexing_stops_on_unexpected_duplicate(tmp_path, monkeypatch):
    async def connected():
        return True

    async def initialized():
        return None

    class Repository:
        async def document_exists(self, _document_id):
            return True

    monkeypatch.setattr(ingest_aws_judgments, "check_connection", connected)
    monkeypatch.setattr(ingest_aws_judgments, "init_db", initialized)
    monkeypatch.setattr(ingest_aws_judgments, "DocumentRepository", Repository)
    monkeypatch.setattr(ingest_aws_judgments, "STATUS_PATH", tmp_path / "status.json")
    status = _index_status()

    try:
        asyncio.run(
            ingest_aws_judgments._index_documents(
                [(tmp_path / "first.pdf", "first", {}), (tmp_path / "second.pdf", "second", {})],
                status,
                stop_on_error=True,
                stop_on_duplicate=True,
            )
        )
    except ingest_aws_judgments.BatchSafetyError as exc:
        assert "Unexpected existing document ID" in str(exc)
    else:
        raise AssertionError("A manifest batch must stop on an unexpected duplicate.")

    assert status["documents_processed"] == 1
    assert status["duplicates_skipped"] == 1
    assert status["failed_documents"] == 0
    assert json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))[
        "duplicates_skipped"
    ] == 1
