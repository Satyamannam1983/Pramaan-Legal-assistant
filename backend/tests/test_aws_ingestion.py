import io
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
