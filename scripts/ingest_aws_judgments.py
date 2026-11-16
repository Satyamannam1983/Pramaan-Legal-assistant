"""Safely extract and incrementally index selected AWS court judgments.

The script uses the existing PDF processor and MongoDB document indexer.
Extracted PDFs, progress, and per-document failures stay under data/raw/.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from pymongo.errors import PyMongoError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from app.config import settings
from app.db.connection import check_connection, init_db
from app.db.repository import DocumentRepository
from app.rag.indexing.document_indexer import DocumentIndexer
from app.services.document_processing import DocumentProcessingService
from scripts.process_documents import _dataset_metadata

DATA_ROOT = ROOT / "data" / "raw" / "aws_open_data"
STATUS_PATH = DATA_ROOT / "ingestion_status.json"
FAILURES_PATH = DATA_ROOT / "ingestion_failures.jsonl"
LOG = logging.getLogger("ingest_aws_judgments")
PDF_SIGNATURE = b"%PDF-"
RESERVED_BYTES = 5 * 1024**3


class DatabaseUnavailableError(RuntimeError):
    """Raised when extracted judgments cannot be indexed because MongoDB is down."""


class BatchSafetyError(RuntimeError):
    """Raised when a bounded batch no longer matches its preflight assumptions."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(status: dict[str, Any]) -> None:
    status["updated_at"] = _utc_now()
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for attempt in range(5):
        try:
            os.replace(temporary, STATUS_PATH)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1 * (attempt + 1))


def _read_extraction_checkpoint(index_only: bool) -> dict[str, Any]:
    if not index_only or not STATUS_PATH.is_file():
        return {}
    try:
        previous = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "The existing ingestion checkpoint is unreadable; refusing to replace it."
        ) from exc
    if not isinstance(previous, dict):
        raise RuntimeError(
            "The existing ingestion checkpoint is invalid; refusing to replace it."
        )

    preserved: dict[str, Any] = {}
    for key in (
        "archives_seen",
        "pdfs_total",
        "pdfs_extracted",
        "pending_documents",
        "failed_documents",
    ):
        value = previous.get(key, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RuntimeError(
                "The existing ingestion checkpoint is invalid; refusing to replace it."
            )
        preserved[key] = value
    recent_failures = previous.get("recent_failures", [])
    if not isinstance(recent_failures, list):
        raise RuntimeError(
            "The existing ingestion checkpoint is invalid; refusing to replace it."
        )
    preserved["recent_failures"] = recent_failures[-100:]
    storage_limited = previous.get("storage_limited", False)
    if not isinstance(storage_limited, bool):
        raise RuntimeError(
            "The existing ingestion checkpoint is invalid; refusing to replace it."
        )
    preserved["storage_limited"] = storage_limited
    return preserved


def _record_failure(status: dict[str, Any], source: str, reason: str) -> None:
    entry = {"source": source, "reason": reason, "at": _utc_now()}
    status["failed_documents"] += 1
    status["recent_failures"] = (status["recent_failures"] + [entry])[-100:]
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with FAILURES_PATH.open("a", encoding="utf-8") as failures:
        failures.write(json.dumps(entry, ensure_ascii=False) + "\n")
    LOG.warning("Failed %s: %s", source, reason)


def _pdf_is_valid(path: Path) -> bool:
    try:
        if path.stat().st_size < len(PDF_SIGNATURE):
            return False
        with path.open("rb") as pdf:
            return pdf.read(len(PDF_SIGNATURE)) == PDF_SIGNATURE
    except OSError:
        return False


def _extract_archive(
    archive_path: Path,
    destination: Path,
    status: dict[str, Any],
) -> int:
    if not archive_path.is_file():
        _record_failure(status, str(archive_path), "Archive file is missing.")
        return 0
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            members = archive.getmembers()
            pdf_members = [
                member for member in members
                if member.isfile() and PurePosixPath(member.name).suffix.lower() == ".pdf"
            ]
            index_path = archive_path.with_suffix(".index.json")
            try:
                archive_index = json.loads(index_path.read_text(encoding="utf-8"))
                expected_count = int(archive_index["file_count"])
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                _record_failure(
                    status,
                    str(archive_path),
                    "The matching archive index is missing or invalid; extraction was skipped.",
                )
                return 0
            if expected_count != len(pdf_members):
                _record_failure(
                    status,
                    str(archive_path),
                    f"Archive index lists {expected_count} files but TAR contains {len(pdf_members)} PDFs.",
                )
                return 0
            status["archives_seen"] += 1
            status["pdfs_total"] += len(pdf_members)
            for index, member in enumerate(pdf_members):
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
                    _record_failure(status, f"{archive_path}!{member.name}", "Unsafe TAR member path.")
                    continue
                target = destination.joinpath(*relative.parts)
                if not target.resolve().is_relative_to(destination.resolve()):
                    _record_failure(status, f"{archive_path}!{member.name}", "TAR member escapes extraction directory.")
                    continue
                if target.is_file():
                    if _pdf_is_valid(target):
                        status["pdfs_extracted"] += 1
                        continue
                    _record_failure(status, str(target), "Existing extracted file is not a valid PDF; preserved.")
                    continue

                if shutil.disk_usage(DATA_ROOT.parent).free - member.size < RESERVED_BYTES:
                    status["pending_documents"] += len(pdf_members) - index
                    status["storage_limited"] = True
                    LOG.error("Stopping extraction to preserve the configured free-space reserve.")
                    break

                source = archive.extractfile(member)
                if source is None:
                    _record_failure(status, f"{archive_path}!{member.name}", "TAR member could not be read.")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".pramaan-partial")
                try:
                    with source, temporary.open("wb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                    if not _pdf_is_valid(temporary):
                        _record_failure(status, f"{archive_path}!{member.name}", "TAR member did not contain a valid PDF.")
                        continue
                    if target.exists():
                        _record_failure(status, str(target), "Destination appeared during extraction; preserved.")
                        continue
                    os.replace(temporary, target)
                    extracted += 1
                    status["pdfs_extracted"] += 1
                except OSError as exc:
                    _record_failure(
                        status,
                        f"{archive_path}!{member.name}",
                        f"Extraction I/O error ({type(exc).__name__}).",
                    )
                finally:
                    if temporary.exists():
                        temporary.unlink()
                if status["pdfs_extracted"] % 100 == 0:
                    _write_status(status)
            if status["storage_limited"]:
                _write_status(status)
                return extracted
    except (OSError, tarfile.TarError) as exc:
        _record_failure(status, str(archive_path), f"TAR validation or read error ({type(exc).__name__}).")
    return extracted


def _read_supreme_metadata(metadata_dir: Path, filename: str) -> dict[str, Any]:
    stem = Path(filename).stem
    metadata_stem = stem[:-3] if stem.endswith("_EN") else stem
    metadata_path = metadata_dir / f"{metadata_stem}.json"
    if metadata_path.is_file():
        return _dataset_metadata(metadata_path, filename)
    year_match = re.match(r"(\d{4})_", stem)
    year = year_match.group(1) if year_match else "unknown"
    key = f"data/pdf/year={year}/english/{filename}"
    return {
        "title": Path(filename).stem.replace("_", " "),
        "source": "Indian Supreme Court Judgments dataset (eCourts)",
        "source_url": f"https://indian-supreme-court-judgments.s3.amazonaws.com/{key}",
        "jurisdiction": "India",
        "court": "Supreme Court of India",
        "language": "English",
        "document_type": "judgment",
        "dataset_attribution": "Indian Supreme Court Judgments dataset; source: eCourts; license: CC BY 4.0",
    }


def _load_high_court_metadata(year: int, court_id: str) -> dict[str, dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("The high-court metadata reader requires pyarrow.") from exc

    partition = (
        DATA_ROOT / "high_courts" / "metadata" / "parquet"
        / f"year={year}" / f"court={court_id}"
    )
    records: dict[str, dict[str, Any]] = {}
    for parquet_path in sorted(partition.rglob("*.parquet")):
        table = parquet.read_table(parquet_path)
        for row in table.to_pylist():
            pdf_link = row.get("pdf_link")
            if not isinstance(pdf_link, str):
                continue
            name = PurePosixPath(pdf_link).name
            if not name.lower().endswith(".pdf"):
                continue
            row["dataset_bench"] = parquet_path.parent.name.removeprefix("bench=")
            records.setdefault(name, row)
    return records


def _high_court_metadata(
    row: dict[str, Any] | None,
    filename: str,
    year: int,
    court_id: str,
) -> dict[str, Any]:
    court = (row or {}).get("court") or f"High Court ({court_id})"
    pdf_link = (row or {}).get("pdf_link") or filename
    value_date = (row or {}).get("decision_date")
    if hasattr(value_date, "isoformat"):
        value_date = value_date.isoformat()[:10]
    return {
        "title": (row or {}).get("title") or Path(filename).stem,
        "source": "Indian High Court Judgments dataset (eCourts)",
        "source_url": f"https://indian-high-court-judgments.s3.amazonaws.com/{pdf_link}",
        "jurisdiction": "India",
        "court": court,
        "case_name": (row or {}).get("title"),
        "case_number": (row or {}).get("cnr"),
        "date": str(value_date) if value_date else None,
        "language": "English",
        "document_type": "judgment",
        "dataset_attribution": "Indian High Court Judgments dataset; source: eCourts; license: CC BY 4.0",
        "dataset_court_code": (row or {}).get("court_code") or court_id,
        "dataset_bench": (row or {}).get("dataset_bench"),
        "dataset_judge": (row or {}).get("judge"),
        "dataset_disposal_nature": (row or {}).get("disposal_nature"),
        "dataset_source_year": year,
    }


def _source_documents(
    years: list[int],
    hc_year: int,
    hc_id: str,
    include_high_court: bool = True,
) -> list[tuple[Path, str, dict[str, Any]]]:
    documents: list[tuple[Path, str, dict[str, Any]]] = []
    seen_document_ids: set[str] = set()
    for year in years:
        pdf_dir = DATA_ROOT / "supreme_court" / "extracted" / f"year={year}" / "english"
        metadata_dir = DATA_ROOT / "supreme_court" / "metadata" / "json" / f"year={year}"
        if not pdf_dir.exists():
            continue
        for path in sorted(pdf_dir.rglob("*.pdf")):
            try:
                metadata = _read_supreme_metadata(metadata_dir, path.name)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                metadata = {}
                LOG.warning("Metadata unavailable for %s (%s).", path.name, type(exc).__name__)
            metadata["dataset_source_archive"] = str(
                path.relative_to(DATA_ROOT / "supreme_court" / "extracted").parent
            )
            documents.append((path, path.stem, metadata))
            seen_document_ids.add(path.stem)

    legacy_batch = ROOT / "data" / "raw" / "supreme_court_english_2023_batch"
    legacy_pdf_dir = legacy_batch / "pdfs"
    if legacy_pdf_dir.exists():
        for path in sorted(legacy_pdf_dir.rglob("*.pdf")):
            if path.stem in seen_document_ids:
                continue
            metadata = _read_supreme_metadata(legacy_batch / "metadata", path.name)
            documents.append((path, path.stem, metadata))
            seen_document_ids.add(path.stem)

    if include_high_court:
        high_court_root = (
            DATA_ROOT / "high_courts" / "extracted"
            / f"year={hc_year}" / f"court={hc_id}"
        )
        if high_court_root.exists():
            metadata_by_name = _load_high_court_metadata(hc_year, hc_id)
            for path in sorted(high_court_root.rglob("*.pdf")):
                metadata = _high_court_metadata(
                    metadata_by_name.get(path.name), path.name, hc_year, hc_id
                )
                metadata["dataset_source_archive"] = str(
                    path.relative_to(DATA_ROOT / "high_courts" / "extracted").parent
                )
                source_key = metadata["source_url"]
                document_id = f"aws-hc-{hashlib.sha256(source_key.encode('utf-8')).hexdigest()[:24]}"
                documents.append((path, document_id, metadata))
    return documents


def _parse_years(value: str) -> list[int]:
    years = set()
    for item in value.split(","):
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError("Year ranges must be in ascending order.")
            years.update(range(start, end + 1))
        else:
            years.add(int(item))
    if not years or any(year < 1950 or year > 2026 for year in years):
        raise ValueError("Years must be between 1950 and 2026.")
    return sorted(years)


async def _index_documents(
    documents: list[tuple[Path, str, dict[str, Any]]],
    status: dict[str, Any],
    *,
    stop_on_error: bool = False,
    stop_on_duplicate: bool = False,
) -> None:
    batch = status.get("batch")
    status["documents_to_index"] = (
        batch.get("size", len(documents))
        if isinstance(batch, dict)
        else len(documents)
    )
    _write_status(status)
    if not await check_connection():
        raise DatabaseUnavailableError
    try:
        await init_db()
    except PyMongoError:
        raise DatabaseUnavailableError from None
    repository = DocumentRepository()
    indexer = DocumentIndexer()
    processor = DocumentProcessingService()

    for index, (path, document_id, metadata) in enumerate(documents, start=1):
        status["documents_processed"] += 1
        try:
            if await repository.document_exists(document_id):
                status["duplicates_skipped"] += 1
                if stop_on_duplicate:
                    _write_status(status)
                    raise BatchSafetyError(
                        f"Unexpected existing document ID in selected batch: {document_id}"
                    )
            elif not _pdf_is_valid(path):
                _record_failure(status, str(path), "Extracted file failed PDF signature validation.")
                if stop_on_error:
                    _write_status(status)
                    raise BatchSafetyError("Selected PDF failed signature validation.")
            else:
                processed = await processor.process(
                    path.name,
                    path.read_bytes(),
                    include_all_chunks=True,
                    document_id=document_id,
                    source="aws_open_data",
                    metadata_overrides=metadata,
                )
                chunks = await indexer.index_record(
                    processed.model_dump(mode="json")
                )
                status["documents_indexed"] += 1
                status["chunks_indexed"] += chunks
        except Exception as exc:
            if isinstance(exc, BatchSafetyError):
                raise
            _record_failure(
                status,
                str(path),
                f"Processing or indexing error ({type(exc).__name__}).",
            )
            if stop_on_error:
                _write_status(status)
                raise
        if index % 10 == 0 or index == len(documents):
            _write_status(status)
            LOG.info(
                "Indexed progress %d/%d; added=%d chunks=%d duplicates=%d failures=%d",
                index,
                len(documents),
                status["documents_indexed"],
                status["chunks_indexed"],
                status["duplicates_skipped"],
                status["failed_documents"],
            )


def _extract_selected_archives(
    years: list[int],
    hc_year: int,
    hc_id: str,
    status: dict[str, Any],
    skip_high_court: bool,
) -> None:
    for year in years:
        archive = DATA_ROOT / "supreme_court" / "data" / "tar" / f"year={year}" / "english" / "english.tar"
        destination = DATA_ROOT / "supreme_court" / "extracted" / f"year={year}" / "english"
        if archive.is_file():
            _extract_archive(archive, destination, status)
        if status["storage_limited"]:
            return
    if skip_high_court:
        return
    hc_archive_dir = (
        DATA_ROOT / "high_courts" / "data" / "tar"
        / f"year={hc_year}" / f"court={hc_id}"
    )
    if hc_archive_dir.exists():
        for archive in sorted(hc_archive_dir.rglob("*.tar")):
            destination = (
                DATA_ROOT / "high_courts" / "extracted"
                / f"year={hc_year}" / f"court={hc_id}"
            )
            _extract_archive(archive, destination, status)
            if status["storage_limited"]:
                return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supreme-years", default="2010-2026")
    parser.add_argument("--high-court-year", type=int, default=2023)
    parser.add_argument("--high-court-id", default="5_15")
    parser.add_argument("--skip-high-court", action="store_true")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    try:
        years = _parse_years(args.supreme_years)
    except ValueError as exc:
        parser.error(str(exc))
    if args.index_only and args.extract_only:
        parser.error("--index-only and --extract-only cannot be combined.")
    if not re.fullmatch(r"\d+_\d+", args.high_court_id):
        parser.error("--high-court-id must use the dataset's numeric court code format.")
    if args.high_court_year < 1950 or args.high_court_year > 2026:
        parser.error("--high-court-year must be between 1950 and 2026.")

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        extraction_progress = _read_extraction_checkpoint(args.index_only)
    except RuntimeError as exc:
        LOG.error("%s", exc)
        return 1
    status = {
        "status": "running",
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "selection": {
            "supreme_court_english_years": years,
            "high_court_year": args.high_court_year,
            "high_court_id": args.high_court_id,
        },
        "archives_seen": extraction_progress.get("archives_seen", 0),
        "pdfs_total": extraction_progress.get("pdfs_total", 0),
        "pdfs_extracted": extraction_progress.get("pdfs_extracted", 0),
        "documents_to_index": 0,
        "documents_processed": 0,
        "documents_indexed": 0,
        "chunks_indexed": 0,
        "duplicates_skipped": 0,
        "failed_documents": extraction_progress.get("failed_documents", 0),
        "pending_documents": extraction_progress.get("pending_documents", 0),
        "storage_limited": extraction_progress.get("storage_limited", False),
        "recent_failures": extraction_progress.get("recent_failures", []),
    }
    _write_status(status)

    try:
        if not args.index_only:
            _extract_selected_archives(
                years,
                args.high_court_year,
                args.high_court_id,
                status,
                args.skip_high_court,
            )
            _write_status(status)
        if not args.extract_only:
            sources = _source_documents(
                years,
                args.high_court_year,
                args.high_court_id,
                include_high_court=not args.skip_high_court,
            )
            asyncio.run(_index_documents(sources, status))
        if status["failed_documents"]:
            status["status"] = "completed_with_errors"
        elif args.extract_only:
            status["status"] = "extraction_complete"
        else:
            status["status"] = "completed"
        status["completed_at"] = _utc_now()
        _write_status(status)
        LOG.info(
            "Ingestion finished: extracted=%d indexed=%d chunks=%d duplicates=%d failed=%d",
            status["pdfs_extracted"],
            status["documents_indexed"],
            status["chunks_indexed"],
            status["duplicates_skipped"],
            status["failed_documents"],
        )
        return 1 if status["failed_documents"] else 0
    except Exception as exc:
        status["status"] = "failed"
        if isinstance(exc, DatabaseUnavailableError):
            status["fatal_error"] = "MongoDB is unavailable; extracted PDFs are preserved for retry."
        else:
            status["fatal_error"] = f"Ingestion stopped ({type(exc).__name__})."
        status["completed_at"] = _utc_now()
        _write_status(status)
        if isinstance(exc, DatabaseUnavailableError):
            LOG.error("MongoDB is unavailable; extracted PDFs are preserved for retry.")
        else:
            LOG.error("Dataset ingestion stopped (error_type=%s).", type(exc).__name__)
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
