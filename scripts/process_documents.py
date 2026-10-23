"""Process raw documents and emit JSONL with ALL chunks for indexing.

Each JSONL line contains:
  - metadata: document-level metadata
  - pages: list of cleaned page objects
  - chunk_count: total number of chunks
  - sample_chunks: first 3 chunks (for API preview)
  - all_chunks: ALL chunks (used by index_documents.py)

Usage
-----
  python scripts/process_documents.py
  python scripts/process_documents.py data/raw/sample --output data/processed/sample.jsonl
"""
import argparse
import asyncio
import hashlib
from html.parser import HTMLParser
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))

from app.rag.loaders import LoaderError
from app.services.document_processing import DocumentProcessingService

logger = logging.getLogger(__name__)
DATASET_BUCKET_URL = "https://indian-supreme-court-judgments.s3.amazonaws.com"
DATASET_ATTRIBUTION = (
    "Indian Supreme Court Judgments dataset; source: eCourts; license: CC BY 4.0"
)


class _MetadataTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


def _dataset_metadata(metadata_path: Path, filename: str) -> dict:
    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not isinstance(record.get("raw_html"), str):
        raise ValueError("Dataset metadata must contain a raw_html string.")

    parser = _MetadataTextParser()
    parser.feed(record["raw_html"])
    text = " ".join(" ".join(parser.parts).split())
    title_match = re.search(
        r"^(.+?)\s+(?:versus|v\.?)\s+(.+?)\s+-\s+\[\d{4}\]",
        text,
        re.IGNORECASE,
    )
    case_name_from_html = (
        f"{title_match.group(1).strip()} v. {title_match.group(2).strip()}"
        if title_match
        else None
    )
    petitioner = record.get("petitioner")
    respondent = record.get("respondent")
    if (
        isinstance(petitioner, str)
        and petitioner.strip()
        and isinstance(respondent, str)
        and respondent.strip()
    ):
        case_name = f"{petitioner.strip()} v. {respondent.strip()}"
    else:
        case_name = case_name_from_html

    source_title = record.get("title")
    title = source_title.strip() if isinstance(source_title, str) and source_title.strip() else case_name

    date_match = re.search(r"Decision Date\s*:\s*([^|]+)", text, re.IGNORECASE)
    case_number_match = re.search(
        r"Case No\s*:\s*([^|]+)", text, re.IGNORECASE
    )
    citation_match = re.fullmatch(
        r"(\d{4})INSC(\d+)", str(record.get("nc_display", ""))
    )
    year = record.get("citation_year") or record.get("year")
    if not isinstance(year, (int, str)) or not str(year).isdigit():
        raise ValueError("Dataset metadata is missing a valid citation year.")

    decision_date = record.get("decision_date")
    pdf_key = f"data/pdf/year={year}/english/{filename}"
    metadata_key = f"metadata/json/year={year}/{metadata_path.name}"
    return {
        "title": title or Path(filename).stem,
        "source": "Indian Supreme Court Judgments dataset (eCourts)",
        "source_url": f"{DATASET_BUCKET_URL}/{pdf_key}",
        "jurisdiction": "India",
        "court": "Supreme Court of India",
        "case_name": case_name,
        "case_number": record.get("case_number") or (
            case_number_match.group(1).strip() if case_number_match else None
        ),
        "citation": (
            f"{citation_match.group(1)} INSC {citation_match.group(2)}"
            if citation_match
            else None
        ),
        "date": decision_date or (date_match.group(1).strip() if date_match else None),
        "language": "English",
        "document_type": "judgment",
        "dataset_attribution": DATASET_ATTRIBUTION,
        "dataset_metadata_url": f"{DATASET_BUCKET_URL}/{metadata_key}",
        "dataset_metadata": record,
    }


def stable_document_id(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if len(relative.parts) == 1:
        return relative.stem
    suffix = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:12]
    return f"{relative.stem}__{suffix}"


async def process_file(
    path: Path,
    root: Path | None = None,
    document_id: str | None = None,
    metadata_dir: Path | None = None,
) -> dict:
    root = root or path.parent
    try:
        source_metadata = None
        if metadata_dir is not None:
            metadata_stem = path.stem[:-3] if path.stem.endswith("_EN") else path.stem
            metadata_path = metadata_dir / f"{metadata_stem}.json"
            if not metadata_path.is_file():
                return {
                    "filename": path.name,
                    "error": "The matching dataset metadata file was not found.",
                }
            source_metadata = _dataset_metadata(metadata_path, path.name)
        processed = await DocumentProcessingService().process(
            path.name,
            path.read_bytes(),
            include_all_chunks=True,
            document_id=document_id or stable_document_id(path, root),
            source="raw_file",
            metadata_overrides=source_metadata,
        )
    except OSError as exc:
        logger.warning("Unable to read source document %s (%s)", path.name, type(exc).__name__)
        return {"filename": path.name, "error": "The file could not be read."}
    except LoaderError as exc:
        logger.warning("Unable to process source document %s: %s", path.name, exc)
        return {"filename": path.name, "error": str(exc)}
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Unable to read metadata for source document %s: %s", path.name, exc)
        return {"filename": path.name, "error": "The dataset metadata is invalid or unreadable."}

    return processed.model_dump(mode="json")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Process raw documents into JSONL for indexing")
    parser.add_argument("input", nargs="?", default="data/raw/sample")
    parser.add_argument("--output", default="data/processed/sample.jsonl")
    parser.add_argument(
        "--metadata-dir",
        help="Optional directory of JSON metadata sidecars matching source documents.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = [input_path] if input_path.is_file() else [
        p for p in input_path.rglob("*") if p.is_file()
    ]
    print(f"Processing {len(files)} file(s) from {input_path} -> {output_path}\n")
    root = input_path.parent if input_path.is_file() else input_path
    identifiers: dict[Path, str] = {}
    candidates: dict[str, list[Path]] = {}
    for path in files:
        candidates.setdefault(stable_document_id(path, root), []).append(path)
    for candidate, matching_paths in candidates.items():
        for path in matching_paths:
            if len(matching_paths) == 1:
                identifiers[path] = candidate
            else:
                relative = path.relative_to(root).as_posix()
                suffix = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
                identifiers[path] = f"{path.stem}__{suffix}"

    results = []
    errors = []

    with open(output_path, "w", encoding="utf-8") as out:
        for path in sorted(files):
            result = await process_file(path, root, identifiers[path], metadata_dir)
            out.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
            if "error" in result:
                errors.append(result)
                print(f"  [ERR] {path.name}: {result['error']}")
            else:
                results.append(result)
                chunks = result["chunk_count"]
                print(f"  [OK]  {path.name}: {chunks} chunk(s)")

    print(f"\nDocument ingestion complete")
    print(f"  Documents processed : {len(results)}")
    print(f"  Errors              : {len(errors)}")
    print(f"  Total chunks        : {sum(r['chunk_count'] for r in results)}")
    print(f"  Output              : {output_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(asyncio.run(main()))
