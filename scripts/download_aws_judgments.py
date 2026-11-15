"""Resume a bounded download of public Indian court judgment datasets.

Downloads Supreme Court English archives and metadata for the selected years,
plus the 2023 Uttarakhand High Court archive. No TLS verification is disabled.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "raw" / "aws_open_data"
MANIFEST_PATH = DATA_ROOT / "download_manifest.json"
RESERVED_BYTES = 5 * 1024**3
BUCKETS = {
    "supreme_court": "indian-supreme-court-judgments",
    "high_courts": "indian-high-court-judgments",
}
LOG = logging.getLogger("download_aws_judgments")
OBJECT_LINE = re.compile(
    r"^\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+(\d+)\s+(.+)$"
)


def _write_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, MANIFEST_PATH)


def _record_entry(manifest: dict, entry: dict) -> None:
    for index, existing in enumerate(manifest["objects"]):
        if existing.get("dataset") == entry["dataset"] and existing.get("key") == entry["key"]:
            manifest["objects"][index] = entry
            return
    manifest["objects"].append(entry)


def _list_objects(bucket: str, prefix: str) -> list[dict]:
    result = subprocess.run(
        [
            "aws", "s3", "ls", f"s3://{bucket}/{prefix}",
            "--recursive", "--no-sign-request",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        raise RuntimeError(
            f"Unable to list public S3 prefix {prefix!r} in {bucket} "
            f"(aws exit code {result.returncode})."
        )
    objects = []
    for line in result.stdout.splitlines():
        match = OBJECT_LINE.match(line)
        if match:
            objects.append({"key": match.group(2), "size": int(match.group(1))})
    return objects


def _valid_tar(path: Path) -> bool:
    try:
        with tarfile.open(path, mode="r:") as archive:
            members = archive.getmembers()
            return any(member.isfile() for member in members)
    except (OSError, tarfile.TarError):
        return False


def _local_path(dataset: str, key: str) -> Path:
    return DATA_ROOT / dataset / Path(*key.split("/"))


def _entry(dataset: str, item: dict, status: str, error: str | None = None) -> dict:
    entry = {
        "dataset": dataset,
        "key": item["key"],
        "size_bytes": item["size"],
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if error:
        entry["error"] = error
    return entry


def _sync_group(
    dataset: str,
    bucket: str,
    prefix: str,
    items: list[dict],
    manifest: dict,
) -> bool:
    target_prefix = DATA_ROOT / dataset / Path(*prefix.rstrip("/").split("/"))
    target_prefix.mkdir(parents=True, exist_ok=True)
    missing = []
    excluded = []
    failed = False

    for item in items:
        path = _local_path(dataset, item["key"])
        if not path.is_file():
            missing.append(item)
            continue
        if path.stat().st_size != item["size"]:
            _record_entry(
                manifest,
                _entry(
                    dataset,
                    item,
                    "failed",
                    "Existing file size differs from the S3 object; preserved without overwrite.",
                ),
            )
            excluded.append(item["key"][len(prefix):])
            failed = True
        elif path.suffix.lower() == ".tar" and not _valid_tar(path):
            _record_entry(
                manifest,
                _entry(
                    dataset,
                    item,
                    "failed",
                    "Existing TAR failed integrity validation; preserved without overwrite.",
                ),
            )
            excluded.append(item["key"][len(prefix):])
            failed = True
        else:
            _record_entry(manifest, _entry(dataset, item, "skipped_existing"))
            excluded.append(item["key"][len(prefix):])

    if missing:
        required_bytes = sum(item["size"] for item in missing)
        if shutil.disk_usage(DATA_ROOT.parent).free - required_bytes < RESERVED_BYTES:
            for item in missing:
                _record_entry(
                    manifest,
                    _entry(
                        dataset,
                        item,
                        "pending",
                        "Download deferred to preserve the 5 GiB free-space reserve.",
                    ),
                )
            _write_manifest(manifest)
            return False

        command = [
            "aws", "s3", "sync", f"s3://{bucket}/{prefix}", str(target_prefix),
            "--no-sign-request", "--only-show-errors",
            "--exclude", "*",
        ]
        if prefix.startswith("metadata/"):
            command.extend(["--include", "*.json"])
        else:
            command.extend(["--include", "*.tar", "--include", "*.index.json"])
        for key in excluded:
            command.extend(["--exclude", key])
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode:
            if "certificate verify failed" in result.stderr.lower():
                sync_error = "TLS certificate validation failed; TLS verification remains enabled."
            else:
                sync_error = f"AWS CLI sync reported exit code {result.returncode}."
        else:
            sync_error = ""

        for item in missing:
            path = _local_path(dataset, item["key"])
            if (
                path.is_file()
                and path.stat().st_size == item["size"]
                and (path.suffix.lower() != ".tar" or _valid_tar(path))
            ):
                _record_entry(manifest, _entry(dataset, item, "downloaded"))
            else:
                _record_entry(
                    manifest,
                    _entry(
                        dataset,
                        item,
                        "failed",
                        sync_error or "AWS CLI sync did not produce a verified file.",
                    ),
                )
                failed = True

    _write_manifest(manifest)
    LOG.info(
        "Synced %s: objects=%d missing=%d failed=%s",
        prefix,
        len(items),
        len(missing),
        failed,
    )
    return not failed


def _parse_years(value: str) -> list[int]:
    years = []
    for item in value.split(","):
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError("Year ranges must be in ascending order.")
            years.extend(range(start, end + 1))
        else:
            years.append(int(item))
    if not years or any(year < 1950 or year > 2026 for year in years):
        raise ValueError("Years must be between 1950 and 2026.")
    return sorted(set(years))


def _metadata_record_counts() -> dict[str, int]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to count metadata rows without loading Parquet contents."
        ) from exc

    counts = {}
    for dataset in ("supreme_court", "high_courts"):
        parquet_root = DATA_ROOT / dataset / "metadata" / "parquet"
        counts[dataset] = sum(
            parquet.ParquetFile(path).metadata.num_rows
            for path in parquet_root.rglob("*.parquet")
        ) if parquet_root.exists() else 0
    return counts


def _archive_coverage(objects: list[dict]) -> dict:
    supreme_years = set()
    high_courts = set()
    for item in objects:
        if not item.get("key", "").lower().endswith(".tar"):
            continue
        path = _local_path(item["dataset"], item["key"])
        if not path.is_file() or path.stat().st_size != item.get("size_bytes"):
            continue
        if item["dataset"] == "supreme_court":
            match = re.search(r"data/tar/year=(\d{4})/english/", item["key"])
            if match:
                supreme_years.add(int(match.group(1)))
        elif item["dataset"] == "high_courts":
            match = re.search(r"data/tar/year=(\d{4})/court=([^/]+)/", item["key"])
            if match:
                high_courts.add((int(match.group(1)), match.group(2)))
    return {
        "supreme_court_english_years": sorted(supreme_years),
        "high_court_year_courts": [
            {"year": year, "court_id": court_id}
            for year, court_id in sorted(high_courts)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supreme-years", default="2010-2026")
    parser.add_argument("--high-court-year", type=int, default=2023)
    parser.add_argument("--high-court-id", default="5_15")
    parser.add_argument("--skip-metadata", action="store_true")
    args = parser.parse_args()
    years = _parse_years(args.supreme_years)
    if args.high_court_year < 1950 or args.high_court_year > 2026:
        parser.error("--high-court-year must be between 1950 and 2026.")
    if not re.fullmatch(r"\d+_\d+", args.high_court_id):
        parser.error("--high-court-id must use the dataset's numeric court code format.")

    manifest = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "objects": [],
        "selection": {
            "supreme_court_english_years": years,
            "high_court_year": args.high_court_year,
            "high_court_id": args.high_court_id,
        },
    }
    if MANIFEST_PATH.is_file():
        try:
            previous = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            manifest["objects"] = [
                item for item in previous.get("objects", [])
                if item.get("status") in {"downloaded", "skipped_existing"}
            ]
            previous_selection = previous.get("selection", {})
            previous_years = previous_selection.get(
                "supreme_court_english_years", []
            )
            previous_courts = previous_selection.get("high_court_selections", [])
            if not previous_courts and previous_selection.get("high_court_year"):
                previous_courts = [{
                    "year": previous_selection["high_court_year"],
                    "court_id": previous_selection.get("high_court_id", ""),
                }]
            manifest["selection"]["supreme_court_english_years"] = sorted(
                set(years) | {
                    int(year) for year in previous_years
                    if str(year).isdigit()
                }
            )
            manifest["selection"]["high_court_selections"] = sorted(
                {
                    (int(item["year"]), str(item["court_id"]))
                    for item in previous_courts
                    if isinstance(item, dict)
                    and str(item.get("year", "")).isdigit()
                    and item.get("court_id")
                } | {(args.high_court_year, args.high_court_id)}
            )
        except (OSError, json.JSONDecodeError):
            LOG.warning("Previous download manifest is unreadable; starting a fresh manifest.")
    _write_manifest(manifest)

    groups = []
    for year in years:
        archive_prefix = f"data/tar/year={year}/english/"
        archive_items = _list_objects(BUCKETS["supreme_court"], archive_prefix)
        groups.append(
            ("supreme_court", BUCKETS["supreme_court"], archive_prefix, archive_items)
        )
        if not args.skip_metadata:
            metadata_prefix = f"metadata/json/year={year}/"
            metadata_items = [
                item for item in _list_objects(
                    BUCKETS["supreme_court"], metadata_prefix
                )
                if item["key"].lower().endswith(".json")
            ]
            groups.append(
                (
                    "supreme_court",
                    BUCKETS["supreme_court"],
                    metadata_prefix,
                    metadata_items,
                )
            )

    high_court_prefix = (
        f"data/tar/year={args.high_court_year}/court={args.high_court_id}/"
    )
    high_court_items = _list_objects(BUCKETS["high_courts"], high_court_prefix)
    groups.append(
        ("high_courts", BUCKETS["high_courts"], high_court_prefix, high_court_items)
    )

    selections = [
        (dataset, item)
        for dataset, _, _, items in groups
        for item in items
    ]
    expected_total = sum(item["size"] for _, item in selections)
    missing_total = sum(
        item["size"]
        for dataset, item in selections
        if not _local_path(dataset, item["key"]).is_file()
    )
    LOG.info(
        "Selected %d S3 objects (%.2f GiB total; %.2f GiB missing; %.2f GiB free).",
        len(selections),
        expected_total / 1024**3,
        missing_total / 1024**3,
        shutil.disk_usage(DATA_ROOT.parent).free / 1024**3,
    )
    failed = False
    for dataset, bucket, prefix, items in groups:
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        if not _sync_group(dataset, bucket, prefix, items, manifest):
            failed = True
            if shutil.disk_usage(DATA_ROOT.parent).free <= RESERVED_BYTES:
                break

    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "incomplete" if failed else "complete"
    manifest["metadata_records_available"] = _metadata_record_counts()
    coverage = _archive_coverage(manifest["objects"])
    manifest["archive_coverage"] = coverage
    manifest["selection"]["supreme_court_english_years"] = (
        coverage["supreme_court_english_years"]
    )
    manifest["selection"]["high_court_selections"] = (
        coverage["high_court_year_courts"]
    )
    _write_manifest(manifest)
    downloaded = sum(item.get("status") == "downloaded" for item in manifest["objects"])
    skipped = sum(item.get("status") == "skipped_existing" for item in manifest["objects"])
    failures = sum(item.get("status") in {"failed", "pending"} for item in manifest["objects"])
    LOG.info(
        "Download run finished: downloaded=%d existing=%d failed_or_pending=%d.",
        downloaded, skipped, failures,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
