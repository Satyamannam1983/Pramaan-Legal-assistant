from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def build_report(
    *,
    status: str,
    dataset_size: int,
    metrics: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    """Compatibility helper; omitted metrics remain absent rather than zero-filled."""
    return {
        "schema_version": 1,
        "status": status,
        "dataset_size": dataset_size,
        "evaluations": metrics or {},
        "limitations": limitations or [],
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Legal RAG Evaluation",
        "",
        f"Status: **{report.get('status', 'NOT RUN')}**",
        f"Run date: `{report.get('timestamp', 'not recorded')}`",
        "",
        "## Datasets",
    ]
    for stage, dataset in report.get("datasets", {}).items():
        lines.extend([
            f"- **{stage}**: {dataset.get('name')} v{dataset.get('version')}, "
            f"{dataset.get('example_count', 0)} examples, SHA-256 "
            f"`{dataset.get('sha256', 'unavailable')}`",
            f"  - Provenance: {dataset.get('provenance') or 'not supplied'}",
        ])
    lines.extend(["", "## Configuration", ""])
    for key, value in report.get("configuration", {}).items():
        lines.append(f"- `{key}`: {value}")

    lines.extend(["", "## Evaluations", ""])
    for name, stage in report.get("evaluations", {}).items():
        lines.extend([
            f"### {name.replace('_', ' ').title()} - {stage.get('status', 'NOT RUN')}",
            "",
        ])
        if stage.get("reason"):
            lines.extend([str(stage["reason"]), ""])
        metrics = stage.get("metrics", {})
        category_metrics = stage.get("category_metrics", {})
        operational = stage.get("operational_metrics", {})
        for section_name, values in (
            ("Metrics", metrics),
            ("Category metrics", category_metrics),
            ("Operational metrics", operational),
        ):
            if values:
                lines.extend([f"**{section_name}**", ""])
                for key, value in values.items():
                    if isinstance(value, (dict, list)):
                        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
                    else:
                        rendered = "not measured" if value is None else str(value)
                    lines.append(f"- `{key}`: {rendered}")
                lines.append("")
        diagnostics = stage.get("diagnostics", {})
        if diagnostics:
            lines.extend(["**Diagnostics**", ""])
            for key, value in diagnostics.items():
                lines.append(f"- `{key}`: {json.dumps(value, ensure_ascii=False, sort_keys=True)}")
            lines.append("")
        quality = stage.get("quality_evaluation")
        if quality:
            lines.extend([
                f"**Answer quality: {quality.get('status', 'NOT RUN')}**",
                str(quality.get("reason") or ""),
                "",
            ])
        for limitation in stage.get("limitations", []):
            lines.append(f"- {limitation}")
        lines.append("")

    limitations = report.get("limitations", [])
    if limitations:
        lines.extend(["## Run limitations", ""])
        lines.extend(f"- {limitation}" for limitation in limitations)
    return "\n".join(lines).rstrip() + "\n"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_report(report: dict[str, Any], output_dir: str | Path) -> None:
    """Write JSON and Markdown reports without manufacturing absent scores."""
    destination = Path(output_dir)
    json_content = json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    markdown_content = _markdown(report)
    _atomic_write(destination / "evaluation_report.json", json_content)
    _atomic_write(destination / "evaluation_report.md", markdown_content)
