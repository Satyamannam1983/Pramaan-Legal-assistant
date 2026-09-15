from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum


class SourceType(StrEnum):
    ACT = "act"
    CENTRAL_LAW = "central_law"
    STATE_LAW = "state_law"
    RULE = "rule"
    SUPREME_COURT_JUDGMENT = "supreme_court_judgment"
    HIGH_COURT_JUDGMENT = "high_court_judgment"
    NOTIFICATION = "notification"
    GOVERNMENT_DOCUMENT = "government_document"


@dataclass
class LegalSource:
    title: str
    source_type: SourceType
    court: str | None = None
    case_name: str | None = None
    citation: str | None = None
    date: datetime | None = None
    jurisdiction: str | None = None
    section: str | None = None
    document_url: str | None = None
    content: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class Document:
    filename: str
    content_type: str | None
    size: int
    status: str = "uploaded"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
