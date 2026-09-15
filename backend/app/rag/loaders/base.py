from dataclasses import dataclass, field
from typing import Any, Protocol

@dataclass
class LoadedPage:
    page_number: int
    text: str

@dataclass
class LoadedDocument:
    filename: str
    content_type: str | None
    pages: list[LoadedPage]
    metadata: dict[str, Any] = field(default_factory=dict)

class LoaderError(ValueError):
    """A user-correctable document format or parsing error."""

class BaseLoader(Protocol):
    def load(self, content: bytes, filename: str) -> LoadedDocument: ...
