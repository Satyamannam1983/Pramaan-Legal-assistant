from typing import Protocol


class DocumentLoader(Protocol):
    async def load(self, content: bytes, filename: str) -> str: ...


class PlaceholderDocumentLoader:
    async def load(self, content: bytes, filename: str) -> str:
        return content.decode("utf-8", errors="replace")
