from .base import LoadedDocument, LoadedPage

class TXTLoader:
    def load(self, content: bytes, filename: str) -> LoadedDocument:
        text = content.decode("utf-8-sig", errors="replace")
        return LoadedDocument(filename, "text/plain", [LoadedPage(1, text)])
