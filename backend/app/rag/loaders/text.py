from .base import LoadedDocument, LoadedPage, LoaderError

class TXTLoader:
    def load(self, content: bytes, filename: str) -> LoadedDocument:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise LoaderError("Malformed text document: expected UTF-8 encoding.") from exc
        return LoadedDocument(filename, "text/plain", [LoadedPage(1, text)])
