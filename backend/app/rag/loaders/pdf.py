from .base import LoadedDocument, LoadedPage, LoaderError

class PDFLoader:
    def load(self, content: bytes, filename: str) -> LoadedDocument:
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(content))
            pages = [LoadedPage(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
            return LoadedDocument(filename, "application/pdf", pages)
        except ImportError as exc:
            raise LoaderError("PDF support is unavailable; install pypdf") from exc
        except Exception as exc:
            raise LoaderError(f"Malformed PDF document: {exc}") from exc
