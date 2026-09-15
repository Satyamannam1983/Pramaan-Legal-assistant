import json
from .base import LoadedDocument, LoadedPage, LoaderError

class JSONLoader:
    def load(self, content: bytes, filename: str) -> LoadedDocument:
        try:
            value = json.loads(content.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LoaderError(f"Malformed JSON document: {exc}") from exc
        text = json.dumps(value, ensure_ascii=False, indent=2)
        metadata = value if isinstance(value, dict) else {}
        return LoadedDocument(filename, "application/json", [LoadedPage(1, text)], metadata=metadata)
