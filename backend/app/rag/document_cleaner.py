import re


class DocumentCleaner:
    def clean(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()
