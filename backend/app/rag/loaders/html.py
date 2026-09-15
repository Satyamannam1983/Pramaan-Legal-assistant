from html.parser import HTMLParser
from .base import LoadedDocument, LoadedPage

class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.hidden = 0
    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}: self.hidden += 1
    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"}: self.hidden = max(0, self.hidden - 1)
    def handle_data(self, data):
        if not self.hidden: self.parts.append(data)

class HTMLLoader:
    def load(self, content: bytes, filename: str) -> LoadedDocument:
        parser = _TextParser()
        parser.feed(content.decode("utf-8-sig", errors="replace"))
        return LoadedDocument(filename, "text/html", [LoadedPage(1, "\n".join(parser.parts))])
