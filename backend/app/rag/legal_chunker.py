from dataclasses import dataclass
from .loaders.base import LoadedPage
from .processors import detect_section

@dataclass
class Chunk:
    index: int
    text: str
    page_start: int | None
    page_end: int | None
    section: str | None

class LegalChunker:
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 150):
        if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_size must be positive and overlap must be smaller than size")
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap

    def chunk(self, pages: list[LoadedPage]) -> list[Chunk]:
        text = "\n\n".join(p.text for p in pages if p.text)
        if not text: return []
        offsets, cursor = [], 0
        for p in pages:
            start, end = cursor, cursor + len(p.text)
            offsets.append((start, end, p.page_number))
            cursor = end + 2
        chunks, start, index = [], 0, 0
        while start < len(text):
            end = min(len(text), start + self.chunk_size)
            if end < len(text):
                boundary = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
                if boundary > start + self.chunk_size // 2: end = boundary + (2 if text[boundary:boundary+2] == ". " else 0)
            segment = text[start:end].strip()
            if segment:
                covered = [p for a, b, p in offsets if b > start and a < end]
                chunks.append(Chunk(index, segment, min(covered) if covered else None, max(covered) if covered else None, detect_section(segment)))
                index += 1
            if end >= len(text): break
            start = max(start + 1, end - self.chunk_overlap)
        return chunks
