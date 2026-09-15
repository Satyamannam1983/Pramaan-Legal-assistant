import re
from .loaders.base import LoadedPage

def clean_text(text: str) -> str:
    text = text.replace("\ufffd", "").replace("\u00ad", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"(?m)^\s*[-_]{3,}\s*$", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()

def process_pages(pages: list[LoadedPage]) -> list[LoadedPage]:
    cleaned = [LoadedPage(p.page_number, clean_text(p.text)) for p in pages]
    lines = [[line for line in p.text.splitlines() if line.strip()] for p in cleaned]
    candidates = [group[0] for group in lines if group]
    candidates += [group[-1] for group in lines if group]
    repeated = {line for line in candidates if sum(line in group for group in lines) >= 2}
    return [LoadedPage(p.page_number, "\n".join(line for line in p.text.splitlines() if line.strip() not in repeated).strip()) for p in cleaned]

def detect_section(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line and (re.match(r"^(section|chapter|part|article)\b", line, re.I) or re.match(r"^\d+[.)]\s+\S+", line)):
            return line
    return None
