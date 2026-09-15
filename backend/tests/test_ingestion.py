import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from pypdf import PdfWriter
from app.rag.loaders import JSONLoader, PDFLoader, TXTLoader
from app.rag.processors import clean_text, process_pages
from app.rag.legal_chunker import LegalChunker
from app.rag.loaders.base import LoadedPage


def test_txt_and_json_loaders():
    assert TXTLoader().load(b"Section 1.\nText", "a.txt").pages[0].text.startswith("Section")
    loaded = JSONLoader().load(b'{"title":"Synthetic","content":"Section 1."}', "a.json")
    assert '"title": "Synthetic"' in loaded.pages[0].text


def test_pdf_loader_preserves_page_numbers():
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buffer = io.BytesIO()
    writer.write(buffer)
    loaded = PDFLoader().load(buffer.getvalue(), "blank.pdf")
    assert loaded.pages[0].page_number == 1


def test_cleaning_normalizes_and_removes_repeated_headers():
    pages = process_pages([LoadedPage(1, "HEADER\nA   B"), LoadedPage(2, "HEADER\nC\r\nD")])
    assert pages[0].text == "A B"
    assert "HEADER" not in pages[1].text
    assert "\ufffd" not in clean_text("bad\ufffdtext")


def test_chunking_preserves_overlap_sections_and_traceability():
    pages = [LoadedPage(1, "Chapter 1\nSection 1. " + ("legal text " * 20)), LoadedPage(2, "Section 2. " + ("more text " * 20))]
    chunks = LegalChunker(chunk_size=100, chunk_overlap=20).chunk(pages)
    assert len(chunks) > 1
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert chunks[0].section
