"""PDF ingestion module."""

from manuals_lib.ingest.models import BoundingBox, PageContent, TextBlock
from manuals_lib.ingest.normalizer import NormalizedPage, normalize_pages, normalize_text
from manuals_lib.ingest.ocr_extractor import extract_text_with_ocr, should_use_ocr
from manuals_lib.ingest.pdf_extractor import extract_pdf, extract_pdf_blocks

__all__ = [
    "BoundingBox",
    "PageContent",
    "TextBlock",
    "extract_pdf",
    "extract_pdf_blocks",
    "NormalizedPage",
    "normalize_pages",
    "normalize_text",
    "extract_text_with_ocr",
    "should_use_ocr",
]
