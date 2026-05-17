"""PDF ingestion module."""

from manuals_lib.ingest.models import PageContent
from manuals_lib.ingest.pdf_extractor import extract_pdf

__all__ = ["PageContent", "extract_pdf"]
