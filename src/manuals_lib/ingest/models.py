"""Data models for document ingestion."""

from dataclasses import dataclass


@dataclass
class PageContent:
    """Represents extracted content from a single page of a document.
    
    Attributes:
        source: The filename of the source document
        page_number: The page number (1-indexed)
        text: The extracted text content from the page
    """
    source: str
    page_number: int
    text: str
