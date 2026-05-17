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


@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates.
    
    Attributes:
        x0: Left x coordinate
        y0: Top y coordinate
        x1: Right x coordinate
        y1: Bottom y coordinate
    """
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class TextBlock:
    """Represents a text block extracted from a PDF page.
    
    Attributes:
        source: The filename of the source document
        page_number: The page number (1-indexed)
        block_number: The block number within the page (0-indexed)
        bbox: Bounding box coordinates of the block
        text: The extracted text content from the block
    """
    source: str
    page_number: int
    block_number: int
    bbox: BoundingBox
    text: str


@dataclass
class Chunk:
    """Represents a text chunk for retrieval.
    
    Attributes:
        chunk_id: Unique identifier for the chunk
        source: The filename of the source document
        page_start: First page number in the chunk (1-indexed)
        page_end: Last page number in the chunk (1-indexed)
        text: The chunk text content
        char_count: Number of characters in the chunk
    """
    chunk_id: str
    source: str
    page_start: int
    page_end: int
    text: str
    char_count: int
