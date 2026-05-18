"""Data models for document ingestion."""

from dataclasses import dataclass


@dataclass
class PageContent:
    """Represents extracted content from a single page of a document.
    
    Attributes:
        source: The filename of the source document
        page_number: The page number (1-indexed)
        text: The extracted text content from the page
        extraction_method: Method used to extract text (pymupdf or ocr)
        ocr_engine: OCR engine used if extraction_method is ocr
        ocr_trigger_reason: Reason OCR was triggered
    """
    source: str
    page_number: int
    text: str
    extraction_method: str = "pymupdf"
    ocr_engine: str | None = None
    ocr_trigger_reason: str | None = None


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
class TableData:
    """Represents extracted table data.
    
    Attributes:
        table_id: Unique identifier for this table
        source: The filename of the source document
        page_number: The page number where table appears (1-indexed)
        bbox: Bounding box coordinates (optional)
        extraction_method: Method used to extract table
        headers: Inferred column headers (optional)
        rows: List of rows, each row is a list of cell values
        table_title: Inferred table title from nearby text (optional)
    """
    table_id: str
    source: str
    page_number: int
    bbox: BoundingBox | None
    extraction_method: str
    headers: list[str] | None
    rows: list[list[str]]
    table_title: str | None = None


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
        chunk_type: Type of chunk (content, toc, table, etc.)
        extraction_method: Method used to extract source text
        table_id: Reference to source table if chunk_type is table/table_row
        section_title: Section heading if available
        section_path: Hierarchical path of section titles (e.g., ["Chapter 1", "Section 1.1"])
    """
    chunk_id: str
    source: str
    page_start: int
    page_end: int
    text: str
    char_count: int
    chunk_type: str = "content"
    extraction_method: str = "pymupdf"
    table_id: str | None = None
    section_title: str | None = None
    section_path: list[str] | None = None
