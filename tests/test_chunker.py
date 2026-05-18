"""Tests for text chunking."""

from manuals_lib.ingest.chunker import (
    ChunkingConfig,
    chunk_pages,
    create_chunk_id,
    get_overlap_text,
    split_into_paragraphs,
)
from manuals_lib.ingest.normalizer import NormalizedPage


def test_split_into_paragraphs():
    """Test paragraph splitting."""
    text = "First paragraph.\n\nSecond paragraph.\n\n\nThird paragraph."
    paragraphs = split_into_paragraphs(text)
    
    assert len(paragraphs) == 3
    assert paragraphs[0] == "First paragraph."
    assert paragraphs[1] == "Second paragraph."
    assert paragraphs[2] == "Third paragraph."


def test_split_into_paragraphs_single():
    """Test splitting text with no paragraph breaks."""
    text = "Single paragraph with no breaks."
    paragraphs = split_into_paragraphs(text)
    
    assert len(paragraphs) == 1
    assert paragraphs[0] == text


def test_create_chunk_id():
    """Test chunk ID creation."""
    chunk_id = create_chunk_id("manual.pdf", 0)
    assert chunk_id == "manual_chunk_0000"
    
    chunk_id = create_chunk_id("manual.pdf", 42)
    assert chunk_id == "manual_chunk_0042"


def test_get_overlap_text():
    """Test overlap text extraction."""
    text = "First sentence. Second sentence. Third sentence."
    overlap = get_overlap_text(text, 20)
    
    # Should start at sentence boundary
    assert "Third sentence." in overlap
    assert len(overlap) <= 20


def test_get_overlap_text_paragraph():
    """Test overlap with paragraph boundary."""
    text = "First paragraph.\n\nSecond paragraph with more text."
    overlap = get_overlap_text(text, 30)
    
    # Should start at paragraph boundary (within the last 30 chars)
    assert overlap == "paragraph with more text."


def test_chunk_pages_simple():
    """Test basic chunking of pages."""
    pages = [
        NormalizedPage(
            source="test.pdf",
            page_number=1,
            text="First paragraph.\n\nSecond paragraph."
        ),
        NormalizedPage(
            source="test.pdf",
            page_number=2,
            text="Third paragraph.\n\nFourth paragraph."
        ),
    ]
    
    config = ChunkingConfig(target_size=50, max_size=100, overlap_size=10)
    chunks = chunk_pages(pages, config)
    
    assert len(chunks) > 0
    assert all(chunk.source == "test.pdf" for chunk in chunks)
    assert all(chunk.page_start >= 1 for chunk in chunks)
    assert all(chunk.char_count <= config.max_size for chunk in chunks)


def test_chunk_pages_preserves_page_numbers():
    """Test that chunks preserve page number metadata."""
    pages = [
        NormalizedPage(source="test.pdf", page_number=1, text="Page 1 content."),
        NormalizedPage(source="test.pdf", page_number=2, text="Page 2 content."),
    ]
    
    chunks = chunk_pages(pages)
    
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.page_start >= 1
        assert chunk.page_end >= chunk.page_start


def test_chunk_pages_respects_max_size():
    """Test that chunks don't exceed max size."""
    # Create a page with long paragraphs
    long_para = "A" * 1000
    pages = [
        NormalizedPage(
            source="test.pdf",
            page_number=1,
            text=f"{long_para}\n\n{long_para}\n\n{long_para}"
        ),
    ]
    
    config = ChunkingConfig(target_size=1500, max_size=2000, overlap_size=100)
    chunks = chunk_pages(pages, config)
    
    for chunk in chunks:
        assert chunk.char_count <= config.max_size


def test_chunk_pages_empty_input():
    """Test chunking with empty input."""
    chunks = chunk_pages([])
    assert chunks == []


def test_chunk_pages_single_short_page():
    """Test chunking a single short page."""
    pages = [
        NormalizedPage(source="test.pdf", page_number=1, text="Short content."),
    ]
    
    chunks = chunk_pages(pages)
    
    assert len(chunks) == 1
    assert chunks[0].text == "Short content."
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1


def test_chunk_pages_preserves_extraction_method():
    """Test that chunking preserves extraction method metadata."""
    pages = [
        NormalizedPage(
            source="test.pdf",
            page_number=1,
            text="First page content. " * 100,
            extraction_method="ocr",
            ocr_engine="tesseract",
            ocr_trigger_reason="low_text_content",
        ),
        NormalizedPage(
            source="test.pdf",
            page_number=2,
            text="Second page content. " * 100,
            extraction_method="pymupdf",
        ),
    ]
    
    chunks = chunk_pages(pages)
    
    assert len(chunks) > 0
    # First chunk should have OCR metadata
    assert chunks[0].extraction_method == "ocr"


def test_chunk_pages_detects_chunk_type():
    """Test that chunking detects chunk types."""
    # Make content large enough to force separate chunks
    content_text = "This is regular content about the product specifications. " * 50
    
    pages = [
        NormalizedPage(
            source="test.pdf",
            page_number=1,
            text="Table of Contents\n\nChapter 1 ........ 5\nChapter 2 ........ 10",
        ),
        NormalizedPage(
            source="test.pdf",
            page_number=2,
            text=content_text,
        ),
    ]
    
    chunks = chunk_pages(pages)
    
    assert len(chunks) >= 2
    # First chunk should be detected as TOC
    assert chunks[0].chunk_type == "toc"
    # At least one chunk should be content
    assert any(c.chunk_type == "content" for c in chunks)


def test_chunk_tables():
    """Test chunking tables into rows."""
    from manuals_lib.ingest.chunker import chunk_tables
    from manuals_lib.ingest.models import TableData
    
    table = TableData(
        table_id="test_table_1",
        source="test.pdf",
        page_number=5,
        bbox=None,
        extraction_method="pymupdf_find_tables",
        headers=["Item", "Value", "Unit"],
        rows=[
            ["Temperature", "25", "°C"],
            ["Pressure", "100", "kPa"],
            ["Flow Rate", "50", "L/min"],
        ],
    )
    
    chunks = chunk_tables([table])
    
    assert len(chunks) == 3
    assert all(c.chunk_type == "table_row" for c in chunks)
    assert all(c.table_id == "test_table_1" for c in chunks)
    assert all(c.page_start == 5 for c in chunks)
    
    # Check that row data is flattened
    assert "Item: Temperature" in chunks[0].text
    assert "Value: 25" in chunks[0].text
    assert "Unit: °C" in chunks[0].text


def test_chunk_tables_with_title():
    """Test chunking tables with title row."""
    from manuals_lib.ingest.chunker import chunk_tables
    from manuals_lib.ingest.models import TableData
    
    table = TableData(
        table_id="test_table_2",
        source="test.pdf",
        page_number=10,
        bbox=None,
        extraction_method="pymupdf_find_tables",
        headers=None,
        rows=[
            ["Technical Specifications", "", ""],  # Title row
            ["Length", "5.2", "m"],
            ["Width", "2.1", "m"],
        ],
    )
    
    chunks = chunk_tables([table])
    
    # Should skip title row and create chunks for data rows
    assert len(chunks) == 2
    assert "Table: Technical Specifications" in chunks[0].text


def test_chunk_tables_with_metadata_title():
    """Test chunking tables with title from metadata."""
    from manuals_lib.ingest.chunker import chunk_tables
    from manuals_lib.ingest.models import TableData
    
    table = TableData(
        table_id="test_table_4",
        source="test.pdf",
        page_number=20,
        bbox=None,
        extraction_method="pymupdf_find_tables",
        headers=["Parameter", "Value"],
        rows=[
            ["Speed", "100 km/h"],
            ["Range", "500 km"],
        ],
        table_title="Performance Metrics",
    )
    
    chunks = chunk_tables([table])
    
    assert len(chunks) == 2
    assert "Table: Performance Metrics" in chunks[0].text
    assert chunks[0].section_title == "Performance Metrics"
    assert chunks[1].section_title == "Performance Metrics"


def test_chunk_tables_skips_empty_rows():
    """Test that empty table rows are skipped."""
    from manuals_lib.ingest.chunker import chunk_tables
    from manuals_lib.ingest.models import TableData
    
    table = TableData(
        table_id="test_table_3",
        source="test.pdf",
        page_number=15,
        bbox=None,
        extraction_method="pymupdf_find_tables",
        headers=["A", "B"],
        rows=[
            ["Value1", "Value2"],
            ["", ""],  # Empty row
            ["Value3", "Value4"],
        ],
    )
    
    chunks = chunk_tables([table])
    
    # Should skip empty row
    assert len(chunks) == 2


def test_chunk_pages_has_overlap():
    """Test that adjacent chunks have overlap."""
    # Create content that will span multiple chunks
    paragraphs = [f"Paragraph {i} with some content." for i in range(20)]
    text = "\n\n".join(paragraphs)
    
    pages = [
        NormalizedPage(source="test.pdf", page_number=1, text=text),
    ]
    
    config = ChunkingConfig(target_size=200, max_size=300, overlap_size=50)
    chunks = chunk_pages(pages, config)
    
    # Should have multiple chunks
    assert len(chunks) > 1
    
    # Check for overlap between adjacent chunks
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i + 1]
        
        # Next chunk should start with some text from current chunk
        current_end = current_chunk.text[-100:]
        next_start = next_chunk.text[:100]
        
        # There should be some common words (overlap)
        current_words = set(current_end.split())
        next_words = set(next_start.split())
        common_words = current_words & next_words
        
        # Should have at least some overlap
        assert len(common_words) > 0
