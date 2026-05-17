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
    
    # Should start at paragraph boundary
    assert overlap == "Second paragraph with more text."


def test_chunk_pages_simple():
    """Test basic chunking of pages."""
    pages = [
        NormalizedPage(source="test.pdf", page_number=1, text="First paragraph.\n\nSecond paragraph."),
        NormalizedPage(source="test.pdf", page_number=2, text="Third paragraph.\n\nFourth paragraph."),
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
