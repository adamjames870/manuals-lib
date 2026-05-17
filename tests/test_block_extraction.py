"""Tests for block-level PDF extraction."""

from pathlib import Path

import pytest

from manuals_lib.ingest import extract_pdf_blocks
from manuals_lib.ingest.models import BoundingBox, TextBlock


def test_extract_pdf_blocks_with_real_file():
    """Test extracting blocks from a real PDF file."""
    pdf_path = Path("tests/data/sample.pdf")
    
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")
    
    blocks = extract_pdf_blocks(pdf_path)
    
    # Basic assertions
    assert isinstance(blocks, list)
    assert len(blocks) > 0
    
    # Check first block structure
    first_block = blocks[0]
    assert isinstance(first_block, TextBlock)
    assert first_block.source == "sample.pdf"
    assert first_block.page_number >= 1
    assert first_block.block_number >= 0
    assert isinstance(first_block.bbox, BoundingBox)
    assert isinstance(first_block.text, str)
    assert len(first_block.text) > 0


def test_extract_pdf_blocks_bounding_boxes():
    """Test that bounding boxes have valid coordinates."""
    pdf_path = Path("tests/data/sample.pdf")
    
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")
    
    blocks = extract_pdf_blocks(pdf_path)
    
    for block in blocks:
        # Check bbox coordinates are valid
        assert block.bbox.x0 >= 0
        assert block.bbox.y0 >= 0
        assert block.bbox.x1 > block.bbox.x0
        assert block.bbox.y1 > block.bbox.y0


def test_extract_pdf_blocks_page_numbers():
    """Test that blocks have correct page numbers."""
    pdf_path = Path("tests/data/sample.pdf")
    
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")
    
    blocks = extract_pdf_blocks(pdf_path)
    
    # All blocks should have page numbers >= 1
    for block in blocks:
        assert block.page_number >= 1


def test_extract_pdf_blocks_file_not_found():
    """Test that FileNotFoundError is raised for non-existent files."""
    pdf_path = Path("tests/data/nonexistent.pdf")
    
    with pytest.raises(FileNotFoundError) as exc_info:
        extract_pdf_blocks(pdf_path)
    
    assert "PDF file not found" in str(exc_info.value)


def test_extract_pdf_blocks_no_empty_text():
    """Test that extracted blocks don't contain empty text."""
    pdf_path = Path("tests/data/sample.pdf")
    
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")
    
    blocks = extract_pdf_blocks(pdf_path)
    
    # All blocks should have non-empty text
    for block in blocks:
        assert block.text
        assert block.text.strip()
