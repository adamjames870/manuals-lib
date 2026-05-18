"""Tests for PDF extraction functionality."""

from pathlib import Path

import pytest

from manuals_lib.ingest import extract_pdf
from manuals_lib.ingest.models import PageContent


def test_extract_pdf_with_real_file():
    """Test extracting text from a real PDF file."""
    pdf_path = Path("tests/data/sample.pdf")
    
    # Skip if test PDF doesn't exist
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")
    
    pages = extract_pdf(pdf_path)
    
    # Basic assertions
    assert isinstance(pages, list)
    assert len(pages) > 0
    
    # Check first page structure
    first_page = pages[0]
    assert isinstance(first_page, PageContent)
    assert first_page.source == "sample.pdf"
    assert first_page.page_number == 1
    assert isinstance(first_page.text, str)
    assert len(first_page.text) > 0


def test_extract_pdf_page_numbers():
    """Test that page numbers are sequential and 1-indexed."""
    pdf_path = Path("tests/data/sample.pdf")
    
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")
    
    pages = extract_pdf(pdf_path)
    
    # Verify page numbers are sequential starting from 1
    for i, page in enumerate(pages, start=1):
        assert page.page_number == i


def test_extract_pdf_source_filename():
    """Test that source filename is correctly captured."""
    pdf_path = Path("tests/data/sample.pdf")
    
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")
    
    pages = extract_pdf(pdf_path)
    
    # All pages should have the same source filename
    for page in pages:
        assert page.source == "sample.pdf"


def test_extract_pdf_file_not_found():
    """Test that FileNotFoundError is raised for non-existent files."""
    pdf_path = Path("tests/data/nonexistent.pdf")
    
    with pytest.raises(FileNotFoundError) as exc_info:
        extract_pdf(pdf_path)
    
    assert "PDF file not found" in str(exc_info.value)
    assert "nonexistent.pdf" in str(exc_info.value)


def test_extract_pdf_with_string_path():
    """Test that extract_pdf accepts string paths."""
    pdf_path = "tests/data/sample.pdf"
    
    if not Path(pdf_path).exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")
    
    pages = extract_pdf(pdf_path)
    
    assert isinstance(pages, list)
    assert len(pages) > 0
    assert pages[0].source == "sample.pdf"


def test_extract_pdf_with_path_object():
    """Test that extract_pdf accepts Path objects."""
    pdf_path = Path("tests/data/sample.pdf")
    
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")
    
    pages = extract_pdf(pdf_path)
    
    assert isinstance(pages, list)
    assert len(pages) > 0
    assert pages[0].source == "sample.pdf"


def test_extract_pdf_preserves_ocr_metadata():
    """Test that OCR metadata is preserved in PageContent."""
    from unittest.mock import Mock, patch
    
    # Mock a PDF with low text content that triggers OCR
    with patch("manuals_lib.ingest.pdf_extractor.pymupdf.open") as mock_open:
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "A"  # Very short text
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_doc.__iter__.return_value = iter([mock_page])
        mock_open.return_value = mock_doc
        
        # Mock OCR to return text
        with patch("manuals_lib.ingest.pdf_extractor.extract_text_with_ocr") as mock_ocr:
            mock_ocr.return_value = "OCR extracted text content"
            
            pages = extract_pdf("test.pdf", use_ocr_fallback=True)
            
            assert len(pages) == 1
            assert pages[0].extraction_method == "ocr"
            assert pages[0].ocr_engine == "tesseract"
            assert pages[0].ocr_trigger_reason is not None
            assert "low_text_content" in pages[0].ocr_trigger_reason


def test_extract_pdf_tables():
    """Test table extraction from PDF."""
    pdf_path = Path("tests/data/sample.pdf")
    
    if not pdf_path.exists():
        pytest.skip("Test PDF not found")
    
    tables = extract_pdf_tables(pdf_path)
    
    # Basic validation
    assert isinstance(tables, list)
    for table in tables:
        assert hasattr(table, "table_id")
        assert hasattr(table, "source")
        assert hasattr(table, "page_number")
        assert hasattr(table, "rows")
        assert table.extraction_method == "pymupdf_find_tables"


def test_extract_pdf_invalid_file():
    """Test handling of invalid PDF files."""
    # Create a temporary non-PDF file
    invalid_path = Path("tests/data/invalid.txt")
    
    # This test assumes the file exists but is not a valid PDF
    # If you want to test this, you'd need to create a dummy file
    # For now, we'll skip if it doesn't exist
    if not invalid_path.exists():
        pytest.skip("Invalid test file not available")
    
    # pymupdf should raise an error for invalid PDFs
    with pytest.raises(Exception):  # Could be more specific with pymupdf.FileDataError
        extract_pdf(invalid_path)
