"""Tests for data models."""

from manuals_lib.ingest.models import PageContent


def test_page_content_creation():
    """Test creating a PageContent instance."""
    page = PageContent(
        source="test.pdf",
        page_number=1,
        text="Sample text content",
    )
    
    assert page.source == "test.pdf"
    assert page.page_number == 1
    assert page.text == "Sample text content"


def test_page_content_with_empty_text():
    """Test PageContent with empty text."""
    page = PageContent(
        source="empty.pdf",
        page_number=5,
        text="",
    )
    
    assert page.source == "empty.pdf"
    assert page.page_number == 5
    assert page.text == ""
