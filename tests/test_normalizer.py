"""Tests for text normalization."""

from manuals_lib.ingest.models import PageContent
from manuals_lib.ingest.normalizer import (
    collapse_blank_lines,
    is_special_line,
    join_wrapped_lines,
    normalize_line_endings,
    normalize_pages,
    normalize_text,
)


def test_normalize_line_endings():
    """Test line ending normalization."""
    text = "Line 1\r\nLine 2\rLine 3\nLine 4"
    result = normalize_line_endings(text)
    assert result == "Line 1\nLine 2\nLine 3\nLine 4"


def test_collapse_blank_lines():
    """Test collapsing excessive blank lines."""
    text = "Line 1\n\n\n\n\nLine 2"
    result = collapse_blank_lines(text, max_consecutive=2)
    assert result == "Line 1\n\nLine 2"


def test_collapse_blank_lines_preserves_double():
    """Test that double blank lines are preserved."""
    text = "Line 1\n\nLine 2"
    result = collapse_blank_lines(text, max_consecutive=2)
    assert result == "Line 1\n\nLine 2"


def test_is_special_line_bullet():
    """Test detection of bullet points."""
    assert is_special_line("• Item one")
    assert is_special_line("- Item two")
    assert is_special_line("* Item three")


def test_is_special_line_numbered():
    """Test detection of numbered lists."""
    assert is_special_line("1. First item")
    assert is_special_line("a) Second item")
    assert is_special_line("2) Third item")


def test_is_special_line_short():
    """Test detection of short lines."""
    assert is_special_line("Chapter 1")
    assert is_special_line("Introduction")


def test_is_special_line_ending_punctuation():
    """Test detection of lines ending with punctuation."""
    assert is_special_line("This is a complete sentence.")
    assert is_special_line("A question?")
    assert is_special_line("An exclamation!")


def test_is_special_line_normal():
    """Test that normal paragraph lines are not special."""
    assert not is_special_line(
        "This is a normal paragraph line that should be joined with the next line"
    )


def test_join_wrapped_lines_simple():
    """Test joining simple wrapped lines."""
    text = "This is a long line that has been\nwrapped to the next line."
    result = join_wrapped_lines(text)
    assert result == "This is a long line that has been wrapped to the next line."


def test_join_wrapped_lines_preserves_bullets():
    """Test that bullet points are not joined."""
    text = "• First item\n• Second item"
    result = join_wrapped_lines(text)
    assert result == "• First item\n• Second item"


def test_join_wrapped_lines_preserves_headings():
    """Test that short lines (headings) are not joined."""
    text = "Chapter 1\nThis is the content of chapter one that continues here."
    result = join_wrapped_lines(text)
    # Chapter 1 should not be joined with next line
    assert "Chapter 1\n" in result


def test_normalize_text_full():
    """Test full normalization pipeline."""
    text = "Line 1\r\n\n\n\nLine 2\nwrapped line  "
    result = normalize_text(text, join_wrapped=True)
    
    # Should normalize line endings, collapse blanks, join wrapped, and trim
    assert "\r" not in result
    assert "\n\n\n" not in result
    assert result.strip() == result


def test_normalize_text_without_joining():
    """Test normalization without joining wrapped lines."""
    text = "Line 1\nLine 2"
    result = normalize_text(text, join_wrapped=False)
    assert result == "Line 1\nLine 2"


def test_normalize_pages():
    """Test normalizing a list of pages."""
    pages = [
        PageContent(source="test.pdf", page_number=1, text="Line 1\r\nLine 2  "),
        PageContent(source="test.pdf", page_number=2, text="  Line 3\n\n\n\nLine 4"),
    ]
    
    result = normalize_pages(pages, join_wrapped=False)
    
    assert len(result) == 2
    assert result[0].source == "test.pdf"
    assert result[0].page_number == 1
    assert "\r" not in result[0].text
    assert result[0].text.strip() == result[0].text
    
    assert result[1].page_number == 2
    assert "\n\n\n\n" not in result[1].text


def test_normalize_pages_preserves_metadata():
    """Test that normalization preserves source and page number."""
    pages = [
        PageContent(source="manual.pdf", page_number=5, text="Some text"),
    ]
    
    result = normalize_pages(pages)
    
    assert result[0].source == "manual.pdf"
    assert result[0].page_number == 5
