"""Tests for section context inference."""

from manuals_lib.ingest.section_utils import (
    extract_heading_level,
    infer_section_context,
    is_heading_line,
)


def test_is_heading_line_numbered():
    """Test detecting numbered headings."""
    assert is_heading_line("1. Introduction")
    assert is_heading_line("1.1 Overview")
    assert is_heading_line("1.1.1 Details")
    assert is_heading_line("A. Appendix")


def test_is_heading_line_all_caps():
    """Test detecting all-caps headings."""
    assert is_heading_line("CHAPTER ONE")
    assert is_heading_line("SAFETY INFORMATION")
    assert is_heading_line("INTRODUCTION")


def test_is_heading_line_title_case():
    """Test detecting title case headings."""
    assert is_heading_line("Technical Specifications")
    assert is_heading_line("Operating Instructions")
    assert is_heading_line("Maintenance Schedule:")


def test_is_heading_line_not_heading():
    """Test that regular text is not detected as heading."""
    assert not is_heading_line("This is a regular sentence.")
    assert not is_heading_line("• Bullet point item")
    assert not is_heading_line("- Another bullet")
    assert not is_heading_line("This is a very long line that exceeds the character limit for headings.")


def test_extract_heading_level():
    """Test extracting heading levels."""
    assert extract_heading_level("1. Introduction") == 1
    assert extract_heading_level("1.1 Overview") == 2
    assert extract_heading_level("1.1.1 Details") == 3
    assert extract_heading_level("1.2.3.4 Deep Section") == 4
    
    assert extract_heading_level("CHAPTER 1") == 1
    assert extract_heading_level("SECTION A") == 2
    assert extract_heading_level("APPENDIX B") == 2


def test_infer_section_context_simple():
    """Test inferring section context from simple text."""
    text = """1. Introduction

This is the introduction section with some content.
More content here."""
    
    title, path = infer_section_context(text)
    
    assert title == "1. Introduction"
    assert path == ["1. Introduction"]


def test_infer_section_context_hierarchical():
    """Test inferring hierarchical section context."""
    text = """1. Chapter One

1.1 First Section

This is content under section 1.1."""
    
    title, path = infer_section_context(text)
    
    assert title == "1.1 First Section"
    assert path == ["1. Chapter One", "1.1 First Section"]


def test_infer_section_context_no_headings():
    """Test that no context is inferred when no headings present."""
    text = "This is just regular paragraph text without any headings."
    
    title, path = infer_section_context(text)
    
    assert title is None
    assert path is None


def test_infer_section_context_all_caps():
    """Test inferring context from all-caps headings."""
    text = """SAFETY INFORMATION

WARNINGS

Always follow safety procedures."""
    
    title, path = infer_section_context(text)
    
    assert title == "WARNINGS"
    assert "SAFETY INFORMATION" in path
    assert "WARNINGS" in path
