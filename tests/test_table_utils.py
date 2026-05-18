"""Tests for table processing utilities."""

import pytest

from manuals_lib.ingest.models import BoundingBox, TableData
from manuals_lib.ingest.table_utils import (
    detect_chunk_type,
    flatten_table_row,
    is_front_matter_or_toc,
)


def test_flatten_table_row_with_headers():
    """Test flattening a table row with headers."""
    row = ["Value1", "Value2", "Value3"]
    headers = ["Column A", "Column B", "Column C"]
    
    result = flatten_table_row(row, headers)
    
    assert "Column A: Value1" in result
    assert "Column B: Value2" in result
    assert "Column C: Value3" in result


def test_flatten_table_row_with_title():
    """Test flattening a table row with a title."""
    row = ["10", "20", "30"]
    headers = ["Min", "Max", "Avg"]
    title = "Temperature Data"
    
    result = flatten_table_row(row, headers, title)
    
    assert "Table: Temperature Data" in result
    assert "Min: 10" in result
    assert "Max: 20" in result
    assert "Avg: 30" in result


def test_flatten_table_row_without_headers():
    """Test flattening a table row without headers."""
    row = ["A", "B", "C"]
    
    result = flatten_table_row(row)
    
    assert "Column 1: A" in result
    assert "Column 2: B" in result
    assert "Column 3: C" in result


def test_flatten_table_row_empty_cells():
    """Test flattening a table row with empty cells."""
    row = ["Value1", "", "Value3"]
    headers = ["Col1", "Col2", "Col3"]
    
    result = flatten_table_row(row, headers)
    
    assert "Col1: Value1" in result
    assert "Col2" not in result  # Empty cell should be skipped
    assert "Col3: Value3" in result


def test_detect_chunk_type_toc():
    """Test detecting table of contents."""
    text = """Table of Contents
    
    Chapter 1 ........................ 5
    Chapter 2 ........................ 12
    Chapter 3 ........................ 20
    """
    
    assert detect_chunk_type(text) == "toc"


def test_detect_chunk_type_toc_keywords():
    """Test detecting TOC by keywords."""
    text = "Quick Links\n\nSection A\nSection B\nSection C"
    
    assert detect_chunk_type(text) == "toc"


def test_detect_chunk_type_index():
    """Test detecting index pages."""
    text = "Index\n\nAcoustic signals ... 45\nBattery ... 67\nCables ... 89"
    
    assert detect_chunk_type(text) == "index"


def test_detect_chunk_type_front_matter():
    """Test detecting front matter."""
    text = "Preface\n\nThis manual provides important safety information..."
    
    assert detect_chunk_type(text) == "front_matter"


def test_detect_chunk_type_content():
    """Test detecting regular content."""
    text = """The engine operates at optimal efficiency when maintained 
    according to the schedule. Regular oil changes and filter replacements 
    are essential for longevity."""
    
    assert detect_chunk_type(text) == "content"


def test_detect_chunk_type_dotted_leaders():
    """Test detecting TOC by dotted leader lines."""
    text = """Introduction ............... 1
    Safety Guidelines .......... 5
    Operation ................. 10
    Maintenance ............... 25
    """
    
    assert detect_chunk_type(text) == "toc"


def test_is_front_matter_or_toc():
    """Test exclusion filter."""
    assert is_front_matter_or_toc("toc") is True
    assert is_front_matter_or_toc("front_matter") is True
    assert is_front_matter_or_toc("index") is True
    assert is_front_matter_or_toc("content") is False
    assert is_front_matter_or_toc("table_row") is False
