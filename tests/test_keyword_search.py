"""Tests for keyword search retrieval."""

import json

import pytest

from manuals_lib.retrieval.keyword_search import (
    ChunkMatch,
    load_chunks_from_json,
    score_chunk,
    search_chunks,
)


def test_score_chunk_exact_phrase():
    """Test scoring with exact phrase match."""
    chunk_text = "This is a test document about Python programming."
    query = "Python programming"
    
    score, _, _, _ = score_chunk(chunk_text, query)
    
    # Should have high score for exact phrase match
    assert score >= 10.0


def test_score_chunk_case_insensitive():
    """Test that scoring is case-insensitive."""
    chunk_text = "This is a TEST document."
    query = "test"
    
    score, _, _, _ = score_chunk(chunk_text, query)
    
    assert score > 0


def test_score_chunk_individual_keywords():
    """Test scoring with individual keyword matches."""
    chunk_text = "Python is great. I love Python. Python rocks."
    query = "Python"
    
    score, _, _, _ = score_chunk(chunk_text, query)
    
    # Should score for multiple occurrences
    assert score > 0


def test_score_chunk_no_match():
    """Test scoring with no matches."""
    chunk_text = "This is about Java programming."
    query = "Python"
    
    score, matched_terms, exact_phrase_match, first_match_pos = score_chunk(chunk_text, query)
    
    assert score == 0.0
    assert matched_terms == []
    assert exact_phrase_match is False
    assert first_match_pos == -1


def test_score_chunk_phrase_higher_than_keywords():
    """Test that exact phrase scores higher than individual keywords."""
    chunk_text = "Python programming is fun. Python is great. Programming rocks."
    
    phrase_score, _, _, _ = score_chunk(chunk_text, "Python programming")
    keyword_score, _, _, _ = score_chunk(chunk_text, "Python")
    
    # Phrase match should score higher
    assert phrase_score > keyword_score


def test_score_chunk_multiple_phrase_occurrences():
    """Test scoring with multiple exact phrase matches."""
    chunk_text = "Python programming. More Python programming. Even more Python programming."
    query = "Python programming"
    
    score, _, _, _ = score_chunk(chunk_text, query)
    
    # Should count all three occurrences (now 100 points each)
    assert score >= 300.0  # 3 phrases * 100 points each


def test_search_chunks_basic():
    """Test basic chunk searching."""
    chunks = [
        {
            "chunk_id": "test_chunk_0001",
            "source": "test.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "This is about Python programming.",
            "char_count": 33,
        },
        {
            "chunk_id": "test_chunk_0002",
            "source": "test.pdf",
            "page_start": 2,
            "page_end": 2,
            "text": "This is about Java development.",
            "char_count": 31,
        },
    ]
    
    matches = search_chunks(chunks, "Python", top_k=5)
    
    assert len(matches) == 1
    assert matches[0].chunk_id == "test_chunk_0001"
    assert matches[0].score > 0


def test_search_chunks_ranking():
    """Test that results are ranked by score."""
    chunks = [
        {
            "chunk_id": "test_chunk_0001",
            "source": "test.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "Python is mentioned once.",
            "char_count": 25,
        },
        {
            "chunk_id": "test_chunk_0002",
            "source": "test.pdf",
            "page_start": 2,
            "page_end": 2,
            "text": "Python Python Python appears three times.",
            "char_count": 41,
        },
    ]
    
    matches = search_chunks(chunks, "Python", top_k=5)
    
    assert len(matches) == 2
    # Higher scoring chunk should be first
    assert matches[0].chunk_id == "test_chunk_0002"
    assert matches[0].score > matches[1].score


def test_search_chunks_top_k():
    """Test that top_k limits results."""
    chunks = [
        {
            "chunk_id": f"test_chunk_{i:04d}",
            "source": "test.pdf",
            "page_start": i,
            "page_end": i,
            "text": f"Python content {i}",
            "char_count": 20,
        }
        for i in range(1, 11)
    ]
    
    matches = search_chunks(chunks, "Python", top_k=3)
    
    assert len(matches) == 3


def test_search_chunks_empty_query():
    """Test searching with empty query."""
    chunks = [
        {
            "chunk_id": "test_chunk_0001",
            "source": "test.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "Some content",
            "char_count": 12,
        },
    ]
    
    matches = search_chunks(chunks, "", top_k=5)
    
    assert len(matches) == 0


def test_search_chunks_no_matches():
    """Test searching with no matching chunks."""
    chunks = [
        {
            "chunk_id": "test_chunk_0001",
            "source": "test.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "This is about Java.",
            "char_count": 19,
        },
    ]
    
    matches = search_chunks(chunks, "Python", top_k=5)
    
    assert len(matches) == 0


def test_load_chunks_from_json(tmp_path):
    """Test loading chunks from JSON file."""
    chunks_data = {
        "source": "test.pdf",
        "total_chunks": 2,
        "chunks": [
            {
                "chunk_id": "test_chunk_0001",
                "page_start": 1,
                "page_end": 1,
                "text": "First chunk",
                "char_count": 11,
            },
            {
                "chunk_id": "test_chunk_0002",
                "page_start": 2,
                "page_end": 2,
                "text": "Second chunk",
                "char_count": 12,
            },
        ],
    }
    
    chunks_path = tmp_path / "test.chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f)
    
    chunks = load_chunks_from_json(chunks_path)
    
    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "test_chunk_0001"
    assert chunks[1]["chunk_id"] == "test_chunk_0002"


def test_load_chunks_from_json_file_not_found():
    """Test loading from non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_chunks_from_json("nonexistent.json")


def test_chunk_match_dataclass():
    """Test ChunkMatch dataclass creation."""
    match = ChunkMatch(
        chunk_id="test_chunk_0001",
        source="test.pdf",
        page_start=1,
        page_end=2,
        text="Test content",
        score=15.5,
        char_count=12,
        matched_terms=["test"],
        exact_phrase_match=False,
        first_match_pos=0,
    )
    
    assert match.chunk_id == "test_chunk_0001"
    assert match.source == "test.pdf"
    assert match.page_start == 1
    assert match.page_end == 2
    assert match.text == "Test content"
    assert match.score == 15.5
    assert match.char_count == 12
    assert match.matched_terms == ["test"]
    assert match.exact_phrase_match is False
    assert match.first_match_pos == 0


def test_score_chunk_position_bonus():
    """Test that matches near the start score higher."""
    # Match at the beginning
    early_text = "Python programming is great. " + "Other content. " * 50
    early_score, _, _, _ = score_chunk(early_text, "Python programming")
    
    # Match later in the text
    late_text = "Other content. " * 50 + "Python programming is great."
    late_score, _, _, _ = score_chunk(late_text, "Python programming")
    
    # Early match should score higher
    assert early_score > late_score


def test_is_front_matter_toc():
    """Test detection of table of contents."""
    from manuals_lib.retrieval.keyword_search import is_front_matter
    
    toc_text = """Table of Contents
    
    Chapter 1: Introduction .................. 1
    Chapter 2: Getting Started ............... 15
    Chapter 3: Advanced Topics ............... 42
    Chapter 4: Conclusion .................... 89
    """
    
    assert is_front_matter(toc_text) is True


def test_is_front_matter_regular_content():
    """Test that regular content is not detected as front matter."""
    from manuals_lib.retrieval.keyword_search import is_front_matter
    
    regular_text = """This is a detailed explanation of Python programming.
    
    Python is a high-level programming language that emphasizes code readability.
    It supports multiple programming paradigms including procedural, object-oriented,
    and functional programming. Python's design philosophy emphasizes code readability
    with its notable use of significant whitespace.
    """
    
    assert is_front_matter(regular_text) is False


def test_search_chunks_content_beats_toc():
    """Test that relevant content ranks higher than table of contents."""
    chunks = [
        {
            "chunk_id": "manual_chunk_0001",
            "source": "manual.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": """Table of Contents
            
            Chapter 1: Python Basics ................ 5
            Chapter 2: Python Functions ............. 20
            Chapter 3: Python Classes ............... 45
            Chapter 4: Python Modules ............... 78
            """,
            "char_count": 200,
        },
        {
            "chunk_id": "manual_chunk_0005",
            "source": "manual.pdf",
            "page_start": 5,
            "page_end": 5,
            "text": """Python Basics
            
            Python is a high-level programming language. To start programming in Python,
            you need to understand variables, data types, and control structures. Python
            uses indentation to define code blocks, making it very readable. Python supports
            multiple programming paradigms and has a rich standard library.
            """,
            "char_count": 300,
        },
    ]
    
    matches = search_chunks(chunks, "Python programming", top_k=5)
    
    # Content chunk should rank higher than TOC
    assert len(matches) == 2
    assert matches[0].chunk_id == "manual_chunk_0005"
    assert matches[0].score > matches[1].score


def test_search_chunks_exact_phrase_metadata():
    """Test that exact phrase match is properly recorded."""
    chunks = [
        {
            "chunk_id": "test_chunk_0001",
            "source": "test.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "This document covers Python programming in detail.",
            "char_count": 50,
        },
    ]
    
    matches = search_chunks(chunks, "Python programming", top_k=5)
    
    assert len(matches) == 1
    assert matches[0].exact_phrase_match is True
    assert "Python programming" in matches[0].matched_terms


def test_search_chunks_keyword_metadata():
    """Test that keyword matches are properly recorded."""
    chunks = [
        {
            "chunk_id": "test_chunk_0001",
            "source": "test.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "Python is great. Programming is fun.",
            "char_count": 36,
        },
    ]
    
    matches = search_chunks(chunks, "Python programming", top_k=5)
    
    assert len(matches) == 1
    assert matches[0].exact_phrase_match is False
    assert "python" in matches[0].matched_terms
    assert "programming" in matches[0].matched_terms


def test_search_chunks_first_match_position():
    """Test that first match position is recorded."""
    chunks = [
        {
            "chunk_id": "test_chunk_0001",
            "source": "test.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "Some introductory text. Python programming is discussed here.",
            "char_count": 61,
        },
    ]
    
    matches = search_chunks(chunks, "Python", top_k=5)
    
    assert len(matches) == 1
    assert matches[0].first_match_pos > 0
    assert "python" in matches[0].text[matches[0].first_match_pos:].lower()
