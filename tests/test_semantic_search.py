"""Tests for semantic search functionality."""

import json
from pathlib import Path

import numpy as np
import pytest

from manuals_lib.embeddings.embedder import (
    cosine_similarity_matrix,
    load_index,
    semantic_search,
)
from manuals_lib.embeddings.models import ChunkMetadata, IndexManifest


def test_cosine_similarity_matrix_identical():
    """Test cosine similarity with identical vectors."""
    query = np.array([1.0, 0.0, 0.0])
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    
    similarities = cosine_similarity_matrix(query, embeddings)
    
    assert similarities.shape == (3,)
    assert similarities[0] == pytest.approx(1.0)
    assert similarities[1] == pytest.approx(0.0)
    assert similarities[2] == pytest.approx(0.0)


def test_cosine_similarity_matrix_orthogonal():
    """Test cosine similarity with orthogonal vectors."""
    query = np.array([1.0, 0.0])
    embeddings = np.array([
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    
    similarities = cosine_similarity_matrix(query, embeddings)
    
    assert similarities[0] == pytest.approx(0.0)
    assert similarities[1] == pytest.approx(0.707, abs=0.01)


def test_cosine_similarity_matrix_opposite():
    """Test cosine similarity with opposite vectors."""
    query = np.array([1.0, 0.0, 0.0])
    embeddings = np.array([
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    
    similarities = cosine_similarity_matrix(query, embeddings)
    
    assert similarities[0] == pytest.approx(-1.0)
    assert similarities[1] == pytest.approx(1.0)


def test_cosine_similarity_matrix_ranking():
    """Test that cosine similarity correctly ranks vectors by similarity."""
    query = np.array([1.0, 1.0, 0.0])
    embeddings = np.array([
        [1.0, 1.0, 0.0],  # Most similar (identical)
        [1.0, 0.5, 0.0],  # Medium similarity
        [0.0, 0.0, 1.0],  # Least similar (orthogonal)
    ])
    
    similarities = cosine_similarity_matrix(query, embeddings)
    
    # Check ranking
    assert similarities[0] > similarities[1]
    assert similarities[1] > similarities[2]
    assert similarities[0] == pytest.approx(1.0)


def test_load_index(tmp_path):
    """Test loading an index from disk."""
    index_dir = tmp_path / "test_index"
    index_dir.mkdir()
    
    # Create manifest
    manifest_data = {
        "model_name": "all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "chunk_count": 2,
        "source_file": "test.chunks.json",
        "created_at": "2024-01-01T00:00:00",
    }
    with open(index_dir / "manifest.json", "w") as f:
        json.dump(manifest_data, f)
    
    # Create chunks
    chunks_data = {
        "chunks": [
            {
                "row_index": 0,
                "chunk_id": "test_0",
                "source": "test.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "First chunk",
            },
            {
                "row_index": 1,
                "chunk_id": "test_1",
                "source": "test.pdf",
                "page_start": 2,
                "page_end": 2,
                "text": "Second chunk",
            },
        ]
    }
    with open(index_dir / "chunks.json", "w") as f:
        json.dump(chunks_data, f)
    
    # Create embeddings
    embeddings = np.random.rand(2, 384)
    np.save(index_dir / "embeddings.npy", embeddings)
    
    # Load index
    manifest, chunk_metadata, loaded_embeddings = load_index(index_dir)
    
    assert manifest.model_name == "all-MiniLM-L6-v2"
    assert manifest.embedding_dim == 384
    assert manifest.chunk_count == 2
    assert len(chunk_metadata) == 2
    assert chunk_metadata[0].chunk_id == "test_0"
    assert chunk_metadata[1].chunk_id == "test_1"
    assert loaded_embeddings.shape == (2, 384)


def test_load_index_missing_manifest(tmp_path):
    """Test loading index with missing manifest."""
    index_dir = tmp_path / "test_index"
    index_dir.mkdir()
    
    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        load_index(index_dir)


def test_load_index_missing_chunks(tmp_path):
    """Test loading index with missing chunks."""
    index_dir = tmp_path / "test_index"
    index_dir.mkdir()
    
    # Create only manifest
    manifest_data = {
        "model_name": "all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "chunk_count": 2,
        "source_file": "test.chunks.json",
        "created_at": "2024-01-01T00:00:00",
    }
    with open(index_dir / "manifest.json", "w") as f:
        json.dump(manifest_data, f)
    
    with pytest.raises(FileNotFoundError, match="Chunks not found"):
        load_index(index_dir)


def test_load_index_missing_embeddings(tmp_path):
    """Test loading index with missing embeddings."""
    index_dir = tmp_path / "test_index"
    index_dir.mkdir()
    
    # Create manifest and chunks
    manifest_data = {
        "model_name": "all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "chunk_count": 2,
        "source_file": "test.chunks.json",
        "created_at": "2024-01-01T00:00:00",
    }
    with open(index_dir / "manifest.json", "w") as f:
        json.dump(manifest_data, f)
    
    chunks_data = {"chunks": []}
    with open(index_dir / "chunks.json", "w") as f:
        json.dump(chunks_data, f)
    
    with pytest.raises(FileNotFoundError, match="Embeddings not found"):
        load_index(index_dir)


def test_semantic_search_excludes_toc_by_default(tmp_path):
    """Test that TOC chunks are excluded from search by default."""
    from manuals_lib.embeddings.embedder import build_index, semantic_search
    
    # Create test chunks with different types
    chunks_data = {
        "source": "test.pdf",
        "total_chunks": 3,
        "chunks": [
            {
                "chunk_id": "test_0",
                "source": "test.pdf",
                "page_start": 1,
                "page_end": 1,
                "char_count": 50,
                "text": "Table of Contents\nChapter 1 ... 5\nChapter 2 ... 10",
                "chunk_type": "toc",
                "extraction_method": "pymupdf",
                "table_id": None,
                "section_title": None,
            },
            {
                "chunk_id": "test_1",
                "source": "test.pdf",
                "page_start": 2,
                "page_end": 2,
                "char_count": 100,
                "text": "This document describes the technical specifications of the product.",
                "chunk_type": "content",
                "extraction_method": "pymupdf",
                "table_id": None,
                "section_title": None,
            },
            {
                "chunk_id": "test_2",
                "source": "test.pdf",
                "page_start": 3,
                "page_end": 3,
                "char_count": 80,
                "text": "The product operates at high efficiency with low maintenance.",
                "chunk_type": "content",
                "extraction_method": "pymupdf",
                "table_id": None,
                "section_title": None,
            },
        ],
    }
    
    chunks_file = tmp_path / "chunks.json"
    with open(chunks_file, "w") as f:
        import json
        json.dump(chunks_data, f)
    
    index_dir = tmp_path / "index"
    build_index(chunks_file, index_dir)
    
    # Search should exclude TOC by default
    results = semantic_search("technical specifications", index_dir, top_k=5)
    
    # Should only return content chunks, not TOC
    assert len(results) == 2
    assert all(chunk.chunk_type == "content" for _, _, chunk in results)


def test_semantic_search_include_all_types(tmp_path):
    """Test that all chunk types can be included if requested."""
    from manuals_lib.embeddings.embedder import build_index, semantic_search
    
    chunks_data = {
        "source": "test.pdf",
        "total_chunks": 2,
        "chunks": [
            {
                "chunk_id": "test_0",
                "source": "test.pdf",
                "page_start": 1,
                "page_end": 1,
                "char_count": 50,
                "text": "Table of Contents\nChapter 1 ... 5",
                "chunk_type": "toc",
                "extraction_method": "pymupdf",
                "table_id": None,
                "section_title": None,
            },
            {
                "chunk_id": "test_1",
                "source": "test.pdf",
                "page_start": 2,
                "page_end": 2,
                "char_count": 100,
                "text": "Technical content about the product.",
                "chunk_type": "content",
                "extraction_method": "pymupdf",
                "table_id": None,
                "section_title": None,
            },
        ],
    }
    
    chunks_file = tmp_path / "chunks.json"
    with open(chunks_file, "w") as f:
        import json
        json.dump(chunks_data, f)
    
    index_dir = tmp_path / "index"
    build_index(chunks_file, index_dir)
    
    # Search with no exclusions
    results = semantic_search(
        "table of contents",
        index_dir,
        top_k=5,
        exclude_chunk_types=[]
    )
    
    # Should return all chunks
    assert len(results) == 2


def test_semantic_search_invalid_top_k(tmp_path):
    """Test semantic search with invalid top_k."""
    index_dir = tmp_path / "test_index"
    index_dir.mkdir()
    
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        semantic_search("test query", index_dir, top_k=0)
