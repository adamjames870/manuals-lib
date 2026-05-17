"""Tests for embeddings module."""

import json
from pathlib import Path

import numpy as np
import pytest

from manuals_lib.embeddings import (
    Embedder,
    build_index,
    cosine_similarity,
    cosine_similarity_matrix,
    load_chunks_from_json,
)
from manuals_lib.embeddings.models import ChunkMetadata, IndexManifest


def test_embedder_initialization():
    """Test embedder can be initialized."""
    embedder = Embedder()
    
    assert embedder.model_name == "all-MiniLM-L6-v2"
    assert embedder.embedding_dim > 0


def test_embedder_embed_texts():
    """Test embedding generation."""
    embedder = Embedder()
    
    texts = [
        "This is a test sentence.",
        "Another test sentence here.",
    ]
    
    embeddings = embedder.embed_texts(texts)
    
    assert embeddings.shape == (2, embedder.embedding_dim)
    assert embeddings.dtype == np.float32 or embeddings.dtype == np.float64


def test_embedder_consistent_embeddings():
    """Test that same text produces same embedding."""
    embedder = Embedder()
    
    text = "Consistent test sentence."
    
    embedding1 = embedder.embed_texts([text])
    embedding2 = embedder.embed_texts([text])
    
    np.testing.assert_array_almost_equal(embedding1, embedding2)


def test_cosine_similarity():
    """Test cosine similarity computation."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    
    # Identical vectors
    assert cosine_similarity(a, b) == pytest.approx(1.0)
    
    # Orthogonal vectors
    assert cosine_similarity(a, c) == pytest.approx(0.0)


def test_cosine_similarity_matrix():
    """Test batch cosine similarity computation."""
    query = np.array([1.0, 0.0, 0.0])
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
    ])
    
    similarities = cosine_similarity_matrix(query, embeddings)
    
    assert similarities.shape == (3,)
    assert similarities[0] == pytest.approx(1.0)  # Identical
    assert similarities[1] == pytest.approx(0.0)  # Orthogonal
    assert 0.0 < similarities[2] < 1.0  # Partial match


def test_load_chunks_from_json(tmp_path):
    """Test loading chunks from JSON."""
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
        load_chunks_from_json(Path("nonexistent.json"))


def test_build_index(tmp_path):
    """Test building an embedding index."""
    # Create test chunks file
    chunks_data = {
        "source": "test.pdf",
        "total_chunks": 3,
        "chunks": [
            {
                "chunk_id": "test_chunk_0001",
                "source": "test.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "First test chunk with some content.",
                "char_count": 35,
            },
            {
                "chunk_id": "test_chunk_0002",
                "source": "test.pdf",
                "page_start": 2,
                "page_end": 2,
                "text": "Second test chunk with different content.",
                "char_count": 41,
            },
            {
                "chunk_id": "test_chunk_0003",
                "source": "test.pdf",
                "page_start": 3,
                "page_end": 3,
                "text": "Third test chunk with more text.",
                "char_count": 32,
            },
        ],
    }
    
    chunks_path = tmp_path / "test.chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f)
    
    output_dir = tmp_path / "index"
    
    # Build index
    index_dir, manifest = build_index(
        chunks_path=chunks_path,
        output_dir=output_dir,
    )
    
    # Check index directory exists
    assert index_dir.exists()
    assert index_dir.name == "test"
    
    # Check manifest
    assert manifest.chunk_count == 3
    assert manifest.embedding_dim > 0
    assert manifest.model_name == "all-MiniLM-L6-v2"
    
    # Check files exist
    assert (index_dir / "manifest.json").exists()
    assert (index_dir / "chunks.json").exists()
    assert (index_dir / "embeddings.npy").exists()
    
    # Load and verify embeddings
    embeddings = np.load(index_dir / "embeddings.npy")
    assert embeddings.shape == (3, manifest.embedding_dim)
    
    # Load and verify chunks
    with open(index_dir / "chunks.json", "r", encoding="utf-8") as f:
        saved_chunks = json.load(f)
    
    assert len(saved_chunks["chunks"]) == 3
    assert saved_chunks["chunks"][0]["row_index"] == 0
    assert saved_chunks["chunks"][0]["chunk_id"] == "test_chunk_0001"
    assert saved_chunks["chunks"][1]["row_index"] == 1
    assert saved_chunks["chunks"][2]["row_index"] == 2


def test_build_index_empty_chunks(tmp_path):
    """Test building index with empty chunks list."""
    chunks_data = {
        "source": "test.pdf",
        "total_chunks": 0,
        "chunks": [],
    }
    
    chunks_path = tmp_path / "test.chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f)
    
    output_dir = tmp_path / "index"
    
    with pytest.raises(ValueError, match="No chunks found"):
        build_index(chunks_path=chunks_path, output_dir=output_dir)


def test_chunk_metadata_creation():
    """Test ChunkMetadata dataclass."""
    metadata = ChunkMetadata(
        row_index=0,
        chunk_id="test_chunk_0001",
        source="test.pdf",
        page_start=1,
        page_end=2,
        text="Test content",
    )
    
    assert metadata.row_index == 0
    assert metadata.chunk_id == "test_chunk_0001"
    assert metadata.source == "test.pdf"
    assert metadata.page_start == 1
    assert metadata.page_end == 2
    assert metadata.text == "Test content"


def test_index_manifest_creation():
    """Test IndexManifest creation."""
    manifest = IndexManifest.create(
        model_name="test-model",
        embedding_dim=384,
        chunk_count=10,
        source_file="test.chunks.json",
    )
    
    assert manifest.model_name == "test-model"
    assert manifest.embedding_dim == 384
    assert manifest.chunk_count == 10
    assert manifest.source_file == "test.chunks.json"
    assert manifest.created_at.endswith("Z")
