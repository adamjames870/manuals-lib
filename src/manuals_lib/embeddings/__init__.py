"""Embeddings module for generating and managing text embeddings."""

from manuals_lib.embeddings.embedder import (
    Embedder,
    build_index,
    cosine_similarity,
    cosine_similarity_matrix,
    load_chunks_from_json,
)
from manuals_lib.embeddings.models import ChunkMetadata, IndexManifest

__all__ = [
    "Embedder",
    "build_index",
    "cosine_similarity",
    "cosine_similarity_matrix",
    "load_chunks_from_json",
    "ChunkMetadata",
    "IndexManifest",
]
