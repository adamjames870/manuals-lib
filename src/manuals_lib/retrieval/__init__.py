"""Retrieval module for searching chunks."""

from manuals_lib.retrieval.keyword_search import (
    ChunkMatch,
    load_chunks_from_json,
    search_chunks,
)

__all__ = [
    "ChunkMatch",
    "load_chunks_from_json",
    "search_chunks",
]
