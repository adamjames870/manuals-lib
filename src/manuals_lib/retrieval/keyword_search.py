"""Simple keyword-based chunk retrieval."""

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ChunkMatch:
    """A chunk match result.
    
    Attributes:
        chunk_id: Unique chunk identifier
        source: Source document filename
        page_start: Starting page number
        page_end: Ending page number
        text: Full chunk text
        score: Match score (higher is better)
        char_count: Number of characters in chunk
    """
    chunk_id: str
    source: str
    page_start: int
    page_end: int
    text: str
    score: float
    char_count: int


def load_chunks_from_json(chunks_path: str | Path) -> list[dict]:
    """Load chunks from a JSON file.
    
    Args:
        chunks_path: Path to chunks JSON file
        
    Returns:
        List of chunk dictionaries
        
    Raises:
        FileNotFoundError: If chunks file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    chunks_path = Path(chunks_path)
    
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data.get("chunks", [])


def score_chunk(chunk_text: str, query: str) -> float:
    """Score a chunk based on keyword matches.
    
    Scoring strategy:
    - Exact phrase match (multi-word): 10.0 points per occurrence
    - Individual keyword matches: 1.0 point per occurrence
    - Single-word queries are treated as keyword matches, not phrases
    
    Args:
        chunk_text: Text content of the chunk
        query: Search query string
        
    Returns:
        Match score (higher is better)
    """
    chunk_lower = chunk_text.lower()
    query_lower = query.lower()
    
    score = 0.0
    
    # Split query into words to determine if it's a phrase
    keywords = re.findall(r'\w+', query_lower)
    
    # Only treat as phrase match if query has multiple words
    if len(keywords) > 1 and query_lower in chunk_lower:
        # Count occurrences of exact phrase
        phrase_count = chunk_lower.count(query_lower)
        score += phrase_count * 10.0
    else:
        # Score individual keywords
        for keyword in keywords:
            if len(keyword) > 2:  # Skip very short words
                # Count occurrences of this keyword
                keyword_count = chunk_lower.count(keyword)
                score += keyword_count * 1.0
    
    return score


def search_chunks(
    chunks: list[dict],
    query: str,
    top_k: int = 5
) -> list[ChunkMatch]:
    """Search chunks using keyword matching.
    
    Args:
        chunks: List of chunk dictionaries
        query: Search query string
        top_k: Number of top results to return
        
    Returns:
        List of ChunkMatch objects, sorted by score (descending)
    """
    if not query.strip():
        return []
    
    matches = []
    
    for chunk in chunks:
        text = chunk.get("text", "")
        score = score_chunk(text, query)
        
        if score > 0:
            matches.append(
                ChunkMatch(
                    chunk_id=chunk.get("chunk_id", ""),
                    source=chunk.get("source", ""),
                    page_start=chunk.get("page_start", 0),
                    page_end=chunk.get("page_end", 0),
                    text=text,
                    score=score,
                    char_count=chunk.get("char_count", len(text)),
                )
            )
    
    # Sort by score (descending)
    matches.sort(key=lambda m: m.score, reverse=True)
    
    # Return top k results
    return matches[:top_k]
