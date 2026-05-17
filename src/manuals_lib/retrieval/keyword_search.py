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
        matched_terms: List of terms that matched
        exact_phrase_match: Whether the exact phrase was found
        first_match_pos: Character position of first match
    """
    chunk_id: str
    source: str
    page_start: int
    page_end: int
    text: str
    score: float
    char_count: int
    matched_terms: list[str]
    exact_phrase_match: bool
    first_match_pos: int


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


def is_front_matter(chunk_text: str) -> bool:
    """Detect if chunk looks like table of contents or front matter.
    
    Args:
        chunk_text: Text content of the chunk
        
    Returns:
        True if chunk appears to be front matter
    """
    text_lower = chunk_text.lower()
    
    # Check for common front matter indicators
    front_matter_indicators = [
        'table of contents',
        'contents',
        'chapter',
        'section',
        'page',
    ]
    
    # Count lines that look like TOC entries (short lines with numbers)
    lines = chunk_text.split('\n')
    toc_like_lines = 0
    
    for line in lines:
        line_stripped = line.strip()
        if len(line_stripped) < 50 and re.search(r'\d+$', line_stripped):
            toc_like_lines += 1
    
    # If more than 30% of lines look like TOC entries
    if len(lines) > 0 and toc_like_lines / len(lines) > 0.3:
        return True
    
    # Check for front matter keywords
    for indicator in front_matter_indicators:
        if indicator in text_lower[:200]:  # Check first 200 chars
            return True
    
    return False


def score_chunk(chunk_text: str, query: str) -> tuple[float, list[str], bool, int]:
    """Score a chunk based on keyword matches.
    
    Scoring strategy:
    - Exact phrase match (multi-word): 100.0 points per occurrence
    - Individual keyword matches: 1.0 point per occurrence
    - Position bonus: matches near start score higher (up to 20% bonus)
    - Front matter penalty: -50% if chunk looks like TOC/front matter
    - Single-word queries are treated as keyword matches, not phrases
    
    Args:
        chunk_text: Text content of the chunk
        query: Search query string
        
    Returns:
        Tuple of (score, matched_terms, exact_phrase_match, first_match_pos)
    """
    chunk_lower = chunk_text.lower()
    query_lower = query.lower()
    
    score = 0.0
    matched_terms = []
    exact_phrase_match = False
    first_match_pos = -1
    
    # Split query into words to determine if it's a phrase
    keywords = re.findall(r'\w+', query_lower)
    
    # Only treat as phrase match if query has multiple words
    if len(keywords) > 1 and query_lower in chunk_lower:
        exact_phrase_match = True
        # Find first occurrence position
        first_match_pos = chunk_lower.find(query_lower)
        
        # Count occurrences of exact phrase
        phrase_count = chunk_lower.count(query_lower)
        score += phrase_count * 100.0
        
        matched_terms.append(query)
        
        # Position bonus: earlier matches score higher
        if first_match_pos >= 0:
            # Bonus decreases from 20% at position 0 to 0% at position 500
            position_bonus = max(0, 1 - (first_match_pos / 500)) * 0.2
            score *= (1 + position_bonus)
    else:
        # Score individual keywords
        for keyword in keywords:
            if len(keyword) > 2:  # Skip very short words
                if keyword in chunk_lower:
                    matched_terms.append(keyword)
                    
                    # Find first occurrence of this keyword
                    keyword_pos = chunk_lower.find(keyword)
                    if first_match_pos < 0 or keyword_pos < first_match_pos:
                        first_match_pos = keyword_pos
                    
                    # Count occurrences of this keyword
                    keyword_count = chunk_lower.count(keyword)
                    keyword_score = keyword_count * 1.0
                    
                    # Position bonus for keywords too
                    if keyword_pos >= 0:
                        position_bonus = max(0, 1 - (keyword_pos / 500)) * 0.2
                        keyword_score *= (1 + position_bonus)
                    
                    score += keyword_score
    
    # Apply front matter penalty
    if is_front_matter(chunk_text):
        score *= 0.5
    
    return score, matched_terms, exact_phrase_match, first_match_pos


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
        score, matched_terms, exact_phrase_match, first_match_pos = score_chunk(text, query)
        
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
                    matched_terms=matched_terms,
                    exact_phrase_match=exact_phrase_match,
                    first_match_pos=first_match_pos,
                )
            )
    
    # Sort by score (descending)
    matches.sort(key=lambda m: m.score, reverse=True)
    
    # Return top k results
    return matches[:top_k]
