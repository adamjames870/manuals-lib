#!/usr/bin/env python3
"""Semantic search over embedding index using sentence transformers."""

import argparse
import re
from pathlib import Path

from manuals_lib.embeddings.embedder import semantic_search


def format_page_range(page_start: int, page_end: int) -> str:
    """Format page range for display.
    
    Args:
        page_start: Starting page number
        page_end: Ending page number
        
    Returns:
        Formatted page range string
    """
    if page_start == page_end:
        return f"p.{page_start}"
    return f"pp.{page_start}-{page_end}"


def get_preview(text: str, query: str, max_length: int = 300) -> str:
    """Get a preview of text, centered around query match if possible.
    
    Args:
        text: Full text to preview
        query: Query string to search for
        max_length: Maximum preview length
        
    Returns:
        Preview string with ellipsis if truncated
    """
    # Try to find query terms in text (case-insensitive)
    query_lower = query.lower()
    text_lower = text.lower()
    
    # Try exact phrase match first
    match_pos = text_lower.find(query_lower)
    
    # If no exact match, try individual words
    if match_pos == -1:
        words = re.findall(r'\w+', query_lower)
        for word in words:
            if len(word) > 2:  # Skip very short words
                match_pos = text_lower.find(word)
                if match_pos != -1:
                    break
    
    # If we found a match, center preview around it
    if match_pos != -1:
        # Calculate start position to center the match
        half_length = max_length // 2
        start = max(0, match_pos - half_length)
        
        # Adjust start to word boundary if possible
        if start > 0:
            # Look for space before start
            space_pos = text.rfind(' ', 0, start + 20)
            if space_pos != -1 and space_pos > start - 20:
                start = space_pos + 1
        
        end = start + max_length
        preview = text[start:end]
        
        # Add ellipsis
        if start > 0:
            preview = "..." + preview
        if end < len(text):
            preview = preview + "..."
    else:
        # No match found, show beginning
        preview = text[:max_length]
        if len(text) > max_length:
            preview = preview + "..."
    
    return preview.strip()


def main():
    """Run semantic search CLI."""
    parser = argparse.ArgumentParser(
        description="Semantic search over embedding index"
    )
    parser.add_argument(
        "--index",
        type=Path,
        required=True,
        help="Path to index directory (containing manifest.json, chunks.json, embeddings.npy)",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query text",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    
    args = parser.parse_args()
    
    # Validate index directory
    if not args.index.exists():
        print(f"Error: Index directory not found: {args.index}")
        return 1
    
    if not args.index.is_dir():
        print(f"Error: Index path is not a directory: {args.index}")
        return 1
    
    # Perform search
    try:
        results = semantic_search(
            query=args.query,
            index_dir=args.index,
            top_k=args.top_k,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error during search: {e}")
        return 1
    
    # Display results
    print(f"\nSearch results for: '{args.query}'")
    print(f"Index: {args.index}")
    print("=" * 80)
    print()
    
    if not results:
        print("No results found.")
        return 0
    
    for rank, similarity, chunk in results:
        page_range = format_page_range(chunk.page_start, chunk.page_end)
        preview = get_preview(chunk.text, args.query)
        
        print(f"Rank {rank} | Score: {similarity:.4f}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Source: {chunk.source} | {page_range}")
        print(f"Preview: {preview}")
        print("-" * 80)
        print()
    
    return 0


if __name__ == "__main__":
    exit(main())
