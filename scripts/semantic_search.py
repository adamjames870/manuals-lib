#!/usr/bin/env python3
"""Semantic search over embedding index using sentence transformers."""

import argparse
import re
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from manuals_lib.embeddings.embedder import semantic_search
from manuals_lib.embeddings.models import ChunkMetadata


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


def find_all_indexes(base_dir: Path = Path("data/index")) -> list[Path]:
    """Find all index directories under base_dir.
    
    An index directory must contain a manifest.json file.
    
    Args:
        base_dir: Base directory to search for indexes
        
    Returns:
        List of index directory paths
    """
    if not base_dir.exists():
        return []
    
    indexes = []
    for path in base_dir.rglob("manifest.json"):
        index_dir = path.parent
        indexes.append(index_dir)
    
    return sorted(indexes)


def search_all_indexes(
    query: str,
    indexes: list[Path],
    top_k: int = 5,
    exclude_chunk_types: list[str] | None = None,
) -> list[tuple[int, float, ChunkMetadata, Path]]:
    """Search multiple indexes and return combined results.
    
    Args:
        query: Search query text
        indexes: List of index directories to search
        top_k: Number of results to return per index
        exclude_chunk_types: List of chunk types to exclude
        
    Returns:
        List of (rank, similarity_score, chunk_metadata, index_path) tuples,
        sorted by similarity score (highest first)
    """
    all_results = []
    
    for index_dir in indexes:
        try:
            results = semantic_search(
                query=query,
                index_dir=index_dir,
                top_k=top_k,
                exclude_chunk_types=exclude_chunk_types,
            )
            # Add index_dir to each result
            for _, similarity, chunk in results:
                all_results.append((similarity, chunk, index_dir))
        except Exception as e:
            print(f"Warning: Failed to search {index_dir}: {e}", flush=True)
            continue
    
    # Sort by similarity (descending)
    all_results.sort(key=lambda x: x[0], reverse=True)
    
    # Re-rank and return top results
    ranked_results = []
    for rank, (similarity, chunk, index_dir) in enumerate(all_results[:top_k], start=1):
        ranked_results.append((rank, similarity, chunk, index_dir))
    
    return ranked_results


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
        help="Path to index directory (containing manifest.json, chunks.json, embeddings.npy). "
             "If omitted, searches all indexes under data/index/",
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
    parser.add_argument(
        "--include-toc",
        action="store_true",
        help="Include TOC/front-matter/index chunks in results",
    )
    
    args = parser.parse_args()
    console = Console()
    
    # Determine which indexes to search
    if args.index:
        # Single index specified
        if not args.index.exists():
            console.print(f"[red]Error:[/red] Index directory not found: {args.index}")
            return 1
        
        if not args.index.is_dir():
            console.print(f"[red]Error:[/red] Index path is not a directory: {args.index}")
            return 1
        
        indexes = [args.index]
        search_mode = "single"
    else:
        # Search all indexes
        indexes = find_all_indexes()
        if not indexes:
            console.print("[red]Error:[/red] No indexes found under data/index/")
            console.print("[dim]Run scripts/build_embeddings.py first to create indexes.[/dim]")
            return 1
        search_mode = "all"
    
    # Determine exclusions
    exclude_types = [] if args.include_toc else None
    
    # Perform search
    try:
        with console.status("[bold blue]Searching...", spinner="dots"):
            if search_mode == "single":
                results = semantic_search(
                    query=args.query,
                    index_dir=indexes[0],
                    top_k=args.top_k,
                    exclude_chunk_types=exclude_types,
                )
                # Convert to format with index_dir
                results_with_index = [
                    (rank, similarity, chunk, indexes[0])
                    for rank, similarity, chunk in results
                ]
            else:
                results_with_index = search_all_indexes(
                    query=args.query,
                    indexes=indexes,
                    top_k=args.top_k,
                    exclude_chunk_types=exclude_types,
                )
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1
    except Exception as e:
        console.print(f"[red]Error during search:[/red] {e}")
        return 1
    
    # Display header
    console.print()
    header = Text()
    header.append("Search Query: ", style="bold blue")
    header.append(f'"{args.query}"', style="bold yellow")
    console.print(Panel(header, border_style="blue"))
    
    if search_mode == "single":
        console.print(f"[dim]Index:[/dim] {indexes[0]}")
    else:
        console.print(f"[dim]Searched {len(indexes)} index(es)[/dim]")
    
    console.print()
    
    if not results_with_index:
        console.print("[yellow]No results found.[/yellow]")
        return 0
    
    # Display results
    for rank, similarity, chunk, index_dir in results_with_index:
        page_range = format_page_range(chunk.page_start, chunk.page_end)
        preview = get_preview(chunk.text, args.query)
        
        # Create result table
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="dim", width=12)
        table.add_column()
        
        # Rank and score
        score_color = "green" if similarity > 0.7 else "yellow" if similarity > 0.5 else "white"
        table.add_row(
            "Rank:",
            f"[bold]{rank}[/bold] | Score: [{score_color}]{similarity:.4f}[/{score_color}]"
        )
        
        # Source info
        type_label = f" [{chunk.chunk_type}]" if chunk.chunk_type != "content" else ""
        table.add_row("Source:", f"[cyan]{chunk.source}[/cyan] | {page_range}{type_label}")
        
        # Index (only for multi-index search)
        if search_mode == "all":
            table.add_row("Index:", f"[dim]{index_dir.name}[/dim]")
        
        # Chunk ID
        table.add_row("Chunk ID:", f"[dim]{chunk.chunk_id}[/dim]")
        
        console.print(table)
        
        # Preview in a panel
        preview_panel = Panel(
            preview,
            title="[bold]Preview[/bold]",
            title_align="left",
            border_style="dim",
            padding=(0, 1),
        )
        console.print(preview_panel)
        console.print()
    
    return 0


if __name__ == "__main__":
    exit(main())
