#!/usr/bin/env python3
"""Search chunks using keyword matching."""

import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from manuals_lib.retrieval import load_chunks_from_json, search_chunks


def format_page_range(page_start: int, page_end: int) -> str:
    """Format page range for display.
    
    Args:
        page_start: Starting page number
        page_end: Ending page number
        
    Returns:
        Formatted page range string
    """
    if page_start == page_end:
        return f"Page {page_start}"
    return f"Pages {page_start}-{page_end}"


def get_preview(text: str, max_length: int = 200) -> str:
    """Get a preview of chunk text.
    
    Args:
        text: Full chunk text
        max_length: Maximum preview length
        
    Returns:
        Preview string with ellipsis if truncated
    """
    if len(text) <= max_length:
        return text
    
    # Try to break at a sentence or word boundary
    preview = text[:max_length]
    
    # Look for last sentence boundary
    last_period = preview.rfind('. ')
    if last_period > max_length // 2:
        return preview[:last_period + 1]
    
    # Look for last word boundary
    last_space = preview.rfind(' ')
    if last_space > 0:
        return preview[:last_space] + "..."
    
    return preview + "..."


def main():
    """Search chunks from command line."""
    parser = argparse.ArgumentParser(
        description="Search chunks using keyword matching"
    )
    parser.add_argument(
        "--chunks",
        required=True,
        help="Path to chunks JSON file",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Search query string",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to return (default: 5)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show full chunk text instead of preview",
    )
    
    args = parser.parse_args()
    console = Console()
    
    # Load chunks
    try:
        chunks = load_chunks_from_json(args.chunks)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1
    except Exception as e:
        console.print(f"[red]Error loading chunks:[/red] {e}")
        return 1
    
    if not chunks:
        console.print("[yellow]No chunks found in file[/yellow]")
        return 0
    
    # Search
    console.print(f"\n[blue]Searching {len(chunks)} chunks for:[/blue] {args.query}\n")
    
    matches = search_chunks(chunks, args.query, args.top_k)
    
    if not matches:
        console.print("[yellow]No matches found[/yellow]")
        return 0
    
    # Display results
    console.print(f"[green]Found {len(matches)} match(es)[/green]\n")
    
    for i, match in enumerate(matches, 1):
        page_range = format_page_range(match.page_start, match.page_end)
        
        # Create header
        header = (
            f"[bold]#{i}[/bold] "
            f"[cyan]{match.chunk_id}[/cyan] "
            f"[dim]({page_range})[/dim] "
            f"[yellow]Score: {match.score:.1f}[/yellow]"
        )
        
        # Get text to display
        if args.full:
            display_text = match.text
        else:
            display_text = get_preview(match.text)
        
        # Create panel with chunk content
        panel = Panel(
            display_text,
            title=header,
            title_align="left",
            border_style="blue",
        )
        
        console.print(panel)
        console.print()
    
    return 0


if __name__ == "__main__":
    exit(main())
