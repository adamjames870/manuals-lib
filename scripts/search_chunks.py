#!/usr/bin/env python3
"""Search chunks using keyword matching."""

import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

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


def get_preview(text: str, first_match_pos: int = -1, max_length: int = 300) -> str:
    """Get a preview of chunk text centered around the first match.
    
    Args:
        text: Full chunk text
        first_match_pos: Position of first match (-1 if unknown)
        max_length: Maximum preview length
        
    Returns:
        Preview string with ellipsis if truncated
    """
    if len(text) <= max_length:
        return text
    
    # If we know where the match is, center preview around it
    if first_match_pos >= 0:
        # Try to center the preview around the match
        start = max(0, first_match_pos - max_length // 3)
        end = min(len(text), start + max_length)
        
        # Adjust start if we're at the end
        if end == len(text):
            start = max(0, end - max_length)
        
        preview = text[start:end]
        
        # Add ellipsis if truncated
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        
        # Try to break at word boundaries
        if prefix:
            first_space = preview.find(' ')
            if first_space > 0 and first_space < 50:
                preview = preview[first_space + 1:]
        
        if suffix:
            last_space = preview.rfind(' ')
            if last_space > len(preview) - 50:
                preview = preview[:last_space]
        
        return prefix + preview + suffix
    
    # Fall back to start of text
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
        help="Path to chunks JSON file (if not specified, searches all files in data/processed/)",
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
    
    # Determine which chunks files to load
    if args.chunks:
        # Load specific file
        chunks_files = [Path(args.chunks)]
    else:
        # Find all chunks files in data/processed/
        processed_dir = Path("data/processed")
        if not processed_dir.exists():
            console.print(f"[red]Error:[/red] Directory not found: {processed_dir}")
            console.print("[yellow]Hint:[/yellow] Run batch_extract.py first to process PDFs")
            return 1
        
        chunks_files = list(processed_dir.glob("*.chunks.json"))
        
        if not chunks_files:
            console.print(f"[yellow]No .chunks.json files found in {processed_dir}[/yellow]")
            console.print("[yellow]Hint:[/yellow] Run batch_extract.py first to process PDFs")
            return 0
    
    # Load chunks from all files
    all_chunks = []
    loaded_files = []
    
    for chunks_file in chunks_files:
        try:
            file_chunks = load_chunks_from_json(chunks_file)
            all_chunks.extend(file_chunks)
            loaded_files.append(chunks_file.name)
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            return 1
        except Exception as e:
            console.print(f"[red]Error loading {chunks_file}:[/red] {e}")
            return 1
    
    if not all_chunks:
        console.print("[yellow]No chunks found[/yellow]")
        return 0
    
    # Display search info
    if len(loaded_files) == 1:
        console.print(f"\n[blue]Searching {len(all_chunks)} chunks from {loaded_files[0]} for:[/blue] {args.query}\n")
    else:
        console.print(f"\n[blue]Searching {len(all_chunks)} chunks from {len(loaded_files)} files for:[/blue] {args.query}")
        console.print(f"[dim]Files: {', '.join(loaded_files)}[/dim]\n")
    
    # Search
    matches = search_chunks(all_chunks, args.query, args.top_k)
    
    if not matches:
        console.print("[yellow]No matches found[/yellow]")
        return 0
    
    # Display results
    console.print(f"[green]Found {len(matches)} match(es)[/green]\n")
    
    for i, match in enumerate(matches, 1):
        page_range = format_page_range(match.page_start, match.page_end)
        
        # Format matched terms
        match_type = "[green]Exact phrase[/green]" if match.exact_phrase_match else "[yellow]Keywords[/yellow]"
        matched_terms_str = ", ".join(f"'{term}'" for term in match.matched_terms)
        
        # Create header
        header = (
            f"[bold]#{i}[/bold] "
            f"[cyan]{match.chunk_id}[/cyan] "
            f"[dim]({page_range})[/dim] "
            f"[yellow]Score: {match.score:.1f}[/yellow]"
        )
        
        # Create subtitle with match info
        subtitle = f"{match_type} | Matched: {matched_terms_str}"
        
        # Get text to display
        if args.full:
            display_text = match.text
        else:
            display_text = get_preview(match.text, match.first_match_pos)
        
        # Create panel with chunk content
        panel = Panel(
            display_text,
            title=header,
            subtitle=subtitle,
            title_align="left",
            subtitle_align="left",
            border_style="blue",
        )
        
        console.print(panel)
        console.print()
    
    return 0


if __name__ == "__main__":
    exit(main())
