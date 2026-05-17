#!/usr/bin/env python3
"""Build embedding index from chunks."""

import argparse
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from manuals_lib.embeddings import build_index


def main():
    """Build embeddings from command line."""
    parser = argparse.ArgumentParser(
        description="Build embedding index from chunks"
    )
    parser.add_argument(
        "--chunks",
        required=True,
        help="Path to chunks JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="data/index",
        help="Output directory for index files (default: data/index)",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (default: all-MiniLM-L6-v2)",
    )
    
    args = parser.parse_args()
    console = Console()
    
    chunks_path = Path(args.chunks)
    output_dir = Path(args.output_dir)
    
    if not chunks_path.exists():
        console.print(f"[red]Error:[/red] Chunks file not found: {chunks_path}")
        return 1
    
    console.print(f"\n[blue]Building embedding index[/blue]")
    console.print(f"[dim]Chunks:[/dim] {chunks_path}")
    console.print(f"[dim]Model:[/dim] {args.model}")
    console.print(f"[dim]Output:[/dim] {output_dir}\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading model and generating embeddings...", total=None)
        
        try:
            index_dir, manifest = build_index(
                chunks_path=chunks_path,
                output_dir=output_dir,
                model_name=args.model,
            )
            
            progress.update(task, description="[green]✓[/green] Index built successfully")
            
            console.print(f"\n[green]Index created:[/green] {index_dir}")
            console.print(f"  → manifest.json")
            console.print(f"  → chunks.json")
            console.print(f"  → embeddings.npy")
            console.print(f"\n[blue]Statistics:[/blue]")
            console.print(f"  Chunks: {manifest.chunk_count}")
            console.print(f"  Embedding dimension: {manifest.embedding_dim}")
            console.print(f"  Model: {manifest.model_name}")
            console.print(f"  Created: {manifest.created_at}")
            
        except FileNotFoundError as e:
            progress.update(task, description="[red]✗[/red] File not found")
            console.print(f"\n[red]Error:[/red] {e}")
            return 1
        except ValueError as e:
            progress.update(task, description="[red]✗[/red] Invalid input")
            console.print(f"\n[red]Error:[/red] {e}")
            return 1
        except Exception as e:
            progress.update(task, description="[red]✗[/red] Failed")
            console.print(f"\n[red]Error:[/red] {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
