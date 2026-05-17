#!/usr/bin/env python3
"""Simple script to extract text from a PDF file."""

import sys
from pathlib import Path

from rich.console import Console

from manuals_lib.ingest import extract_pdf


def main():
    """Extract and display text from a PDF file."""
    console = Console()
    
    if len(sys.argv) != 2:
        console.print("[red]Usage: python scripts/extract_pdf.py <pdf_file>[/red]")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    
    try:
        console.print(f"[blue]Extracting text from:[/blue] {pdf_path}")
        pages = extract_pdf(pdf_path)
        
        console.print(f"[green]Successfully extracted {len(pages)} pages[/green]\n")
        
        for page in pages:
            console.print(f"[yellow]--- Page {page.page_number} ---[/yellow]")
            console.print(page.text[:500])  # Show first 500 chars
            if len(page.text) > 500:
                console.print("[dim]...[/dim]")
            console.print()
            
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error extracting PDF:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
