#!/usr/bin/env python3
"""Test OCR extraction on a specific PDF file."""

import logging
import sys
from pathlib import Path

from rich.console import Console

from manuals_lib.ingest import extract_pdf

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    """Test OCR extraction on a PDF file."""
    console = Console()
    
    if len(sys.argv) != 2:
        console.print("[red]Usage: python scripts/test_ocr.py <pdf_file>[/red]")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    
    if not pdf_path.exists():
        console.print(f"[red]Error:[/red] File not found: {pdf_path}")
        sys.exit(1)
    
    console.print(f"[blue]Testing OCR extraction on:[/blue] {pdf_path}\n")
    
    try:
        # Extract with OCR fallback enabled
        console.print("[yellow]Extracting with OCR fallback enabled...[/yellow]")
        pages = extract_pdf(pdf_path, use_ocr_fallback=True)
        
        console.print(f"\n[green]Extracted {len(pages)} pages[/green]\n")
        
        for page in pages:
            text_len = len(page.text.strip())
            console.print(
                f"[cyan]Page {page.page_number}:[/cyan] {text_len} characters"
            )
            if text_len > 0:
                preview = page.text.strip()[:200].replace('\n', ' ')
                console.print(f"  Preview: {preview}...")
            else:
                console.print("  [red]WARNING: No text extracted![/red]")
        
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
