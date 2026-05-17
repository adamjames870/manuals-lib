#!/usr/bin/env python3
"""Batch extract text from PDFs and save to JSON."""

import json
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from manuals_lib.ingest import chunk_pages, extract_pdf, extract_pdf_blocks, normalize_pages


def get_output_path(pdf_path: Path, output_dir: Path) -> Path:
    """Generate output path for extracted JSON file.
    
    Args:
        pdf_path: Path to the source PDF file
        output_dir: Directory to write output files
        
    Returns:
        Path to the output JSON file
    """
    stem = pdf_path.stem
    return output_dir / f"{stem}.extracted.json"


def get_report_path(pdf_path: Path, output_dir: Path) -> Path:
    """Generate output path for extraction report.
    
    Args:
        pdf_path: Path to the source PDF file
        output_dir: Directory to write output files
        
    Returns:
        Path to the output markdown report file
    """
    stem = pdf_path.stem
    return output_dir / f"{stem}.report.md"


def get_blocks_path(pdf_path: Path, output_dir: Path) -> Path:
    """Generate output path for blocks JSON file.
    
    Args:
        pdf_path: Path to the source PDF file
        output_dir: Directory to write output files
        
    Returns:
        Path to the output blocks JSON file
    """
    stem = pdf_path.stem
    return output_dir / f"{stem}.blocks.json"


def get_normalized_path(pdf_path: Path, output_dir: Path) -> Path:
    """Generate output path for normalized JSON file.
    
    Args:
        pdf_path: Path to the source PDF file
        output_dir: Directory to write output files
        
    Returns:
        Path to the output normalized JSON file
    """
    stem = pdf_path.stem
    return output_dir / f"{stem}.normalized.json"


def get_chunks_path(pdf_path: Path, output_dir: Path) -> Path:
    """Generate output path for chunks JSON file.
    
    Args:
        pdf_path: Path to the source PDF file
        output_dir: Directory to write output files
        
    Returns:
        Path to the output chunks JSON file
    """
    stem = pdf_path.stem
    return output_dir / f"{stem}.chunks.json"


def pdf_already_extracted(pdf_path: Path, output_dir: Path) -> bool:
    """Check if a PDF has already been extracted.
    
    Args:
        pdf_path: Path to the source PDF file
        output_dir: Directory containing output files
        
    Returns:
        True if the extracted JSON file exists
    """
    output_path = get_output_path(pdf_path, output_dir)
    return output_path.exists()


def generate_report(pdf_path: Path, pages: list, report_path: Path) -> None:
    """Generate a markdown report with extraction statistics.
    
    Args:
        pdf_path: Path to the source PDF file
        pages: List of PageContent objects
        report_path: Path to write the markdown report
    """
    # Calculate statistics
    char_counts = [len(page.text) for page in pages]
    total_chars = sum(char_counts)
    empty_pages = [i + 1 for i, count in enumerate(char_counts) if count == 0]
    short_pages = [i + 1 for i, count in enumerate(char_counts) if 0 < count < 100]
    
    # Generate markdown report
    report = f"""# Extraction Report: {pdf_path.name}

## Summary

- **Source**: {pdf_path.name}
- **Page Count**: {len(pages)}
- **Total Characters**: {total_chars:,}
- **Average Characters per Page**: {total_chars / len(pages):.0f}

## Page Statistics

| Page | Characters |
|------|------------|
"""
    
    for page_num, char_count in enumerate(char_counts, start=1):
        report += f"| {page_num} | {char_count:,} |\n"
    
    # Add warnings section if there are issues
    if empty_pages or short_pages:
        report += "\n## Warnings\n\n"
        
        if empty_pages:
            empty_list = ', '.join(map(str, empty_pages))
            report += f"- **Empty Pages** ({len(empty_pages)}): {empty_list}\n"
        
        if short_pages:
            short_list = ', '.join(map(str, short_pages))
            report += (
                f"- **Suspiciously Short Pages** ({len(short_pages)}, "
                f"<100 chars): {short_list}\n"
            )
    
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)


def extract_to_json(
    pdf_path: Path,
    output_path: Path,
    report_path: Path,
    blocks_path: Path,
    normalized_path: Path,
    chunks_path: Path,
    skip_normalization: bool = False,
    skip_chunking: bool = False,
) -> None:
    """Extract PDF content and write to JSON file with report.
    
    Args:
        pdf_path: Path to the source PDF file
        output_path: Path to write the JSON output
        report_path: Path to write the markdown report
        blocks_path: Path to write the blocks JSON output
        normalized_path: Path to write the normalized JSON output
        chunks_path: Path to write the chunks JSON output
        skip_normalization: Whether to skip normalization step
        skip_chunking: Whether to skip chunking step
    """
    pages = extract_pdf(pdf_path)
    
    output_data = {
        "source": pdf_path.name,
        "page_count": len(pages),
        "pages": [
            {
                "page_number": page.page_number,
                "text": page.text,
            }
            for page in pages
        ],
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    generate_report(pdf_path, pages, report_path)
    
    # Extract and save blocks
    blocks = extract_pdf_blocks(pdf_path)
    
    blocks_data = {
        "source": pdf_path.name,
        "total_blocks": len(blocks),
        "blocks": [
            {
                "page_number": block.page_number,
                "block_number": block.block_number,
                "x0": round(block.bbox.x0, 2),
                "y0": round(block.bbox.y0, 2),
                "x1": round(block.bbox.x1, 2),
                "y1": round(block.bbox.y1, 2),
                "text": block.text,
            }
            for block in blocks
        ],
    }
    
    with open(blocks_path, "w", encoding="utf-8") as f:
        json.dump(blocks_data, f, indent=2, ensure_ascii=False)
    
    # Normalize and save normalized content
    if not skip_normalization:
        normalized_pages = normalize_pages(pages, join_wrapped=True)
        
        normalized_data = {
            "source": pdf_path.name,
            "page_count": len(normalized_pages),
            "pages": [
                {
                    "page_number": page.page_number,
                    "text": page.text,
                }
                for page in normalized_pages
            ],
        }
        
        with open(normalized_path, "w", encoding="utf-8") as f:
            json.dump(normalized_data, f, indent=2, ensure_ascii=False)
        
        # Chunk and save chunks
        if not skip_chunking:
            chunks = chunk_pages(normalized_pages)
            
            chunks_data = {
                "source": pdf_path.name,
                "total_chunks": len(chunks),
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "char_count": chunk.char_count,
                        "text": chunk.text,
                    }
                    for chunk in chunks
                ],
            }
            
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)


def main():
    """Process all PDFs in data/raw/ and extract to data/processed/."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch extract PDFs to JSON")
    parser.add_argument(
        "--skip-normalization",
        action="store_true",
        help="Skip text normalization step",
    )
    parser.add_argument(
        "--skip-chunking",
        action="store_true",
        help="Skip text chunking step",
    )
    args = parser.parse_args()
    
    console = Console()
    
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    
    if not raw_dir.exists():
        console.print(f"[red]Error:[/red] Directory not found: {raw_dir}")
        console.print("[yellow]Create it with:[/yellow] mkdir -p data/raw")
        return
    
    # Find all PDF files
    pdf_files = list(raw_dir.glob("*.pdf"))
    
    if not pdf_files:
        console.print(f"[yellow]No PDF files found in {raw_dir}[/yellow]")
        return
    
    # Filter out already processed files
    to_process = [
        pdf for pdf in pdf_files 
        if not pdf_already_extracted(pdf, processed_dir)
    ]
    
    if not to_process:
        console.print("[green]All PDFs have already been extracted![/green]")
        return
    
    console.print(f"[blue]Found {len(pdf_files)} PDF(s), {len(to_process)} to process[/blue]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for pdf_path in to_process:
            task = progress.add_task(f"Processing {pdf_path.name}...", total=None)
            
            try:
                output_path = get_output_path(pdf_path, processed_dir)
                report_path = get_report_path(pdf_path, processed_dir)
                blocks_path = get_blocks_path(pdf_path, processed_dir)
                normalized_path = get_normalized_path(pdf_path, processed_dir)
                chunks_path = get_chunks_path(pdf_path, processed_dir)
                extract_to_json(
                    pdf_path,
                    output_path,
                    report_path,
                    blocks_path,
                    normalized_path,
                    chunks_path,
                    skip_normalization=args.skip_normalization,
                    skip_chunking=args.skip_chunking,
                )
                progress.update(task, description=f"[green]✓[/green] {pdf_path.name}")
                console.print(f"  → {output_path}")
                console.print(f"  → {report_path}")
                console.print(f"  → {blocks_path}")
                if not args.skip_normalization:
                    console.print(f"  → {normalized_path}")
                    if not args.skip_chunking:
                        console.print(f"  → {chunks_path}")
            except Exception as e:
                progress.update(task, description=f"[red]✗[/red] {pdf_path.name}")
                console.print(f"  [red]Error:[/red] {e}")
    
    console.print(f"\n[green]Processed {len(to_process)} PDF(s)[/green]")


if __name__ == "__main__":
    main()
