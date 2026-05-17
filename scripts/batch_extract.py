#!/usr/bin/env python3
"""Batch extract text from PDFs and save to JSON."""

import json
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from manuals_lib.ingest import extract_pdf


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


def extract_to_json(pdf_path: Path, output_path: Path, report_path: Path) -> None:
    """Extract PDF content and write to JSON file with report.
    
    Args:
        pdf_path: Path to the source PDF file
        output_path: Path to write the JSON output
        report_path: Path to write the markdown report
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


def main():
    """Process all PDFs in data/raw/ and extract to data/processed/."""
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
                extract_to_json(pdf_path, output_path, report_path)
                progress.update(task, description=f"[green]✓[/green] {pdf_path.name}")
                console.print(f"  → {output_path}")
                console.print(f"  → {report_path}")
            except Exception as e:
                progress.update(task, description=f"[red]✗[/red] {pdf_path.name}")
                console.print(f"  [red]Error:[/red] {e}")
    
    console.print(f"\n[green]Processed {len(to_process)} PDF(s)[/green]")


if __name__ == "__main__":
    main()
