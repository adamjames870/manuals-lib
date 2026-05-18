#!/usr/bin/env python3
"""Batch extract text from PDFs and save to JSON."""

import json
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from manuals_lib.embeddings import build_index
from manuals_lib.ingest import (
    chunk_pages,
    chunk_tables,
    extract_pdf,
    extract_pdf_blocks,
    extract_pdf_tables,
    normalize_pages,
)


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


def get_tables_path(pdf_path: Path, output_dir: Path) -> Path:
    """Generate output path for tables JSON file.
    
    Args:
        pdf_path: Path to the source PDF file
        output_dir: Directory to write output files
        
    Returns:
        Path to the output tables JSON file
    """
    stem = pdf_path.stem
    return output_dir / f"{stem}.tables.json"


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


def generate_report(
    pdf_path: Path,
    pages: list,
    report_path: Path,
    chunks: list | None = None,
    tables: list | None = None
) -> None:
    """Generate a markdown report with extraction statistics.
    
    Args:
        pdf_path: Path to the source PDF file
        pages: List of PageContent objects
        report_path: Path to write the markdown report
        chunks: Optional list of Chunk objects
        tables: Optional list of TableData objects
    """
    # Calculate statistics
    char_counts = [len(page.text) for page in pages]
    total_chars = sum(char_counts)
    empty_pages = [i + 1 for i, count in enumerate(char_counts) if count == 0]
    short_pages = [i + 1 for i, count in enumerate(char_counts) if 0 < count < 100]
    
    # Count OCR pages
    ocr_pages = [
        page.page_number for page in pages 
        if page.extraction_method == "ocr"
    ]
    native_pages = [
        page.page_number for page in pages 
        if page.extraction_method == "pymupdf"
    ]
    
    # Generate markdown report
    report = f"""# Extraction Report: {pdf_path.name}

## Summary

- **Source**: {pdf_path.name}
- **Page Count**: {len(pages)}
- **Total Characters**: {total_chars:,}
- **Average Characters per Page**: {total_chars / len(pages):.0f}

## Extraction Summary

- **Native Extraction Pages**: {len(native_pages)}
- **OCR Pages**: {len(ocr_pages)}
- **Empty Pages**: {len(empty_pages)}
- **Suspiciously Short Pages** (<100 chars): {len(short_pages)}

## Page Statistics

| Page | Characters | Method |
|------|------------|--------|
"""
    
    for page in pages:
        report += f"| {page.page_number} | {len(page.text):,} | {page.extraction_method} |\n"
    
    # Add table statistics if available
    if tables:
        table_pages = sorted(set(table.page_number for table in tables))
        ocr_table_pages = [
            page_num for page_num in table_pages
            if any(p.page_number == page_num and p.extraction_method == "ocr" for p in pages)
        ]
        
        report += "\n## Table Summary\n\n"
        report += f"- **Tables Detected**: {len(tables)}\n"
        report += f"- **Pages with Tables**: {len(table_pages)}\n"
        report += f"- **OCR Pages** (table extraction skipped): {len(ocr_table_pages)}\n"
        
        if table_pages:
            pages_list = ', '.join(map(str, table_pages[:20]))
            if len(table_pages) > 20:
                pages_list += f", ... ({len(table_pages) - 20} more)"
            report += f"\n  Table pages: {pages_list}\n"
        
        report += "\n### Table Details\n\n"
        report += "| Table ID | Page | Rows | Has Headers |\n"
        report += "|----------|------|------|-------------|\n"
        for table in tables:
            has_headers = "Yes" if table.headers else "No"
            report += (
                f"| {table.table_id} | {table.page_number} | "
                f"{len(table.rows)} | {has_headers} |\n"
            )
    
    # Add chunk statistics if available
    if chunks:
        chunk_sizes = [chunk.char_count for chunk in chunks]
        min_chunk = min(chunk_sizes)
        max_chunk = max(chunk_sizes)
        avg_chunk = sum(chunk_sizes) / len(chunk_sizes)
        
        # Count by chunk type
        chunk_types = {}
        for chunk in chunks:
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
        
        # Count small and large chunks
        small_chunks = [i for i, size in enumerate(chunk_sizes, 1) if size < 500]
        large_chunks = [i for i, size in enumerate(chunk_sizes, 1) if size > 2000]
        
        # Count cross-page chunks
        cross_page_chunks = [
            i for i, chunk in enumerate(chunks, 1)
            if chunk.page_end > chunk.page_start
        ]
        
        # Count excluded chunks
        excluded_chunks = [
            chunk for chunk in chunks
            if chunk.chunk_type in ('toc', 'front_matter', 'index')
        ]
        
        # Estimate duplicate overlap
        total_chunk_chars = sum(chunk_sizes)
        total_page_chars = sum(char_counts)
        overlap_estimate = total_chunk_chars - total_page_chars
        overlap_pct = (overlap_estimate / total_page_chars * 100) if total_page_chars > 0 else 0
        
        report += "\n## Chunk Statistics\n\n"
        report += f"- **Total Chunks**: {len(chunks)}\n"
        report += f"- **Min Chunk Size**: {min_chunk:,} characters\n"
        report += f"- **Max Chunk Size**: {max_chunk:,} characters\n"
        report += f"- **Average Chunk Size**: {avg_chunk:.0f} characters\n"
        report += f"- **Small Chunks** (<500 chars): {len(small_chunks)}\n"
        report += f"- **Large Chunks** (>2000 chars): {len(large_chunks)}\n"
        report += f"- **Cross-Page Chunks**: {len(cross_page_chunks)}\n"
        report += f"- **Estimated Overlap**: {overlap_estimate:,} characters ({overlap_pct:.1f}%)\n"
        
        report += "\n### Chunks by Type\n\n"
        for chunk_type, count in sorted(chunk_types.items()):
            report += f"- **{chunk_type}**: {count}\n"
        
        report += "\n### Retrieval Diagnostics\n\n"
        report += f"- **Chunks Excluded from Default Retrieval**: {len(excluded_chunks)}\n"
        if excluded_chunks:
            excluded_ids = [chunk.chunk_id for chunk in excluded_chunks[:10]]
            excluded_list = ', '.join(excluded_ids)
            if len(excluded_chunks) > 10:
                excluded_list += f", ... ({len(excluded_chunks) - 10} more)"
            report += f"\n  Excluded chunk IDs: {excluded_list}\n"
        
        if small_chunks:
            small_list = ', '.join(map(str, small_chunks[:10]))
            if len(small_chunks) > 10:
                small_list += f", ... ({len(small_chunks) - 10} more)"
            report += f"\n  Small chunk IDs: {small_list}\n"
        
        if large_chunks:
            large_list = ', '.join(map(str, large_chunks[:10]))
            if len(large_chunks) > 10:
                large_list += f", ... ({len(large_chunks) - 10} more)"
            report += f"\n  Large chunk IDs: {large_list}\n"
    
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
    tables_path: Path,
    skip_normalization: bool = False,
    skip_chunking: bool = False,
    skip_embeddings: bool = False,
) -> None:
    """Extract PDF content and write to JSON file with report.
    
    Args:
        pdf_path: Path to the source PDF file
        output_path: Path to write the JSON output
        report_path: Path to write the markdown report
        blocks_path: Path to write the blocks JSON output
        normalized_path: Path to write the normalized JSON output
        chunks_path: Path to write the chunks JSON output
        tables_path: Path to write the tables JSON output
        skip_normalization: Whether to skip normalization step
        skip_chunking: Whether to skip chunking step
        skip_embeddings: Whether to skip embeddings generation
    """
    pages = extract_pdf(pdf_path)
    
    # Extract tables
    tables = extract_pdf_tables(pdf_path)
    
    output_data = {
        "source": pdf_path.name,
        "page_count": len(pages),
        "pages": [
            {
                "page_number": page.page_number,
                "text": page.text,
                "extraction_method": page.extraction_method,
                "ocr_engine": page.ocr_engine,
                "ocr_trigger_reason": page.ocr_trigger_reason,
            }
            for page in pages
        ],
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
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
    
    # Save tables
    tables_data = {
        "source": pdf_path.name,
        "total_tables": len(tables),
        "tables": [
            {
                "table_id": table.table_id,
                "page_number": table.page_number,
                "extraction_method": table.extraction_method,
                "bbox": {
                    "x0": round(table.bbox.x0, 2),
                    "y0": round(table.bbox.y0, 2),
                    "x1": round(table.bbox.x1, 2),
                    "y1": round(table.bbox.y1, 2),
                } if table.bbox else None,
                "headers": table.headers,
                "row_count": len(table.rows),
                "rows": table.rows,
            }
            for table in tables
        ],
    }
    
    with open(tables_path, "w", encoding="utf-8") as f:
        json.dump(tables_data, f, indent=2, ensure_ascii=False)
    
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
                    "extraction_method": page.extraction_method,
                    "ocr_engine": page.ocr_engine,
                    "ocr_trigger_reason": page.ocr_trigger_reason,
                }
                for page in normalized_pages
            ],
        }
        
        with open(normalized_path, "w", encoding="utf-8") as f:
            json.dump(normalized_data, f, indent=2, ensure_ascii=False)
        
        # Chunk and save chunks
        if not skip_chunking:
            # Chunk pages
            page_chunks = chunk_pages(normalized_pages)
            
            # Chunk tables
            table_chunks = chunk_tables(tables)
            
            # Combine all chunks
            chunks = page_chunks + table_chunks
            
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
                        "chunk_type": chunk.chunk_type,
                        "extraction_method": chunk.extraction_method,
                        "table_id": chunk.table_id,
                        "section_title": chunk.section_title,
                    }
                    for chunk in chunks
                ],
            }
            
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
            # Build embeddings if not skipped
            if not skip_embeddings:
                index_dir = Path("data/index")
                build_index(
                    chunks_path=chunks_path,
                    output_dir=index_dir,
                )
            
            # Generate report with chunk and table statistics
            generate_report(pdf_path, pages, report_path, chunks, tables)
        else:
            # Generate report without chunk statistics
            generate_report(pdf_path, pages, report_path, None, tables)
    else:
        # Generate report without normalization or chunking
        generate_report(pdf_path, pages, report_path, None, tables)


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
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embeddings generation step",
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
                tables_path = get_tables_path(pdf_path, processed_dir)
                extract_to_json(
                    pdf_path,
                    output_path,
                    report_path,
                    blocks_path,
                    normalized_path,
                    chunks_path,
                    tables_path,
                    skip_normalization=args.skip_normalization,
                    skip_chunking=args.skip_chunking,
                    skip_embeddings=args.skip_embeddings,
                )
                progress.update(task, description=f"[green]✓[/green] {pdf_path.name}")
                console.print(f"  → {output_path}")
                console.print(f"  → {report_path}")
                console.print(f"  → {blocks_path}")
                console.print(f"  → {tables_path}")
                if not args.skip_normalization:
                    console.print(f"  → {normalized_path}")
                    if not args.skip_chunking:
                        console.print(f"  → {chunks_path}")
                        if not args.skip_embeddings:
                            index_dir = Path("data/index") / pdf_path.stem
                            console.print(f"  → {index_dir}/")
            except Exception as e:
                progress.update(task, description=f"[red]✗[/red] {pdf_path.name}")
                console.print(f"  [red]Error:[/red] {e}")
    
    console.print(f"\n[green]Processed {len(to_process)} PDF(s)[/green]")


if __name__ == "__main__":
    main()
