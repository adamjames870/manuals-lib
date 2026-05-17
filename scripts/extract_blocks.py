#!/usr/bin/env python3
"""Extract text blocks from a PDF file with layout information."""

import json
import sys
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

from manuals_lib.ingest import extract_pdf_blocks


def analyze_layout(blocks: list) -> dict:
    """Analyze layout patterns in extracted blocks.
    
    Args:
        blocks: List of TextBlock objects
        
    Returns:
        Dictionary with layout analysis results
    """
    # Group blocks by page
    pages = {}
    for block in blocks:
        if block.page_number not in pages:
            pages[block.page_number] = []
        pages[block.page_number].append(block)
    
    # Count blocks per page
    blocks_per_page = {page_num: len(page_blocks) for page_num, page_blocks in pages.items()}
    
    # Find empty/short blocks
    short_blocks = [
        (block.page_number, block.block_number, len(block.text))
        for block in blocks
        if len(block.text) < 50
    ]
    
    # Find repeated text across pages (potential headers/footers)
    text_occurrences = Counter(block.text for block in blocks)
    repeated_texts = {
        text: count for text, count in text_occurrences.items() if count > 1
    }
    
    # Identify likely headers/footers (repeated text in similar positions)
    potential_headers = []
    potential_footers = []
    
    for text, count in repeated_texts.items():
        matching_blocks = [b for b in blocks if b.text == text]
        
        # Check if blocks are in similar vertical positions
        y_positions = [b.bbox.y0 for b in matching_blocks]
        avg_y = sum(y_positions) / len(y_positions)
        
        # Headers typically at top (low y), footers at bottom (high y)
        if avg_y < 100:  # Approximate top of page
            potential_headers.append((text[:50], count, avg_y))
        elif avg_y > 700:  # Approximate bottom of page
            potential_footers.append((text[:50], count, avg_y))
    
    return {
        "total_blocks": len(blocks),
        "total_pages": len(pages),
        "blocks_per_page": blocks_per_page,
        "short_blocks": short_blocks,
        "repeated_texts": repeated_texts,
        "potential_headers": potential_headers,
        "potential_footers": potential_footers,
    }


def generate_layout_report(pdf_path: Path, blocks: list, analysis: dict) -> str:
    """Generate a markdown layout inspection report.
    
    Args:
        pdf_path: Path to the source PDF
        blocks: List of TextBlock objects
        analysis: Layout analysis results
        
    Returns:
        Markdown formatted report string
    """
    report = f"""# Layout Inspection Report: {pdf_path.name}

## Summary

- **Total Blocks**: {analysis['total_blocks']}
- **Total Pages**: {analysis['total_pages']}
- **Average Blocks per Page**: {analysis['total_blocks'] / analysis['total_pages']:.1f}

## Blocks per Page

| Page | Block Count |
|------|-------------|
"""
    
    for page_num, count in sorted(analysis['blocks_per_page'].items()):
        report += f"| {page_num} | {count} |\n"
    
    # Short blocks section
    if analysis['short_blocks']:
        report += f"\n## Short Blocks (<50 characters)\n\n"
        report += f"Found {len(analysis['short_blocks'])} short blocks:\n\n"
        report += "| Page | Block | Length |\n"
        report += "|------|-------|--------|\n"
        for page_num, block_num, length in analysis['short_blocks'][:20]:
            report += f"| {page_num} | {block_num} | {length} |\n"
        if len(analysis['short_blocks']) > 20:
            report += f"\n... and {len(analysis['short_blocks']) - 20} more\n"
    
    # Repeated text section
    if analysis['repeated_texts']:
        report += f"\n## Repeated Text Across Pages\n\n"
        report += f"Found {len(analysis['repeated_texts'])} repeated text patterns:\n\n"
        report += "| Text (first 50 chars) | Occurrences |\n"
        report += "|----------------------|-------------|\n"
        for text, count in sorted(
            analysis['repeated_texts'].items(), key=lambda x: x[1], reverse=True
        )[:10]:
            text_preview = text[:50].replace('\n', ' ')
            report += f"| {text_preview} | {count} |\n"
    
    # Headers/footers section
    if analysis['potential_headers']:
        report += f"\n## Potential Headers\n\n"
        report += "| Text (first 50 chars) | Occurrences | Avg Y Position |\n"
        report += "|----------------------|-------------|----------------|\n"
        for text, count, avg_y in analysis['potential_headers'][:5]:
            text_preview = text.replace('\n', ' ')
            report += f"| {text_preview} | {count} | {avg_y:.1f} |\n"
    
    if analysis['potential_footers']:
        report += f"\n## Potential Footers\n\n"
        report += "| Text (first 50 chars) | Occurrences | Avg Y Position |\n"
        report += "|----------------------|-------------|----------------|\n"
        for text, count, avg_y in analysis['potential_footers'][:5]:
            text_preview = text.replace('\n', ' ')
            report += f"| {text_preview} | {count} | {avg_y:.1f} |\n"
    
    return report


def main():
    """Extract blocks from a PDF and generate layout report."""
    console = Console()
    
    if len(sys.argv) != 2:
        console.print("[red]Usage: python scripts/extract_blocks.py <pdf_file>[/red]")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    
    try:
        console.print(f"[blue]Extracting blocks from:[/blue] {pdf_path}")
        blocks = extract_pdf_blocks(pdf_path)
        
        console.print(f"[green]Successfully extracted {len(blocks)} blocks[/green]\n")
        
        # Analyze layout
        analysis = analyze_layout(blocks)
        
        # Display summary table
        table = Table(title="Block Extraction Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Blocks", str(analysis['total_blocks']))
        table.add_row("Total Pages", str(analysis['total_pages']))
        table.add_row(
            "Avg Blocks/Page",
            f"{analysis['total_blocks'] / analysis['total_pages']:.1f}"
        )
        table.add_row("Short Blocks", str(len(analysis['short_blocks'])))
        table.add_row("Repeated Texts", str(len(analysis['repeated_texts'])))
        table.add_row("Potential Headers", str(len(analysis['potential_headers'])))
        table.add_row("Potential Footers", str(len(analysis['potential_footers'])))
        
        console.print(table)
        
        # Save JSON output
        output_json = pdf_path.with_suffix('.blocks.json')
        blocks_data = {
            "source": pdf_path.name,
            "total_blocks": len(blocks),
            "blocks": [
                {
                    "page_number": block.page_number,
                    "block_number": block.block_number,
                    "bbox": {
                        "x0": block.bbox.x0,
                        "y0": block.bbox.y0,
                        "x1": block.bbox.x1,
                        "y1": block.bbox.y1,
                    },
                    "text": block.text,
                }
                for block in blocks
            ],
        }
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(blocks_data, f, indent=2, ensure_ascii=False)
        
        console.print(f"\n[green]Saved JSON to:[/green] {output_json}")
        
        # Save layout report
        output_report = pdf_path.with_suffix('.layout.md')
        report = generate_layout_report(pdf_path, blocks, analysis)
        
        with open(output_report, 'w', encoding='utf-8') as f:
            f.write(report)
        
        console.print(f"[green]Saved layout report to:[/green] {output_report}")
        
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error extracting blocks:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
