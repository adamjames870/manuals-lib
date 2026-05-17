"""PDF text extraction functionality."""

from pathlib import Path

import pymupdf

from manuals_lib.ingest.models import BoundingBox, PageContent, TextBlock


def extract_pdf(pdf_path: str | Path) -> list[PageContent]:
    """Extract text content from a PDF file page by page.
    
    Args:
        pdf_path: Path to the PDF file to extract
        
    Returns:
        List of PageContent objects, one per page
        
    Raises:
        FileNotFoundError: If the PDF file does not exist
        pymupdf.FileDataError: If the file is not a valid PDF
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    pages = []
    
    with pymupdf.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            pages.append(
                PageContent(
                    source=pdf_path.name,
                    page_number=page_num,
                    text=text,
                )
            )
    
    return pages


def extract_pdf_blocks(pdf_path: str | Path) -> list[TextBlock]:
    """Extract text blocks from a PDF file with layout information.
    
    Args:
        pdf_path: Path to the PDF file to extract
        
    Returns:
        List of TextBlock objects with bounding box coordinates
        
    Raises:
        FileNotFoundError: If the PDF file does not exist
        pymupdf.FileDataError: If the file is not a valid PDF
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    blocks = []
    
    with pymupdf.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            # Get text blocks: returns list of (x0, y0, x1, y1, "text", block_no, block_type)
            page_blocks = page.get_text("blocks")
            
            for block_num, block in enumerate(page_blocks):
                # block format: (x0, y0, x1, y1, "text", block_no, block_type)
                x0, y0, x1, y1, text, _, _ = block
                
                # Skip empty blocks
                if not text or not text.strip():
                    continue
                
                blocks.append(
                    TextBlock(
                        source=pdf_path.name,
                        page_number=page_num,
                        block_number=block_num,
                        bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
                        text=text.strip(),
                    )
                )
    
    return blocks
