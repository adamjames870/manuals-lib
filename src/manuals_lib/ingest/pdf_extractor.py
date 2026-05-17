"""PDF text extraction functionality."""

from pathlib import Path

import pymupdf

from manuals_lib.ingest.models import PageContent


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
