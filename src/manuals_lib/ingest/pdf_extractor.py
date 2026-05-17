"""PDF text extraction functionality."""

from pathlib import Path

import pymupdf

from manuals_lib.ingest.models import BoundingBox, PageContent, TextBlock
from manuals_lib.ingest.ocr_extractor import extract_text_with_ocr, should_use_ocr


def extract_pdf(pdf_path: str | Path, use_ocr_fallback: bool = True) -> list[PageContent]:
    """Extract text content from a PDF file page by page.
    
    Args:
        pdf_path: Path to the PDF file to extract
        use_ocr_fallback: If True, use OCR for pages with little/no text
        
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
            
            # If text extraction yields very little content and OCR fallback is enabled
            # assume it's a scanned page and use OCR
            if use_ocr_fallback and should_use_ocr(text):
                try:
                    text = extract_text_with_ocr(page)
                except Exception:
                    # If OCR fails, fall back to whatever text we got
                    pass
            
            pages.append(
                PageContent(
                    source=pdf_path.name,
                    page_number=page_num,
                    text=text,
                )
            )
    
    return pages


def extract_pdf_blocks(pdf_path: str | Path, use_ocr_fallback: bool = True) -> list[TextBlock]:
    """Extract text blocks from a PDF file with layout information.
    
    Args:
        pdf_path: Path to the PDF file to extract
        use_ocr_fallback: If True, use OCR for pages with little/no text
        
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
            
            # Check if page has very little text content
            total_text = ''.join(block[4] for block in page_blocks if len(block) > 4)
            
            # If OCR fallback is enabled and page has little text, use OCR
            if use_ocr_fallback and should_use_ocr(total_text):
                try:
                    ocr_text = extract_text_with_ocr(page)
                    if ocr_text.strip():
                        # Create a single block for OCR text spanning the whole page
                        rect = page.rect
                        blocks.append(
                            TextBlock(
                                source=pdf_path.name,
                                page_number=page_num,
                                block_number=0,
                                bbox=BoundingBox(
                                    x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1
                                ),
                                text=ocr_text.strip(),
                            )
                        )
                        continue
                except Exception:
                    # If OCR fails, fall back to regular block extraction
                    pass
            
            # Regular block extraction
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
