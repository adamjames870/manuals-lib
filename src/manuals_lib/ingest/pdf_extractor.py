"""PDF text extraction functionality."""

import logging
from pathlib import Path

import pymupdf

from manuals_lib.ingest.models import BoundingBox, PageContent, TableData, TextBlock
from manuals_lib.ingest.ocr_extractor import extract_text_with_ocr, should_use_ocr

logger = logging.getLogger(__name__)


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
            original_text_len = len(text.strip())
            extraction_method = "pymupdf"
            ocr_engine = None
            ocr_trigger_reason = None
            
            # If text extraction yields very little content and OCR fallback is enabled
            # assume it's a scanned page and use OCR
            if use_ocr_fallback and should_use_ocr(text):
                ocr_trigger_reason = f"low_text_content_{original_text_len}_chars"
                logger.info(
                    f"Page {page_num}: Low text content ({original_text_len} chars), "
                    "attempting OCR"
                )
                try:
                    ocr_text = extract_text_with_ocr(page)
                    if ocr_text and ocr_text.strip():
                        text = ocr_text
                        extraction_method = "ocr"
                        ocr_engine = "tesseract"
                        logger.info(
                            f"Page {page_num}: OCR successful, extracted "
                            f"{len(text.strip())} characters"
                        )
                    else:
                        logger.warning(
                            f"Page {page_num}: OCR returned empty text, "
                            "keeping original extraction"
                        )
                        ocr_trigger_reason = None
                except Exception as e:
                    logger.error(f"Page {page_num}: OCR failed: {e}")
                    ocr_trigger_reason = None
                    # If OCR fails, fall back to whatever text we got
            
            pages.append(
                PageContent(
                    source=pdf_path.name,
                    page_number=page_num,
                    text=text,
                    extraction_method=extraction_method,
                    ocr_engine=ocr_engine,
                    ocr_trigger_reason=ocr_trigger_reason,
                )
            )
    
    return pages


def extract_pdf_tables(pdf_path: str | Path) -> list[TableData]:
    """Extract tables from a PDF file using PyMuPDF's table detection.
    
    Args:
        pdf_path: Path to the PDF file to extract
        
    Returns:
        List of TableData objects with table content
        
    Raises:
        FileNotFoundError: If the PDF file does not exist
        pymupdf.FileDataError: If the file is not a valid PDF
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    tables = []
    table_counter = 0
    
    with pymupdf.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            # Find tables on this page
            page_tables = page.find_tables()
            
            if not page_tables or not page_tables.tables:
                continue
            
            for table_idx, table in enumerate(page_tables.tables):
                table_counter += 1
                table_id = f"{pdf_path.stem}_table_{table_counter}"
                
                # Extract table data
                try:
                    # Get bounding box if available
                    bbox = None
                    if hasattr(table, 'bbox') and table.bbox:
                        bbox = BoundingBox(
                            x0=table.bbox[0],
                            y0=table.bbox[1],
                            x1=table.bbox[2],
                            y1=table.bbox[3],
                        )
                    
                    # Extract rows
                    rows = table.extract()
                    
                    if not rows:
                        continue
                    
                    # Try to infer headers from first row
                    headers = None
                    if rows and all(cell and isinstance(cell, str) for cell in rows[0]):
                        # First row looks like headers if cells are non-empty strings
                        first_row = [str(cell).strip() for cell in rows[0]]
                        if all(first_row):
                            headers = first_row
                    
                    tables.append(
                        TableData(
                            table_id=table_id,
                            source=pdf_path.name,
                            page_number=page_num,
                            bbox=bbox,
                            extraction_method="pymupdf_find_tables",
                            headers=headers,
                            rows=[[str(cell) if cell else "" for cell in row] for row in rows],
                        )
                    )
                    
                    logger.info(
                        f"Page {page_num}: Extracted table {table_id} "
                        f"with {len(rows)} rows"
                    )
                    
                except Exception as e:
                    logger.error(
                        f"Page {page_num}: Failed to extract table {table_idx}: {e}"
                    )
    
    return tables


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
                logger.info(
                    f"Page {page_num}: Low block text content "
                    f"({len(total_text.strip())} chars), attempting OCR"
                )
                try:
                    ocr_text = extract_text_with_ocr(page)
                    if ocr_text and ocr_text.strip():
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
                        logger.info(
                            f"Page {page_num}: OCR successful, extracted "
                            f"{len(ocr_text.strip())} characters"
                        )
                        continue
                    else:
                        logger.warning(
                            f"Page {page_num}: OCR returned empty text, "
                            "falling back to regular block extraction"
                        )
                except Exception as e:
                    logger.error(f"Page {page_num}: OCR failed: {e}")
                    # If OCR fails, fall back to regular block extraction
            
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
