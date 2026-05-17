"""OCR-based text extraction for scanned documents."""

import pymupdf
import pytesseract
from PIL import Image


def extract_text_with_ocr(page: pymupdf.Page, dpi: int = 300) -> str:
    """Extract text from a page using OCR.
    
    Args:
        page: PyMuPDF page object
        dpi: DPI resolution for rendering (higher = better quality, slower)
        
    Returns:
        Extracted text from OCR
        
    Raises:
        Exception: If OCR processing fails
    """
    # Render page to image at specified DPI for better OCR quality
    pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi/72, dpi/72))
    
    # Convert to PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Run OCR
    text = pytesseract.image_to_string(img)
    
    return text


def should_use_ocr(text: str, min_chars: int = 50) -> bool:
    """Determine if OCR should be used based on extracted text quality.
    
    Args:
        text: Text extracted from PDF
        min_chars: Minimum character threshold
        
    Returns:
        True if OCR should be used (text is too short/empty)
    """
    return len(text.strip()) < min_chars
