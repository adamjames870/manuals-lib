"""OCR-based text extraction for scanned documents."""

import logging

import pymupdf
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


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
    try:
        # Render page to image at specified DPI for better OCR quality
        logger.debug(f"Rendering page at {dpi} DPI for OCR")
        pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi/72, dpi/72))
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        logger.debug(f"Created PIL image: {img.width}x{img.height} pixels")
        
        # Run OCR
        logger.debug("Running Tesseract OCR")
        text = pytesseract.image_to_string(img)
        logger.debug(f"OCR extracted {len(text)} characters")
        
        return text
    except pytesseract.TesseractNotFoundError as e:
        logger.error("Tesseract OCR is not installed or not in PATH")
        raise RuntimeError(
            "Tesseract OCR is not installed. "
            "Install it with: apt-get install tesseract-ocr (Linux) "
            "or brew install tesseract (macOS)"
        ) from e
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        raise


def should_use_ocr(text: str, min_chars: int = 50) -> bool:
    """Determine if OCR should be used based on extracted text quality.
    
    Args:
        text: Text extracted from PDF
        min_chars: Minimum character threshold
        
    Returns:
        True if OCR should be used (text is too short/empty)
    """
    return len(text.strip()) < min_chars
