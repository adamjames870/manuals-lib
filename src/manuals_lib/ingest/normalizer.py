"""Text normalization for extracted PDF content."""

import re
from dataclasses import dataclass

from manuals_lib.ingest.models import PageContent


@dataclass
class NormalizedPage:
    """Represents normalized content from a single page.
    
    Attributes:
        source: The filename of the source document
        page_number: The page number (1-indexed)
        text: The normalized text content
    """
    source: str
    page_number: int
    text: str


def normalize_line_endings(text: str) -> str:
    """Normalize all line endings to \\n.
    
    Args:
        text: Input text with mixed line endings
        
    Returns:
        Text with normalized line endings
    """
    return text.replace('\r\n', '\n').replace('\r', '\n')


def collapse_blank_lines(text: str, max_consecutive: int = 2) -> str:
    """Collapse excessive blank lines.
    
    Args:
        text: Input text
        max_consecutive: Maximum number of consecutive blank lines to keep
        
    Returns:
        Text with collapsed blank lines
    """
    # Replace 3+ consecutive newlines with max_consecutive newlines
    pattern = r'\n{' + str(max_consecutive + 1) + r',}'
    replacement = '\n' * max_consecutive
    return re.sub(pattern, replacement, text)


def is_special_line(line: str) -> bool:
    """Check if a line should not be joined with the next line.
    
    Detects:
    - Headings (short lines, often all caps or title case)
    - Bullets (lines starting with bullet characters)
    - Numbered items (lines starting with numbers/letters followed by punctuation)
    - Table rows (lines with multiple tab/space separations)
    - Short labels (very short lines that look like labels)
    
    Args:
        line: The line to check
        
    Returns:
        True if the line should not be joined
    """
    stripped = line.strip()
    
    # Empty lines
    if not stripped:
        return True
    
    # Bullet points
    if re.match(r'^[\u2022\u2023\u25E6\u2043\u2219•·○●\-\*]\s', stripped):
        return True
    
    # Numbered/lettered lists
    if re.match(r'^(\d+|[a-zA-Z])[\.\)]\s', stripped):
        return True
    
    # Lines that look like table rows (multiple tab/space separations)
    if '\t' in stripped or re.search(r'\s{3,}', stripped):
        return True
    
    # Lines ending with sentence-ending punctuation (complete thoughts)
    if stripped.endswith(('.', ':', ';', '!', '?')):
        return True
    
    # Very short lines (likely labels or headings) - but not if they end with hyphen/comma
    # which suggests continuation
    if len(stripped) < 50 and not stripped.endswith(('-', ',')):
        return True
    
    return False


def join_wrapped_lines(text: str) -> str:
    """Conservatively join wrapped paragraph lines.
    
    Args:
        text: Input text with potential wrapped lines
        
    Returns:
        Text with wrapped lines joined
    """
    lines = text.split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        current = lines[i]
        
        # Check if we should join with next line
        if i < len(lines) - 1:
            next_line = lines[i + 1]
            
            # Don't join if current or next line is special
            if not is_special_line(current) and not is_special_line(next_line):
                # Join lines with a space
                result.append(current.rstrip() + ' ' + next_line.lstrip())
                i += 2
                continue
        
        result.append(current)
        i += 1
    
    return '\n'.join(result)


def normalize_text(text: str, join_wrapped: bool = True) -> str:
    """Apply all normalization steps to text.
    
    Args:
        text: Input text to normalize
        join_wrapped: Whether to join wrapped paragraph lines
        
    Returns:
        Normalized text
    """
    # Normalize line endings
    text = normalize_line_endings(text)
    
    # Trim leading/trailing whitespace from the entire text
    text = text.strip()
    
    # Optionally join wrapped lines
    if join_wrapped:
        text = join_wrapped_lines(text)
    
    # Collapse excessive blank lines
    text = collapse_blank_lines(text)
    
    return text


def normalize_pages(pages: list[PageContent], join_wrapped: bool = True) -> list[NormalizedPage]:
    """Normalize a list of page content.
    
    Args:
        pages: List of PageContent objects to normalize
        join_wrapped: Whether to join wrapped paragraph lines
        
    Returns:
        List of NormalizedPage objects
    """
    normalized = []
    
    for page in pages:
        normalized_text = normalize_text(page.text, join_wrapped=join_wrapped)
        normalized.append(
            NormalizedPage(
                source=page.source,
                page_number=page.page_number,
                text=normalized_text,
            )
        )
    
    return normalized
