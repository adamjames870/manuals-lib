"""Section context inference utilities."""

import re


def is_heading_line(line: str) -> bool:
    """Check if a line looks like a section heading.
    
    Heuristics:
    - Short line (< 80 chars)
    - Starts with capital letter
    - May start with number/letter prefix (e.g., "1.", "A.", "1.1")
    - Not a bullet point
    - Not ending with sentence punctuation (except colon)
    
    Args:
        line: Line to check
        
    Returns:
        True if line looks like a heading
    """
    stripped = line.strip()
    
    if not stripped or len(stripped) > 80:
        return False
    
    # Skip bullet points
    if re.match(r'^[\u2022\u2023\u25E6\u2043\u2219•·○●\-\*]\s', stripped):
        return False
    
    # Check for numbered/lettered heading patterns
    # Examples: "1. Introduction", "1.1 Overview", "A. Section Title"
    if re.match(r'^(\d+\.)+\s+[A-Z]', stripped):  # 1.1 Title
        return True
    if re.match(r'^(\d+|[A-Z])[\.\)]\s+[A-Z]', stripped):  # 1. Title or A. Title
        return True
    
    # Check for all-caps headings
    if stripped.isupper() and len(stripped.split()) <= 8:
        return True
    
    # Check for title case headings
    words = stripped.split()
    if len(words) <= 10:
        # Most words should start with capital
        capitalized = sum(1 for w in words if w and w[0].isupper())
        if capitalized >= len(words) * 0.7:
            # Should not end with sentence punctuation (except colon)
            if not stripped.endswith(('.', '!', '?', ',')):
                return True
            if stripped.endswith(':'):
                return True
    
    return False


def extract_heading_level(line: str) -> int:
    """Extract heading level from a heading line.
    
    Level 1: "1. Title" or "CHAPTER 1"
    Level 2: "1.1 Title" or "Section A"
    Level 3: "1.1.1 Title"
    
    Args:
        line: Heading line
        
    Returns:
        Heading level (1-based), or 1 if cannot determine
    """
    stripped = line.strip()
    
    # Check for numbered sections (1.1.1 format)
    match = re.match(r'^(\d+(?:\.\d+)*)', stripped)
    if match:
        number = match.group(1)
        level = number.count('.') + 1
        return level
    
    # Check for chapter/section keywords
    if re.match(r'^(CHAPTER|PART)\s+\d+', stripped, re.IGNORECASE):
        return 1
    if re.match(r'^(SECTION|APPENDIX)\s+[A-Z0-9]', stripped, re.IGNORECASE):
        return 2
    
    # Default to level 1 for all-caps, level 2 otherwise
    if stripped.isupper():
        return 1
    
    return 2


def infer_section_context(text: str, max_lookback: int = 500) -> tuple[str | None, list[str] | None]:
    """Infer section context from text by looking for nearby headings.
    
    Args:
        text: Text content to analyze
        max_lookback: Maximum characters to look back for headings
        
    Returns:
        Tuple of (section_title, section_path) where:
        - section_title is the most recent heading
        - section_path is hierarchical list of headings
        Returns (None, None) if no headings found
    """
    # Look at beginning of text for headings
    lookback_text = text[:max_lookback]
    lines = lookback_text.split('\n')
    
    headings = []
    for line in lines[:20]:  # Check first 20 lines
        if is_heading_line(line):
            level = extract_heading_level(line)
            heading_text = line.strip()
            headings.append((level, heading_text))
    
    if not headings:
        return None, None
    
    # Build hierarchical path
    section_path = []
    current_level = 0
    
    for level, heading in headings:
        # If this is a higher-level heading, reset path
        if level <= current_level:
            # Remove headings at same or lower level
            section_path = [h for l, h in section_path if l < level]
        
        section_path.append((level, heading))
        current_level = level
    
    # Extract just the heading text
    path_text = [h for _, h in section_path]
    section_title = path_text[-1] if path_text else None
    
    return section_title, path_text if path_text else None
