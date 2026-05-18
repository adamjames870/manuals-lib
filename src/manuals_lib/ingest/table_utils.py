"""Table processing and TOC detection utilities."""

import re


def flatten_table_row(
    row: list[str],
    headers: list[str] | None = None,
    table_title: str | None = None,
) -> str:
    """Flatten a table row into readable labeled text.
    
    Args:
        row: List of cell values
        headers: Optional column headers
        table_title: Optional table title/description
        
    Returns:
        Flattened text representation suitable for embeddings
    """
    parts = []
    
    if table_title:
        parts.append(f"Table: {table_title}.")
    
    if headers and len(headers) == len(row):
        # Create labeled pairs
        for header, value in zip(headers, row):
            if value and value.strip():
                parts.append(f"{header}: {value.strip()}")
    else:
        # No headers, just concatenate values
        for i, value in enumerate(row):
            if value and value.strip():
                parts.append(f"Column {i+1}: {value.strip()}")
    
    return " | ".join(parts) if parts else ""


def detect_chunk_type(text: str) -> str:
    """Detect the type of chunk based on content heuristics.
    
    Args:
        text: Chunk text content
        
    Returns:
        Chunk type: content, toc, front_matter, index, or unknown
    """
    text_lower = text.lower()
    lines = text.split('\n')
    
    # Check for table of contents indicators
    toc_keywords = [
        'table of contents',
        'contents',
        'quick links',
    ]
    
    if any(keyword in text_lower for keyword in toc_keywords):
        return 'toc'
    
    # Check for index indicators
    index_keywords = ['index', 'alphabetical index']
    if any(keyword in text_lower for keyword in index_keywords):
        return 'index'
    
    # Check for front matter patterns
    # Many short lines ending in page numbers
    short_lines_with_numbers = 0
    dotted_leader_lines = 0
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # Check for dotted leaders (common in TOC)
        if re.search(r'\.{3,}', line_stripped):
            dotted_leader_lines += 1
        
        # Check for short lines ending in numbers
        if len(line_stripped) < 80 and re.search(r'\d+\s*$', line_stripped):
            short_lines_with_numbers += 1
    
    # If many lines have TOC-like patterns, classify as TOC
    total_lines = len([l for l in lines if l.strip()])
    if total_lines > 0:
        toc_ratio = (short_lines_with_numbers + dotted_leader_lines) / total_lines
        if toc_ratio > 0.5:
            return 'toc'
    
    # Check for front matter indicators
    front_matter_keywords = [
        'preface',
        'foreword',
        'introduction',
        'about this manual',
        'safety information',
        'warnings',
        'copyright',
    ]
    
    # Front matter is usually at the beginning and has these keywords
    if any(keyword in text_lower[:500] for keyword in front_matter_keywords):
        return 'front_matter'
    
    return 'content'


def is_front_matter_or_toc(chunk_type: str) -> bool:
    """Check if a chunk type should be excluded from retrieval by default.
    
    Args:
        chunk_type: The chunk type string
        
    Returns:
        True if chunk should be excluded from default retrieval
    """
    return chunk_type in ('toc', 'front_matter', 'index')
