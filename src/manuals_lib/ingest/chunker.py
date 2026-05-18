"""Text chunking for retrieval."""

import re
from dataclasses import dataclass

from manuals_lib.ingest.models import Chunk, TableData
from manuals_lib.ingest.normalizer import NormalizedPage
from manuals_lib.ingest.table_utils import detect_chunk_type, flatten_table_row


@dataclass
class ChunkingConfig:
    """Configuration for text chunking.
    
    Attributes:
        target_size: Target chunk size in characters
        max_size: Maximum chunk size in characters
        overlap_size: Number of characters to overlap between chunks
    """
    target_size: int = 1750
    max_size: int = 2500
    overlap_size: int = 250


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs.
    
    Args:
        text: Input text
        
    Returns:
        List of paragraph strings
    """
    # Split on double newlines or more
    paragraphs = re.split(r'\n\n+', text)
    
    # Filter out empty paragraphs and strip whitespace
    return [p.strip() for p in paragraphs if p.strip()]


def create_chunk_id(source: str, chunk_index: int) -> str:
    """Create a unique chunk identifier.
    
    Args:
        source: Source document filename
        chunk_index: Zero-indexed chunk number
        
    Returns:
        Chunk ID string
    """
    # Remove file extension and create ID
    stem = source.rsplit('.', 1)[0] if '.' in source else source
    return f"{stem}_chunk_{chunk_index:04d}"


def chunk_pages(
    pages: list[NormalizedPage],
    config: ChunkingConfig | None = None
) -> list[Chunk]:
    """Chunk normalized pages into retrieval-ready chunks.
    
    Args:
        pages: List of NormalizedPage objects
        config: Chunking configuration (uses defaults if None)
        
    Returns:
        List of Chunk objects
    """
    if config is None:
        config = ChunkingConfig()
    
    if not pages:
        return []
    
    source = pages[0].source
    chunks = []
    chunk_index = 0
    
    # Current chunk being built
    current_text = []
    current_size = 0
    current_page_start = pages[0].page_number
    current_page_end = pages[0].page_number
    current_extraction_method = pages[0].extraction_method
    
    # Previous chunk text for overlap
    previous_chunk_text = ""
    
    for page in pages:
        paragraphs = split_into_paragraphs(page.text)
        
        for paragraph in paragraphs:
            para_len = len(paragraph)
            
            # If adding this paragraph would exceed max size, finalize current chunk
            if current_size > 0 and current_size + para_len + 2 > config.max_size:
                # Finalize current chunk
                chunk_text = '\n\n'.join(current_text)
                chunks.append(
                    Chunk(
                        chunk_id=create_chunk_id(source, chunk_index),
                        source=source,
                        page_start=current_page_start,
                        page_end=current_page_end,
                        text=chunk_text,
                        char_count=len(chunk_text),
                        chunk_type=detect_chunk_type(chunk_text),
                        extraction_method=current_extraction_method,
                    )
                )
                
                # Prepare for next chunk with overlap
                previous_chunk_text = chunk_text
                chunk_index += 1
                
                # Start new chunk with overlap from previous chunk
                overlap_text = get_overlap_text(previous_chunk_text, config.overlap_size)
                if overlap_text:
                    current_text = [overlap_text]
                    current_size = len(overlap_text)
                else:
                    current_text = []
                    current_size = 0
                
                current_page_start = page.page_number
                current_page_end = page.page_number
                current_extraction_method = page.extraction_method
            
            # Add paragraph to current chunk
            current_text.append(paragraph)
            current_size += para_len + 2  # +2 for paragraph separator
            current_page_end = page.page_number
            
            # If we've reached target size, consider finalizing
            if current_size >= config.target_size:
                # Finalize current chunk
                chunk_text = '\n\n'.join(current_text)
                chunks.append(
                    Chunk(
                        chunk_id=create_chunk_id(source, chunk_index),
                        source=source,
                        page_start=current_page_start,
                        page_end=current_page_end,
                        text=chunk_text,
                        char_count=len(chunk_text),
                        chunk_type=detect_chunk_type(chunk_text),
                        extraction_method=current_extraction_method,
                    )
                )
                
                # Prepare for next chunk with overlap
                previous_chunk_text = chunk_text
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = get_overlap_text(previous_chunk_text, config.overlap_size)
                if overlap_text:
                    current_text = [overlap_text]
                    current_size = len(overlap_text)
                else:
                    current_text = []
                    current_size = 0
                
                current_page_start = page.page_number
                current_page_end = page.page_number
                current_extraction_method = page.extraction_method
    
    # Finalize any remaining text
    if current_text:
        chunk_text = '\n\n'.join(current_text)
        chunks.append(
            Chunk(
                chunk_id=create_chunk_id(source, chunk_index),
                source=source,
                page_start=current_page_start,
                page_end=current_page_end,
                text=chunk_text,
                char_count=len(chunk_text),
                chunk_type=detect_chunk_type(chunk_text),
                extraction_method=current_extraction_method,
            )
        )
    
    return chunks


def chunk_tables(tables: list[TableData]) -> list[Chunk]:
    """Convert tables into retrieval-ready chunks.
    
    Args:
        tables: List of TableData objects
        
    Returns:
        List of Chunk objects derived from tables
    """
    chunks = []
    
    for table in tables:
        # Use table_title from metadata if available, otherwise check first row
        table_title = table.table_title
        data_rows = table.rows
        
        # If no title in metadata and first row has only one non-empty cell, treat it as title
        if not table_title and data_rows:
            if len([cell for cell in data_rows[0] if cell.strip()]) == 1:
                table_title = next((cell for cell in data_rows[0] if cell.strip()), None)
                data_rows = data_rows[1:]
        
        # Create a chunk for each row
        for row_idx, row in enumerate(data_rows):
            # Skip empty rows
            if not any(cell.strip() for cell in row):
                continue
            
            # Flatten row to text
            row_text = flatten_table_row(row, table.headers, table_title)
            
            if not row_text:
                continue
            
            chunk_id = f"{table.table_id}_row_{row_idx}"
            
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source=table.source,
                    page_start=table.page_number,
                    page_end=table.page_number,
                    text=row_text,
                    char_count=len(row_text),
                    chunk_type="table_row",
                    extraction_method=table.extraction_method,
                    table_id=table.table_id,
                    section_title=table_title,  # Store table title as section_title
                )
            )
    
    return chunks


def get_overlap_text(text: str, overlap_size: int) -> str:
    """Get the last N characters of text for overlap.
    
    Tries to break at paragraph or sentence boundaries.
    
    Args:
        text: Source text
        overlap_size: Desired overlap size in characters
        
    Returns:
        Overlap text
    """
    if len(text) <= overlap_size:
        return text
    
    # Get the last overlap_size characters
    overlap = text[-overlap_size:]
    
    # Try to start at a paragraph boundary
    para_match = re.search(r'\n\n', overlap)
    if para_match:
        return overlap[para_match.end():].strip()
    
    # Try to start at a sentence boundary
    sentence_match = re.search(r'[.!?]\s+', overlap)
    if sentence_match:
        return overlap[sentence_match.end():].strip()
    
    # Fall back to word boundary
    space_match = re.search(r'\s+', overlap)
    if space_match:
        return overlap[space_match.end():].strip()
    
    return overlap.strip()
