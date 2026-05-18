"""Data models for embeddings."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ChunkMetadata:
    """Metadata for a chunk in the index.
    
    Attributes:
        row_index: Row index in the embeddings array
        chunk_id: Unique chunk identifier
        source: Source document filename
        page_start: Starting page number
        page_end: Ending page number
        text: Chunk text content
        chunk_type: Type of chunk (content, toc, table, etc.)
        extraction_method: Method used to extract source text
        table_id: Reference to source table if applicable
        section_title: Section heading if available
        section_path: Hierarchical path of section titles
    """
    row_index: int
    chunk_id: str
    source: str
    page_start: int
    page_end: int
    text: str
    chunk_type: str = "content"
    extraction_method: str = "pymupdf"
    table_id: str | None = None
    section_title: str | None = None
    section_path: list[str] | None = None


@dataclass
class IndexManifest:
    """Metadata about an embedding index.
    
    Attributes:
        model_name: Name of the embedding model used
        embedding_dim: Dimensionality of embeddings
        chunk_count: Number of chunks in the index
        source_file: Path to source chunks file
        created_at: ISO timestamp of index creation
    """
    model_name: str
    embedding_dim: int
    chunk_count: int
    source_file: str
    created_at: str
    
    @classmethod
    def create(
        cls,
        model_name: str,
        embedding_dim: int,
        chunk_count: int,
        source_file: str,
    ) -> "IndexManifest":
        """Create a new manifest with current timestamp.
        
        Args:
            model_name: Name of the embedding model
            embedding_dim: Dimensionality of embeddings
            chunk_count: Number of chunks
            source_file: Path to source chunks file
            
        Returns:
            New IndexManifest instance
        """
        return cls(
            model_name=model_name,
            embedding_dim=embedding_dim,
            chunk_count=chunk_count,
            source_file=source_file,
            created_at=datetime.utcnow().isoformat() + "Z",
        )
