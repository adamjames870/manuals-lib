"""Embedding generation using sentence-transformers."""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from manuals_lib.embeddings.models import ChunkMetadata, IndexManifest


class Embedder:
    """Generate embeddings for text chunks.
    
    Attributes:
        model: Sentence transformer model
        model_name: Name of the model
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", local_files_only: bool = False):
        """Initialize embedder with a sentence-transformers model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            local_files_only: If True, only use cached models without checking HF Hub
        """
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name, local_files_only=local_files_only)
        except OSError:
            # Model not cached, download it first
            if local_files_only:
                raise
            self.model = SentenceTransformer(model_name, local_files_only=False)
    
    @property
    def embedding_dim(self) -> int:
        """Get the dimensionality of embeddings from this model.
        
        Returns:
            Embedding dimension
        """
        return self.model.get_embedding_dimension()
    
    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding
            
        Returns:
            NumPy array of shape (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings


def load_chunks_from_json(chunks_path: Path) -> list[dict]:
    """Load chunks from a JSON file.
    
    Args:
        chunks_path: Path to chunks JSON file
        
    Returns:
        List of chunk dictionaries
        
    Raises:
        FileNotFoundError: If chunks file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data.get("chunks", [])


def build_index(
    chunks_path: Path,
    output_dir: Path,
    model_name: str = "all-MiniLM-L6-v2",
) -> tuple[Path, IndexManifest]:
    """Build an embedding index from chunks.
    
    Args:
        chunks_path: Path to input chunks JSON file
        output_dir: Directory to write index files
        model_name: Name of sentence-transformers model to use
        
    Returns:
        Tuple of (index_dir, manifest)
        
    Raises:
        FileNotFoundError: If chunks file doesn't exist
        ValueError: If chunks list is empty
    """
    # Load chunks
    chunks = load_chunks_from_json(chunks_path)
    
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")
    
    # Create output directory
    document_stem = chunks_path.stem.replace(".chunks", "")
    index_dir = output_dir / document_stem
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize embedder
    embedder = Embedder(model_name)
    
    # Prepare chunk metadata
    chunk_metadata = []
    texts = []
    
    for i, chunk in enumerate(chunks):
        chunk_metadata.append(
            ChunkMetadata(
                row_index=i,
                chunk_id=chunk.get("chunk_id", ""),
                source=chunk.get("source", ""),
                page_start=chunk.get("page_start", 0),
                page_end=chunk.get("page_end", 0),
                text=chunk.get("text", ""),
            )
        )
        texts.append(chunk.get("text", ""))
    
    # Generate embeddings
    embeddings = embedder.embed_texts(texts)
    
    # Create manifest
    manifest = IndexManifest.create(
        model_name=model_name,
        embedding_dim=embedder.embedding_dim,
        chunk_count=len(chunks),
        source_file=str(chunks_path),
    )
    
    # Write outputs
    # 1. Save embeddings as NumPy array
    embeddings_path = index_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    
    # 2. Save chunk metadata
    chunks_path_out = index_dir / "chunks.json"
    chunks_data = {
        "chunks": [
            {
                "row_index": cm.row_index,
                "chunk_id": cm.chunk_id,
                "source": cm.source,
                "page_start": cm.page_start,
                "page_end": cm.page_end,
                "text": cm.text,
            }
            for cm in chunk_metadata
        ]
    }
    with open(chunks_path_out, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    
    # 3. Save manifest
    manifest_path = index_dir / "manifest.json"
    manifest_data = {
        "model_name": manifest.model_name,
        "embedding_dim": manifest.embedding_dim,
        "chunk_count": manifest.chunk_count,
        "source_file": manifest.source_file,
        "created_at": manifest.created_at,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)
    
    return index_dir, manifest


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score between -1 and 1
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def cosine_similarity_matrix(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between a query and multiple embeddings.
    
    Args:
        query_embedding: Query vector of shape (embedding_dim,)
        embeddings: Matrix of embeddings of shape (n_chunks, embedding_dim)
        
    Returns:
        Array of similarity scores of shape (n_chunks,)
    """
    # Normalize query
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    
    # Normalize embeddings
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Compute dot product
    similarities = np.dot(embeddings_norm, query_norm)
    
    return similarities


def load_index(index_dir: Path) -> tuple[IndexManifest, list[ChunkMetadata], np.ndarray]:
    """Load an embedding index from disk.
    
    Args:
        index_dir: Directory containing manifest.json, chunks.json, and embeddings.npy
        
    Returns:
        Tuple of (manifest, chunk_metadata, embeddings)
        
    Raises:
        FileNotFoundError: If required files are missing
        json.JSONDecodeError: If JSON files are invalid
    """
    manifest_path = index_dir / "manifest.json"
    chunks_path = index_dir / "chunks.json"
    embeddings_path = index_dir / "embeddings.npy"
    
    # Check all files exist
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks not found: {chunks_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
    
    # Load manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    manifest = IndexManifest(
        model_name=manifest_data["model_name"],
        embedding_dim=manifest_data["embedding_dim"],
        chunk_count=manifest_data["chunk_count"],
        source_file=manifest_data["source_file"],
        created_at=manifest_data["created_at"],
    )
    
    # Load chunks
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    chunk_metadata = [
        ChunkMetadata(
            row_index=chunk["row_index"],
            chunk_id=chunk["chunk_id"],
            source=chunk["source"],
            page_start=chunk["page_start"],
            page_end=chunk["page_end"],
            text=chunk["text"],
        )
        for chunk in chunks_data["chunks"]
    ]
    
    # Load embeddings
    embeddings = np.load(embeddings_path)
    
    return manifest, chunk_metadata, embeddings


def semantic_search(
    query: str,
    index_dir: Path,
    top_k: int = 5,
) -> list[tuple[int, float, ChunkMetadata]]:
    """Perform semantic search over an embedding index.
    
    Args:
        query: Search query text
        index_dir: Directory containing the embedding index
        top_k: Number of top results to return
        
    Returns:
        List of (rank, similarity_score, chunk_metadata) tuples, sorted by similarity
        
    Raises:
        FileNotFoundError: If index files are missing
        ValueError: If top_k is invalid
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    
    # Load index
    manifest, chunk_metadata, embeddings = load_index(index_dir)
    
    # Initialize embedder with same model as index
    embedder = Embedder(manifest.model_name)
    
    # Embed query
    query_embedding = embedder.embed_texts([query])[0]
    
    # Compute similarities
    similarities = cosine_similarity_matrix(query_embedding, embeddings)
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Build results
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        similarity = float(similarities[idx])
        chunk = chunk_metadata[idx]
        results.append((rank, similarity, chunk))
    
    return results
