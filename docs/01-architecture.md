# Architecture

## Overview

The manuals-lib system is a local-first document ingestion and retrieval pipeline for technical manuals.

## Pipeline Stages

### 1. Extraction
- **Input**: PDF files in `data/raw/`
- **Output**: `*.extracted.json`, `*.blocks.json`
- **Module**: `src/manuals_lib/ingest/pdf_extractor.py`
- **Purpose**: Extract raw text and block-level content from PDFs

### 2. Normalization
- **Input**: `*.extracted.json`
- **Output**: `*.normalized.json`
- **Module**: `src/manuals_lib/ingest/normalizer.py`
- **Purpose**: Clean text, join wrapped lines, collapse blank lines

### 3. Chunking
- **Input**: `*.normalized.json`
- **Output**: `*.chunks.json`
- **Module**: `src/manuals_lib/ingest/chunker.py`
- **Purpose**: Split normalized text into retrieval-ready chunks with overlap
- **Config**: Target 1750 chars, max 2500 chars, 250 char overlap

### 4. Embedding Generation
- **Input**: `*.chunks.json`
- **Output**: `data/index/<document>/` containing:
  - `manifest.json` - index metadata
  - `chunks.json` - chunk metadata aligned with embeddings
  - `embeddings.npy` - NumPy array of embeddings
- **Module**: `src/manuals_lib/embeddings/`
- **Model**: sentence-transformers (default: all-MiniLM-L6-v2)
- **Purpose**: Generate vector embeddings for semantic search

### 5. Retrieval (Current)
- **Type**: Keyword-based search
- **Module**: `src/manuals_lib/retrieval/keyword_search.py`
- **Features**:
  - Exact phrase matching (100 points)
  - Individual keyword matching (1 point per occurrence)
  - Position-based scoring bonus
  - Front matter detection and penalty
  - Multi-document search

## Data Flow

```
PDFs (data/raw/)
  ↓
Extract (batch_extract.py)
  ↓
*.extracted.json, *.blocks.json (data/processed/)
  ↓
Normalize
  ↓
*.normalized.json (data/processed/)
  ↓
Chunk
  ↓
*.chunks.json (data/processed/)
  ↓
Embed (build_embeddings.py)
  ↓
Index (data/index/<document>/)
  ├── manifest.json
  ├── chunks.json
  └── embeddings.npy
  ↓
Search (search_chunks.py - keyword)
  ↓
Results with citations
```

## Storage

- **Raw PDFs**: `data/raw/`
- **Processed JSON**: `data/processed/`
- **Embedding Indices**: `data/index/<document>/`

## Key Design Decisions

1. **Local-first**: No cloud dependencies, all processing local
2. **Simple serialization**: JSON + NumPy arrays, no vector database yet
3. **Inspectable**: All intermediate outputs saved as readable JSON
4. **Traceable**: Page numbers and source preserved throughout pipeline
5. **Modular**: Each stage independently testable

## Dependencies

- **PDF Processing**: pymupdf, pytesseract (OCR fallback)
- **Text Processing**: Standard library (re, dataclasses)
- **Embeddings**: sentence-transformers, numpy
- **CLI/Display**: rich, argparse
- **Testing**: pytest

## Future Considerations

- Semantic search using embeddings (next phase)
- Vector database integration (if needed for scale)
- LLM-based answer generation
- Multi-document cross-referencing
