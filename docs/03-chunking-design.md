# Chunking Design

## Goal

Convert normalized extracted pages into retrieval-ready chunks while preserving citation metadata.

## Input

`*.normalized.json`

## Output

`*.chunks.json`

## Non-Goals

- no embeddings
- no vector database
- no LLM calls
- no semantic reranking
- no OCR

## Chunk Requirements

Each chunk must include:

- chunk_id
- source
- page_start
- page_end
- text
- character_count
- extraction_version / pipeline_version if available

Optional later:

- section_heading
- previous_chunk_id
- next_chunk_id
- block references
- confidence / quality flags

## Initial Strategy

Use conservative page-aware chunking:

1. read normalized page text
2. split into paragraphs
3. accumulate paragraphs until target size
4. add overlap between adjacent chunks
5. never lose page number metadata
6. allow chunks to span pages only when text naturally continues

## Initial Defaults

- target characters: 1,500–2,000
- max characters: 2,500
- overlap: 200–300 characters

## Success Criteria

A chunk should be:
- readable by itself
- traceable to source pages
- not randomly cut mid-sentence where avoidable
- small enough for retrieval
- large enough to preserve context