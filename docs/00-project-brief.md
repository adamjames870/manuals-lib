# Manuals LLM - Project Brief

## Goal

Build an agent-agnostic, local-first, assistant that can answer questions from a controlled set of manuals and technical documents.

The system should:
- ingest manuals from local files
- preserve source references
- answer using only indexed material where possible
- clearly say when the indexed material does not contain the answer
- support later expansion into agent-style workflows

## Initial Scope

Phase 1 is not a general chatbot.

Phase 1 will:
- load PDF / markdown / text manuals from `data/raw`
- extract text into a normalized format
- chunk documents
- create a local vector index
- allow command-line question answering
- return cited source snippets

## Non-Goals For Phase 1

- no web UI
- no multi-agent orchestration
- no cloud dependency
- no user accounts
- no fine-tuning
- no automatic updating from the internet

## Design Priorities

1. correctness over cleverness
2. traceable answers with citations
3. simple local development
4. easy to inspect/debug
5. replaceable components

## Likely Stack

- Python
- local LLM via Ollama
- embeddings via local model or lightweight embedding library
- vector store: Chroma, LanceDB, or FAISS
- document parsing: PyMuPDF / pypdf initially
- CLI first