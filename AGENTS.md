# Agents Instructions

This repository is intended to be developed incrementally and conservatively.

## Core Principles

- Prefer simple and inspectable code.
- Avoid framework-heavy abstractions unless justified.
- Do not introduce distributed systems patterns prematurely.
- Local-first development is preferred.
- Every major architectural change should be explained before implementation.

## Documents

There are documents for you in the docs/ folder.
- 00-project-brief.md is for me to edit. You must read this file
- 01-architechture.md is for you to edit. You must read and keep this updated with architectural changes, so the file represents the top-level design of the app
- 03-development-rules.md is for you to edit. You must read and keep this updated with any instructions given by the user. Warn when you receive conflicting instructions

## Development Workflow

Before making any architectural changes:
1. explain the intended change
2. explain why it is necessary
3. identify tradeoffs
4. identify affected components
5. wait for approval if the change is architectural

Examples of architectural changes:
- introducing a new database
- changing vector store
- introducing agent orchestration
- adding background workers
- changing document chunking strategy
- changing embedding models
- changing project structure
- adding new files or modules
- changing dependencies within the solution or project
- adding new external dependencies

## Code Standards

- Small focused modules.
- Avoid unnecessary inheritance.
- Prefer explicit types.
- Avoid hidden global state.
- Functions should be easy to debug from logs.
- Favour composition over magic frameworks.

## Logging

All important operations should log:
- document ingestion
- chunk counts
- embedding generation
- retrieval operations
- model requests
- failures and retries

Logs should use a normal hierarchy of log levels, but omitting warnings in favour of error 
- DEBUG for detailed information
- INFO for important events
- ERROR for failures
- CRITICAL for critical errors

Logs should be presented in two ways
- human-readable to STDOUT, include colour coding, with all information required to analyse errors and understand the flow of the program
- machine-readable to a file, favouring too-much detail rather than too-little


## LLM Behaviour Expectations

The assistant should:
- prefer grounded answers
- provide citations/snippets
- admit uncertainty
- avoid hallucinating unsupported answers

## Retrieval Rules

- Retrieval quality is more important than generation quality.
- Preserve source metadata throughout the pipeline.
- Every chunk should retain:
  - source document
  - page number where possible
  - section heading where possible

## Initial Scope Constraints

Phase 1 should NOT include:
- autonomous agents
- automatic tool execution
- internet access
- self-modifying prompts
- memory systems
- workflow planners
- background daemons

## Preferred Early Architecture

Initial pipeline:

documents
→ extraction
→ normalization
→ chunking
→ embeddings
→ vector index
→ retrieval
→ answer synthesis

Keep each stage independently testable.

## Dependency Philosophy

Before adding a dependency:
- explain why standard library or existing code is insufficient
- prefer mature/simple libraries
- avoid highly abstract AI orchestration frameworks early

## Git Expectations

Commit frequently.

Do not change the history or branch unless instructed
You should always be working on a feature branch, if you are on dev, test or main, do not change code but warn the user to change branch

If explicitly requested to merge to the dev branch, you may do so, squashing all commits on the feature branch to a single commit message 

Recommended commit style:
- feat:
- fix:
- refactor:
- docs:
- chore:

## Priority Order

1. correctness
2. debuggability
3. maintainability
4. performance
5. sophistication