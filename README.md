# manuals-lib

A minimal Python library for ingesting technical manuals and documentation from PDF files.

## Features

- Extract text content from PDF files page-by-page
- Simple dataclass-based models for structured content
- No dependencies on embeddings, vector databases, or LLMs
- Clean src layout following Python best practices

## Requirements

- Python 3.10+
- uv (recommended) or pip for package management

## Installation

Using uv (recommended):

```bash
uv sync
```

Using pip:

```bash
pip install -e .
```

For development dependencies:

```bash
uv sync --extra dev
```

## Usage

### As a Library

```python
from manuals_lib.ingest import extract_pdf

# Extract text from a PDF file
pages = extract_pdf("path/to/manual.pdf")

# Access page content
for page in pages:
    print(f"Page {page.page_number} from {page.source}")
    print(page.text)
```

### Command Line Script

Extract and display text from a PDF:

```bash
uv run python scripts/extract_pdf.py path/to/manual.pdf
```

## Development

### Running Tests

```bash
uv run pytest tests/
```

### Code Formatting

```bash
uv run ruff check .
uv run ruff format .
```

## Project Structure

```
manuals-lib/
├── src/
│   └── manuals_lib/
│       ├── __init__.py
│       └── ingest/
│           ├── __init__.py
│           ├── models.py          # Data models
│           └── pdf_extractor.py   # PDF extraction logic
├── scripts/
│   └── extract_pdf.py             # CLI script
├── tests/
│   └── test_models.py
├── pyproject.toml
└── README.md
```

## License

[Add your license here]
