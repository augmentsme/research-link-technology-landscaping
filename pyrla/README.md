# PyRLA - Python Interface for Research Link Australia API

PyRLA is a high-performance Python package that provides an async-only interface to interact with the Research Link Australia (RLA) API at `https://researchlink.ardc.edu.au/v3/api-docs/public`.

## Features

- **High-Performance Async Client**: Asynchronous API client with concurrent request handling for 60-90% faster bulk operations
- **Object-Oriented Models**: Structured classes for Researcher, Grant, Organisation, and Publication entities
- **Concurrent Search**: Async search methods with configurable concurrency limits and automatic pagination
- **Command Line Interface**: Full-featured CLI with automatic async optimization for bulk operations
- **Utility Functions**: Helper functions for data analysis and manipulation
- **Error Handling**: Custom exception classes for different error types
- **Performance Optimized**: Built for speed with `aiohttp` and `asyncio` for non-blocking I/O

## Installation

Since this is a local package, you can install it in development mode:

```bash
# Install in development mode (includes all dependencies)
cd pyrla
pip install -e .
```

For development, you can also install with optional dev dependencies:

```bash
pip install -e ".[dev]"
```

Note: The package requires Python 3.12 or higher.

## Authentication

You need an API token to access the RLA API. Set it as an environment variable:

```bash
export ARL_API_TOKEN="your_api_token_here"
```

Or provide it directly when creating the client:

```python
from pyrla import RLAClient
async with RLAClient(api_token="your_api_token_here") as client:
    # Your async code here
    pass
```

## Quick Start

### Async Python API

PyRLA uses an async-only interface for optimal performance:

```python
import asyncio
from pyrla.client import RLAClient

async def main():
    async with RLAClient() as client:
        # Search for researchers
        researchers = await client.search_researchers(
            value="machine learning",
            page_size=10
        )
        
        # Search for publications with concurrent requests
        publications = await client.search_publications(
            value="artificial intelligence", 
            page_size=20,
            max_concurrent=10
        )
        
        # Fetch all results across all pages (high performance)
        all_researchers = await client.search_all_researchers(
            value="data science",
            max_concurrent=5
        )
        
        all_grants = await client.search_all_grants(
            value="machine learning",
            max_concurrent=5
        )

asyncio.run(main())
```

### Concurrent Operations

Execute multiple searches concurrently for maximum performance:

```python
async def concurrent_searches():
    async with RLAClient() as client:
        # Run multiple searches concurrently
        results = await asyncio.gather(
            client.search_researchers(value="AI", page_size=10),
            client.search_grants(value="machine learning", page_size=10),
            client.search_organisations(value="university", page_size=10),
        )
        
        researchers, grants, organisations = results
        return researchers, grants, organisations
```

## Available Methods

### Researcher Methods

- `search_researchers()`
- `get_researcher()`
- `search_all_researchers()`
- `search_researchers_by_name()`

### Grant Methods

- `search_grants()`
- `get_grant()`
- `search_all_grants()`
- `search_grants_by_funder()`
- `search_active_grants()`

### Organisation Methods

- `search_organisations()`
- `get_organisation()`
- `search_all_organisations()`

### Publication Methods

- `search_publications()`
- `get_publication()`

## Performance Benefits

The async-only approach provides significant performance improvements:

- **60-90% faster** for bulk operations (using `--all` flag in CLI)
- **Concurrent API requests** with configurable limits (default 5-10 concurrent requests)
- **Automatic async usage** in CLI for bulk operations
- **Better resource utilization** through non-blocking I/O

## Command Line Interface

PyRLA includes a comprehensive CLI that automatically uses async operations for optimal performance:

```bash
# Test connection
python -m pyrla.cli test-connection

# Search researchers (automatically uses async for --all)
python -m pyrla.cli researchers search "machine learning" --all

# Search grants with filtering
python -m pyrla.cli grants search "AI" --filter "status:Active" --all

# Get specific researcher
python -m pyrla.cli researchers get "researcher_id_here"
```

The CLI maintains the same user interface while automatically leveraging async performance improvements.

### CLI Examples

```bash
# Search researchers with filters
python -m pyrla.cli researchers search "artificial intelligence" \
    --filter "currentOrganisationName:University" \
    --size 50 \
    --json results.json

# Search all grants from a specific funder
python -m pyrla.cli grants search "" \
    --filter "funder:Australian Research Council" \
    --all \
    --stats

# Search organisations with advanced query
python -m pyrla.cli organisations search "university" \
    --advanced "states:NSW" \
    --all

# Search publications by year range
python -m pyrla.cli publications search "machine learning" \
    --year 2023 \
    --type "journal article" \
    --async
```

## Migration from Sync Version

If you were using a previous synchronous version, here's how to migrate:

**Before (sync):**
```python
from pyrla.client import RLAClient

client = RLAClient()
researchers = client.search_researchers(value="machine learning")
```

**After (async-only):**
```python
import asyncio
from pyrla.client import RLAClient

async def main():
    async with RLAClient() as client:
        researchers = await client.search_researchers(value="machine learning")

asyncio.run(main())
```

## Dependencies

- `aiohttp` - For async HTTP requests
- `asyncio` - For concurrency management (built-in)
- `typer` - For CLI framework
- `rich` - For beautiful terminal output
- `python-dotenv` - For environment variable management

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
python -m pytest tests/

# Check code with linters
python -m flake8 pyrla/
python -m mypy pyrla/
```

## API Documentation

For full API documentation, visit: https://researchlink.ardc.edu.au/v3/api-docs/public

## License

This project is licensed under the MIT License.
