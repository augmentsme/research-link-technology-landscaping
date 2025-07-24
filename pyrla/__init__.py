"""
PyRLA - Python interface for Research Link Australia (RLA) API

This package provides a clean, object-oriented interface to interact with the
Research Link Australia API at https://researchlink.ardc.edu.au/v3/api-docs/public

Main classes:
- RLAClient: Main client for API interactions (supports both sync and async methods)
- Researcher: Represents a researcher entity
- Grant: Represents a grant entity
- Organisation: Represents an organisation entity
- Publication: Represents a publication entity
- SearchResponse: Response container for search results

Performance Features:
- Async support for concurrent API requests (much faster for large queries)
- Optimized publication searches with concurrent researcher fetching
- Efficient --all flag implementation with concurrent pagination

Usage:
    # Synchronous usage
    from pyrla import RLAClient
    client = RLAClient()
    researchers = client.search_researchers(value="AI")
    
    # Asynchronous usage (recommended for better performance)
    import asyncio
    async def main():
        async with RLAClient() as client:
            researchers = await client.search_researchers_async(value="AI")
    asyncio.run(main())
"""

from .client import RLAClient
from .models import Researcher, Grant, Organisation, Publication
from .exceptions import RLAError, RLANotFoundError, RLAAuthenticationError

__version__ = "0.1.0"
__all__ = [
    "RLAClient",
    "Researcher", 
    "Grant",
    "Organisation",
    "Publication",
    "RLAError",
    "RLANotFoundError", 
    "RLAAuthenticationError",
]
