"""
Minimal FastMCP server exposing a single tool `sentence_similarity(a,b)`.

This file intentionally contains only the MCP server code. Use `python similarity.py`
to run the MCP server; FastMCP will expose the `sentence_similarity` tool which returns
cosine similarity between two input strings using the MiniLM sentence-transformer.
"""
from fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
import numpy as np
import argparse


mcp = FastMCP("SimilarityServer")


_MODEL = None
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_model(name: str = _MODEL_NAME) -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(name)
    return _MODEL


@mcp.tool
def sentence_similarity(a: str, b: str) -> float:
    """Return cosine similarity between two sentences as a float between -1 and 1.

    This is a synchronous tool suitable for MCP clients to call.
    """
    model = get_model()
    emb = model.encode([a, b], convert_to_numpy=True)
    u, v = emb[0], emb[1]
    denom = (np.linalg.norm(u) * np.linalg.norm(v))
    if denom == 0:
        return 0.0
    return float(np.dot(u, v) / denom)


if __name__ == "__main__":
    mcp.run()
