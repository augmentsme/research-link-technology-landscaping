from __future__ import annotations

from pipelines.batching import JsonlBatch, iter_jsonl_batches, stream_jsonl

__all__ = [
    "JsonlBatch",
    "iter_batches",
    "stream_jsonl",
]


def iter_batches(directory):
    yield from iter_jsonl_batches(directory)
