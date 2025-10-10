from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar, Any

from tqdm.asyncio import tqdm

T = TypeVar("T")
R = TypeVar("R")


async def run_with_concurrency(
    items: Iterable[T],
    handler: Callable[[T], Awaitable[R]],
    *,
    concurrency: int,
    progress_description: str,
    progress_unit: str,
) -> list[R]:
    semaphore = asyncio.Semaphore(max(concurrency, 1))
    tasks: list[asyncio.Task[R | None]] = []

    async def limited(item: T) -> R | None:
        async with semaphore:
            return await handler(item)

    for item in items:
        tasks.append(asyncio.create_task(limited(item)))

    results: list[R] = []
    for task in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=progress_description,
        unit=progress_unit,
    ):
        result = await task
        if result is not None:
            results.append(result)
    return results
