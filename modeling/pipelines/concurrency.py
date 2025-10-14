from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar

from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)

T = TypeVar("T")
R = TypeVar("R")


async def run_with_concurrency(
    items: Iterable[T],
    handler: Callable[[T], Awaitable[R]],
    *,
    concurrency: int,
    progress_description: str,
    progress_unit: str,
    progress_callback: Callable[[R | None, Progress, int], None] | None = None,
) -> list[R]:
    semaphore = asyncio.Semaphore(max(concurrency, 1))
    tasks: list[asyncio.Task[R | None]] = []

    async def limited(item: T) -> R | None:
        async with semaphore:
            return await handler(item)

    for item in items:
        tasks.append(asyncio.create_task(limited(item)))

    results: list[R] = []
    total_tasks = len(tasks)
    if total_tasks == 0:
        return results

    progress = Progress(
        SpinnerColumn(),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} " + progress_unit),
        TextColumn("{task.fields[rate]}", justify="left"),
        TimeElapsedColumn(),
        transient=True,
    )
    progress.start()
    task_id = progress.add_task(
        progress_description,
        total=total_tasks,
        rate="",
    )
    try:
        for completed in asyncio.as_completed(tasks):
            result = await completed
            progress.advance(task_id)
            if progress_callback is not None:
                progress_callback(result, progress, task_id)
            if result is not None:
                results.append(result)
    finally:
        progress.stop()
    return results
