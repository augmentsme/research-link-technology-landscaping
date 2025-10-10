from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Iterable, AsyncIterator, Optional

from openai import AsyncOpenAI, OpenAI
from tenacity import AsyncRetrying, Retrying, stop_after_attempt, wait_exponential

import config


@asynccontextmanager
async def async_client() -> AsyncIterator[AsyncOpenAI]:
    async with AsyncOpenAI(
        base_url=config.OPENAI_BASE_URL,
        timeout=config.OPENAI_TIMEOUT_SECONDS,
    ) as client:
        yield client


def sync_client(base_url: Optional[str] = None) -> OpenAI:
    target_base_url = base_url or config.OPENAI_BASE_URL
    return OpenAI(
        base_url=target_base_url,
        timeout=config.OPENAI_TIMEOUT_SECONDS,
    )


async def call_json_schema(
    client: AsyncOpenAI,
    *,
    model: str,
    system_prompt: str,
    user_content: str,
    schema_name: str,
    schema: dict[str, Any],
) -> str:
    retry = AsyncRetrying(
        stop=stop_after_attempt(config.OPENAI_MAX_RETRIES),
        wait=wait_exponential(min=1, max=10),
        reraise=True,
    )
    async for attempt in retry:
        with attempt:
            response = await client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    }
                },
                timeout=config.OPENAI_TIMEOUT_SECONDS,
            )
            return response.output_text

    raise RuntimeError("Retrying should have raised before reaching this point")


def retry_embeddings(inputs: Iterable[str], *, client: OpenAI, model: str) -> list[list[float]]:
    results: list[list[float]] = []
    for attempt in Retrying(
        stop=stop_after_attempt(config.OPENAI_MAX_RETRIES),
        wait=wait_exponential(min=1, max=10),
        reraise=True,
    ):
        with attempt:
            response = client.embeddings.create(
                model=model,
                input=list(inputs),
                timeout=config.OPENAI_TIMEOUT_SECONDS,
            )
            results.extend(item.embedding for item in response.data)
            break
    return results
