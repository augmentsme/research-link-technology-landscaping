from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Iterable, AsyncIterator, Optional

from openai import AsyncOpenAI, OpenAI

import config


@asynccontextmanager
async def async_client() -> AsyncIterator[AsyncOpenAI]:
    async with AsyncOpenAI(
        base_url=config.OPENAI_BASE_URL,
        timeout=config.OPENAI_TIMEOUT_SECONDS,
    ) as client:
        yield client


async def call_json_schema(
    client: AsyncOpenAI,
    *,
    model: str,
    system_prompt: str,
    user_content: str,
    schema_name: str,
    schema: dict[str, Any],
) -> str:
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


def create_embeddings(inputs: Iterable[str], *, client: OpenAI, model: str) -> list[list[float]]:
    response = client.embeddings.create(
        model=model,
        input=list(inputs),
        timeout=config.OPENAI_TIMEOUT_SECONDS,
    )
    return [item.embedding for item in response.data]
