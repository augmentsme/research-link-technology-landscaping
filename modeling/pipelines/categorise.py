from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Iterable, List, Sequence

import config
from models import FieldOfResearch

from openai import AsyncOpenAI

from . import llm_client
from .batching import (
    iter_jsonl_batches,
    stream_jsonl,
    prepare_category_outputs,
    write_categorisation_result,
    JsonlBatch,
)
from .concurrency import run_with_concurrency
from pydantic import BaseModel, Field


class Category(BaseModel):
    model_config = {"extra": "forbid"}
    name: str = Field(description="Name of the category")
    description: str = Field(description="A few sentences describing what this category is about, including its scope, focus areas, and the types of research or technologies it encompasses")
    keywords: List[str] = Field(description="List of keywords associated with this category")
    field_of_research: FieldOfResearch = Field(description="The field of research division this category falls under")


class CategoryList(BaseModel):
    model_config = {"extra": "forbid"}
    categories: List[Category] = Field(description="List of research categories")


CATEGORISE_SYSTEM_PROMPT = """
You are an expert Technology Analyst and Innovation Forecaster, specializing in synthesizing information to identify and map emerging technological domains for strategic planning.

Your task is to analyze a list of user-provided keywords and generate a comprehensive set of research and technology categories in a specific JSON format. Each category must be linked to an appropriate FOR (Fields of Research) division.

**IMPORTANT: You MUST complete this task immediately and provide the full categorization. Do NOT ask for clarification, approval, or propose alternative approaches. Proceed directly with categorizing all provided keywords.**

**Input Format:**
Each keyword will be provided in the format: `**keyword_term** (type): description`

**Core Objective:**
Your primary goal is to organize the provided keywords into meaningful, emergent categories that bridge the specificity of the keywords with the broadness of the 23 top-level FOR divisions. Your analysis should favor the identification of potential breakthroughs and new interdisciplinary fields.

---

### **CRITICAL RULES FOR CATEGORIZATION:**

**1. Category Creation & Granularity:**
*   **Natural Groupings:** Create categories based on the natural, thematic relationships between the keywords. The number of categories should be determined by these organic groupings, not a predefined target.
*   **Optimal Abstraction:** Categories should be more specific than the broad FOR divisions but more general than individual keywords. They should represent recognizable research areas.
*   **Focus on Emergence:** Actively look for intersections between keywords that suggest new or interdisciplinary domains. When in doubt, err on the side of creating a new, more specific category to avoid missing potential innovations.

**2. Keyword Handling & Coverage:**
*   **Complete Coverage:** EVERY keyword term provided by the user MUST be included in the `keywords` list of exactly one category. No keywords should be left uncategorized.
*   **No Placeholder Categories:** Do NOT create categories like "Clarification Required" or similar placeholder responses. Only create real, meaningful research categories.
*   **Contextual Analysis:** Use the keyword `(type)` and `description` to understand the context and make more informed grouping decisions.
*   **Exact Term Usage:** The `keywords` array within each category must contain the exact `keyword_term` strings from the input.

**3. Output JSON Structure & Content:**
*   **Field of Research Assignment:** Every category MUST be assigned to exactly ONE of the 23 field of research divisions using the descriptive enum name (e.g., "INFORMATION_COMPUTING_SCIENCES", "PHYSICAL_SCIENCES", "ENGINEERING"). The field_of_research field should contain the full enum name as a string.
*   **Available field of research Divisions:** AGRICULTURAL_VETERINARY_FOOD_SCIENCES, BIOLOGICAL_SCIENCES, BIOMEDICAL_CLINICAL_SCIENCES, BUILT_ENVIRONMENT_DESIGN, CHEMICAL_SCIENCES, COMMERCE_MANAGEMENT_TOURISM_SERVICES, CREATIVE_ARTS_WRITING, EARTH_SCIENCES, ECONOMICS, EDUCATION, ENGINEERING, ENVIRONMENTAL_SCIENCES, HEALTH_SCIENCES, HISTORY_HERITAGE_ARCHAEOLOGY, HUMAN_SOCIETY, INDIGENOUS_STUDIES, INFORMATION_COMPUTING_SCIENCES, LANGUAGE_COMMUNICATION_CULTURE, LAW_LEGAL_STUDIES, MATHEMATICAL_SCIENCES, PHILOSOPHY_RELIGIOUS_STUDIES, PHYSICAL_SCIENCES, PSYCHOLOGY
*   **Category Naming:** Each category requires a short, descriptive `name` (ideally under 4 words) that captures its core focus.
*   **High-Quality Descriptions:** The `description` for each category must be detailed and insightful. It should explain the category's scope and key focus areas, directly reflecting the keywords it contains.

**4. MANDATORY "Unknown" Category for Outliers:**
*   **Creation:** You MUST create a category with the exact name `Unknown`. This category serves as a container for any keywords that are true outliers.
*   **Condition for Use:** Place a keyword in the `Unknown` category ONLY if, after careful analysis, it cannot be logically grouped with any other keywords to form a coherent, thematic category. This is the designated place for single, disparate concepts that do not fit elsewhere.
*   **Description Requirement:** The `description` for the `Unknown` category must explicitly state: "This category contains disparate keywords and technologies that do not fit into the other defined domains and represent potential standalone areas of research."
*   **Field of Research Assignment:** For the `Unknown` category, analyze the keywords placed within it and assign the field of research division that represents the best possible fit for the majority of these keywords, or select "HUMAN_SOCIETY" if no clear fit emerges as it serves as the most general multidisciplinary division.
"""


def _format_keywords(records: Sequence[dict]) -> str:
    entries: Iterable[str] = (config.Keywords.template(record) for record in records)
    return "\n".join(entries)


async def _request_categories(client: AsyncOpenAI, batch_text: str) -> CategoryList:
    output_text = await llm_client.call_json_schema(
        client,
        model=config.OPENAI_MODEL,
        system_prompt=CATEGORISE_SYSTEM_PROMPT,
        user_content=batch_text,
        schema_name="keyword_categories",
        schema=CategoryList.model_json_schema(),
    )
    return CategoryList.model_validate_json(output_text)


async def categorise_async(input_dir: Path, output_dir: Path, concurrency: int = config.CONCURRENCY) -> List[str]:
    batches = list(iter_jsonl_batches(input_dir))
    if not batches:
        return []

    outputs = prepare_category_outputs(output_dir)
    try:
        async with llm_client.async_client() as client:

            async def handler(batch: JsonlBatch) -> tuple[int, str]:
                records = list(stream_jsonl(batch.path))
                batch_text = _format_keywords(records)
                categories = await _request_categories(client, batch_text)
                write_categorisation_result(records, categories, outputs)
                return batch.index, batch.batch_id

            processed = await run_with_concurrency(
                batches,
                handler,
                concurrency=concurrency,
                progress_description="Categorising keywords",
                progress_unit="batch",
            )
    finally:
        outputs.close()

    processed.sort(key=lambda item: item[0])
    return [batch_id for _, batch_id in processed]


def categorise(input_dir: Path | str, output_dir: Path | str, concurrency: int = config.CONCURRENCY) -> List[str]:
    return asyncio.run(categorise_async(Path(input_dir), Path(output_dir), concurrency=concurrency))
