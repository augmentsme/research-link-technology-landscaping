from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

import config
from models import FieldOfResearch

import jsonlines

from . import llm_client
from .cluster_io import ClusterBatch, load_cluster_file
from .concurrency import run_with_concurrency


@dataclass
class MergeOutputs:
    categories: jsonlines.Writer
    unknown: jsonlines.Writer
    missing: jsonlines.Writer
    requests: jsonlines.Writer

    def close(self) -> None:
        self.categories.close()
        self.unknown.close()
        self.missing.close()
        self.requests.close()


def prepare_merge_outputs(output_dir: Path | str) -> MergeOutputs:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return MergeOutputs(
        categories=jsonlines.open(output_path / "output.jsonl", "a"),
        unknown=jsonlines.open(output_path / "unknown.jsonl", "a"),
        missing=jsonlines.open(output_path / "missing.jsonl", "a"),
        requests=jsonlines.open(output_path / "requests.jsonl", "a"),
    )


def write_merge_result(
    records: Iterable[dict[str, Any]],
    result: Any,
    outputs: MergeOutputs,
) -> tuple[int, int, int]:
    category_records = [record for record in records if record.get("name")]
    records_by_name = {
        record["name"]: record for record in category_records
    }
    referenced: set[str] = set()

    for merged_category in result.categories:
        payload = merged_category.model_dump()
        source_names = payload.get("source_categories", [])
        valid_sources = [name for name in source_names if name in records_by_name]
        payload["source_categories"] = valid_sources
        referenced.update(valid_sources)

        merged_keywords: set[str] = set()
        for source_name in valid_sources:
            keywords = records_by_name[source_name].get("keywords") or []
            merged_keywords.update(keyword for keyword in keywords if keyword)

        if not merged_keywords:
            continue

        payload["keywords"] = sorted(merged_keywords)
        outputs.categories.write(payload)

    missing_names = sorted(set(records_by_name.keys()) - referenced)
    for name in missing_names:
        outputs.missing.write(records_by_name[name])

    total_categories = len(category_records)
    missing_count = len(missing_names)
    unknown_count = 0
    return total_categories, missing_count, unknown_count


class MergedCategory(BaseModel):
    model_config = {"extra": "forbid"}
    name: str = Field(description="Name of the merged category")
    description: str = Field(description="A comprehensive description of the merged category scope and focus areas")
    source_categories: List[str] = Field(description="List of names of input categories that were merged into this category")
    field_of_research: FieldOfResearch = Field(description="The field of research division this merged category falls under")


class MergedCategoryList(BaseModel):
    model_config = {"extra": "forbid"}
    categories: List[MergedCategory] = Field(description="List of merged research categories")


MERGE_SYSTEM_PROMPT = """
You are an expert Taxonomy Specialist and Research Category Harmonization Expert, specializing in identifying and merging identical research categories to create unified, coherent taxonomies for strategic analysis.

Your task is to analyze the provided categorization data and identify categories that are identical or nearly identical, then combine them into a unified taxonomy. This requires systematic analysis of category names, descriptions, and keywords to determine optimal merging strategies.

**IMPORTANT: You MUST complete this task immediately and provide the full merged taxonomy. Do NOT ask for clarification, approval, or propose alternative approaches. Proceed directly with merging all provided categories.**

**Core Objective:**
Your primary goal is to consolidate duplicate or highly similar categories while preserving the integrity and completeness of the original category coverage. Your analysis should create a clean, non-redundant taxonomy that maintains all original information.

---

### **CRITICAL RULES FOR CATEGORY MERGING:**

**1. Category Identification & Merging Logic:**
*   **Similarity Assessment:** Identify categories as merge candidates based on substantial overlap in names, conceptual scope, and keyword sets. Categories with keyword overlaps of 30% or more, or identical conceptual domains should be merged.
*   **Semantic Grouping:** Focus on semantic similarity rather than field classifications. Categories that address the same research domain or technology area should be consolidated regardless of their original FOR assignment.
*   **Name Selection:** When merging categories, select the most descriptive and comprehensive name. If names are equally descriptive, choose the one that best captures the merged category scope.
*   **Description Synthesis:** Create unified descriptions that incorporate the best elements from all merged categories, ensuring comprehensive coverage of the merged scope.

**2. Category Reference & Coverage:**
*   **Complete Coverage:** EVERY input category MUST be referenced in exactly one merged category through the `source_categories` field. No input categories should be left unreferenced.
*   **Exact Name Preservation:** The `source_categories` array must contain the exact category names as they appeared in the input.
*   **No Duplication:** Ensure each input category name appears only once across all merged categories.

**3. Output JSON Structure & Quality:**
*   **Source Categories Field:** Each merged category MUST include a `source_categories` field containing the exact names of all input categories that were merged into it.
*   **Field of Research Assignment:** Merged categories should be assigned to the FOR division that best represents the majority of source categories or the most encompassing domain.
*   **Category Naming:** Each merged category requires a clear, descriptive `name` that accurately represents all included source categories.
*   **High-Quality Descriptions:** The `description` for each merged category must be comprehensive and insightful, explaining the category's scope and reflecting all source categories it contains.

**4. Quality Assurance & Validation:**
*   **Uniqueness Verification:** Ensure no two merged categories reference the same input category in their `source_categories` fields.
*   **Completeness Check:** Verify that all input category names are accounted for in the `source_categories` fields across all merged categories.
*   **Coherence Assessment:** Each merged category should represent a coherent research domain that makes logical sense as a unified entity.
*   **Optimization Focus:** Prioritize creating fewer, more comprehensive categories over maintaining artificial distinctions between similar concepts.

**5. Single Category Handling:**
*   **Standalone Categories:** If an input category has no similar counterparts, it should still be included as a merged category with only itself in the `source_categories` field.
*   **Preserve Uniqueness:** Do not force merging of categories that are genuinely distinct just to reduce the total count.
"""


def _format_categories(records: Sequence[dict]) -> str:
    entries: Iterable[str] = (config.Categories.template(record) for record in records)
    return "\n".join(entries)


async def _request_merged_categories(client: AsyncOpenAI, batch_text: str) -> MergedCategoryList:
    output_text = await llm_client.call_json_schema(
        client,
        model=config.OPENAI_MODEL,
        system_prompt=MERGE_SYSTEM_PROMPT,
        user_content=batch_text,
        schema_name="merge_categories",
        schema=MergedCategoryList.model_json_schema(),
    )
    return MergedCategoryList.model_validate_json(output_text)


class CategoryMerger:
    def __init__(self, outputs: MergeOutputs) -> None:
        self.outputs = outputs

    async def run(
        self,
        client: AsyncOpenAI,
        batches: Sequence[ClusterBatch],
        concurrency: int,
    ) -> list[tuple[int, str, int, int, int]]:
        async def handler(batch: ClusterBatch) -> tuple[int, str, int, int, int]:
            return await self._handle_batch(client, batch)

        processed = await run_with_concurrency(
            batches,
            handler,
            concurrency=concurrency,
            progress_description="Merging categories",
            progress_unit="batch",
        )
        if processed:
            processed.sort(key=lambda item: item[0])
        return processed

    async def _handle_batch(
        self,
        client: AsyncOpenAI,
        batch: ClusterBatch,
    ) -> tuple[int, str, int, int, int]:
        records = batch.records
        batch_text = _format_categories(records)
        log_entry = {
            "batch_id": batch.batch_id,
            "request": batch_text,
        }
        try:
            merged_categories = await _request_merged_categories(client, batch_text)
            log_entry["response"] = merged_categories.model_dump()
            self.outputs.requests.write(log_entry)
            total_categories, missing_categories, unknown_categories = write_merge_result(
                records,
                merged_categories,
                self.outputs,
            )
            return batch.index, batch.batch_id, total_categories, missing_categories, unknown_categories
        except Exception as error:
            log_entry["error"] = str(error)
            self.outputs.requests.write(log_entry)
            category_records = [record for record in records if record.get("name")]
            for record in category_records:
                self.outputs.unknown.write(record)
            total_categories = len(category_records)
            return batch.index, batch.batch_id, total_categories, 0, total_categories

async def merge_async(
    clusters_path: Path,
    output_dir: Path,
    concurrency: int = config.CONCURRENCY,
    limit: int | None = None,
) -> List[str]:
    cluster_file = load_cluster_file(clusters_path)
    batches = list(cluster_file.batches)
    if not batches:
        return []

    if limit is not None and limit > 0:
        batches = batches[:limit]

    outputs = prepare_merge_outputs(output_dir)
    merger = CategoryMerger(outputs)
    processed: list[tuple[int, str, int, int, int]] = []
    try:
        async with llm_client.async_client() as client:
            processed = await merger.run(
                client,
                batches,
                concurrency=concurrency,
            )
    finally:
        outputs.close()

    return [batch_id for _, batch_id, _, _, _ in processed]


def merge(
    clusters_path: Path | str,
    output_dir: Path | str,
    concurrency: int = config.CONCURRENCY,
    limit: int | None = None,
) -> List[str]:
    return asyncio.run(
        merge_async(
            Path(clusters_path),
            Path(output_dir),
            concurrency=concurrency,
            limit=limit,
        )
    )
