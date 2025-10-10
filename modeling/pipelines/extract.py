from __future__ import annotations

import asyncio
from itertools import chain
from typing import Iterable, List, Optional, Tuple

import jsonlines
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI

import config
import utils
from models import KeywordsList
from process import postprocess_keywords

from . import llm_client


SYSTEM_PROMPT = """

You are an expert research analyst with deep knowledge across multiple academic disciplines and a keen eye for emerging research trends.

Your task is to extract meaningful keywords from research grant information that would be useful for:
- Identifying emerging research domains and interdisciplinary areas
- Discovering novel methodologies and cutting-edge approaches
- Tracking innovative technologies and emerging tools
- Finding related research projects working on similar frontiers
- Understanding emerging research trends and future directions

**FUNDAMENTAL REQUIREMENT - Keywords Must Exist in Grant Text:**
**ALL KEYWORDS MUST BE DIRECTLY PRESENT IN OR CLEARLY DERIVABLE FROM THE PROVIDED GRANT TEXT (INCLUDING BOTH TITLE AND DESCRIPTION).**
- Only extract keywords that appear explicitly in the grant title, description, or summary
- Keywords can be technical terms, methodologies, technologies, or concepts mentioned in the text
- Do NOT invent or infer keywords that are not clearly present in the source material
- If a concept is implied but not explicitly mentioned, DO NOT include it as a keyword

Focus on extracting keywords that highlight what's new, innovative, and emerging in the research landscape. Prioritize:
- Technical terms that represent novel concepts or emerging fields mentioned in the grant
- Methodologies that are cutting-edge or represent new approaches described in the text
- Technologies that are innovative or represent emerging tools referenced in the grant
- Applications that address new challenges or emerging needs as stated in the grant
- Scientific terminology that indicates research at the frontiers of knowledge as described

**CRITICAL REQUIREMENT - Avoid Overly Broad Keywords:**
- **Keywords must be specific and precise, not generic or overly broad**
- **Instead of "engineering," use "bio-integrated nano-photonics" or "quantum-enhanced engineering"**
- **Instead of "chemistry," use "supramolecular photochemistry" or "catalytic asymmetric synthesis"**
- **Instead of "data analysis," use "multi-modal time-series analysis" or "causal inference modeling"**
- **Instead of "artificial intelligence," use "graph neural networks" or "federated learning algorithms"**
- **Avoid general terms like "research," "development," "innovation," "technology," "analysis," "method"**
- **Each keyword should clearly indicate a specific domain, technique, or application area**
- **Do NOT extract specific country names**

**UNIQUENESS REQUIREMENT - Keywords Must Be Grant-Specific:**
- **REJECT keywords that could apply to 80% or more of research grants (e.g., "interdisciplinary research," "international collaboration," "innovative approach," "cutting-edge technology")**
- **REJECT administrative or process keywords (e.g., "project management," "research methodology," "data collection," "literature review")**
- **REJECT funding-related terms (e.g., "research funding," "grant application," "collaborative research," "research partnership")**
- **REJECT generic outcome terms (e.g., "scientific advancement," "knowledge creation," "research impact," "societal benefit")**
- **Each keyword should be SO SPECIFIC that it could only apply to this particular grant or a very small subset of similar grants**
- **If a keyword could reasonably appear in a generic grant template or boilerplate text, REJECT it**
- **Keywords should capture the UNIQUE technical essence that distinguishes this specific research from all other research**

**SPECIFICITY TEST:**
Before including any keyword, ask: "Could this keyword appear in 100+ different grant applications across various fields?" If YES, REJECT it.
Only extract keywords that are:
1. **Actually present in the grant text (title and description) (MANDATORY)**
2. Technically precise and domain-specific
3. Unique to the particular research approach or subject matter
4. Would help distinguish this grant from 95% of other grants in the database

**LENGTH REQUIREMENT:**
- **Each keyword must contain only 1-4 words maximum**
- **Use concise, specific terminology rather than lengthy phrases**
- **Examples: "quantum dots," "CRISPR-Cas9," "machine learning," "photonic crystals"**
- **Avoid: "advanced quantum dot synthesis techniques" (too long) → use "quantum dot synthesis"**

Provide accurate, specific keywords that capture the innovative and emerging aspects of the research while ensuring they are all grounded in the actual grant text (both title and description). Prioritize technical precision over comprehensiveness.


"""


def load_extract_dataset() -> List[dict]:
    records = config.Grants.load(as_dataframe=False)
    return list(records)


def finished_grants() -> Iterable[str]:
    if not config.Keywords.extracted_keywords_path.exists():
        return []
    keywords = utils.load_jsonl_file(config.Keywords.extracted_keywords_path, as_dataframe=True)
    if keywords.empty:
        return []
    return keywords.grant_id.unique()


def _grant_input(record: dict) -> str:
    return config.Grants.template(record)


async def _request_keywords(client: AsyncOpenAI, grant_text: str) -> KeywordsList:
    output_text = await llm_client.call_json_schema(
        client,
        model=config.OPENAI_MODEL,
        system_prompt=SYSTEM_PROMPT,
        user_content=grant_text,
        schema_name="keywords_extraction",
        schema=KeywordsList.model_json_schema(),
    )
    return KeywordsList.model_validate_json(output_text)


async def extract_async(filter_finished: bool = True, concurrency: int = 10) -> List[str]:
    records = load_extract_dataset()
    finished_ids = set(finished_grants()) if filter_finished else set()

    candidate_iter = (
        record for record in records if record.get("id") and record.get("id") not in finished_ids
    )
    try:
        first_target = next(candidate_iter)
    except StopIteration:
        return []

    targets = list(chain([first_target], candidate_iter))
    semaphore = asyncio.Semaphore(max(concurrency, 1))
    processed: List[Tuple[str, KeywordsList]] = []

    writer = jsonlines.open(config.Keywords.extracted_keywords_path, "a")
    try:
        async with llm_client.async_client() as client:

            async def run_single(record: dict) -> Optional[Tuple[str, KeywordsList]]:
                grant_id = record.get("id")
                if not grant_id:
                    return None
                grant_input = _grant_input(record)
                async with semaphore:
                    keywords = await _request_keywords(client, grant_input)
                return grant_id, keywords

            tasks = [asyncio.create_task(run_single(record)) for record in targets]
            for task in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Extracting keywords",
                unit="grant",
            ):
                result = await task
                if result:
                    processed.append(result)
                    _write_keywords(result[0], result[1], writer)
    finally:
        writer.close()

    postprocess_keywords()
    return [grant_id for grant_id, _ in processed]


def _write_keywords(grant_id: str, keywords: KeywordsList, writer: jsonlines.Writer) -> None:
    for keyword in keywords.keywords:
        payload = keyword.model_dump()
        payload["grant_id"] = grant_id
        writer.write(payload)


def extract(filter_finished: bool = True, concurrency: int = 10) -> List[str]:
    return asyncio.run(extract_async(filter_finished=filter_finished, concurrency=concurrency))
