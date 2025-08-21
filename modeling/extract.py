
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset, FieldSpec, Sample
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.solver import system_message, generate, user_message
from inspect_ai.util import json_schema


from config import GRANTS_FILE, PROMPTS_DIR, RESULTS_DIR, EXTRACTED_KEYWORDS_DIR, EXTRACTED_KEYWORDS_PATH

import json
from typing import List
from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks
from inspect_ai.solver import TaskState
from pydantic import BaseModel, Field
from inspect_ai.scorer import scorer, Score, Target, metric, Metric, SampleScore


from metric import count

@scorer(metrics=[count()])
def keywords_counter():
    """
    Scorer that counts the total number of keywords extracted in the extract task.
    
    This provides metrics for keyword extraction volume and effectiveness,
    counting keywords across all categories (keywords, methodology_keywords,
    application_keywords, technology_keywords).
    
    Returns:
        Score with total keyword count as the metric
    """
    
    async def score(state: TaskState, target: Target) -> Score:
        """Score based on total extracted keyword count."""
        
        if not state.output or not state.output.completion:
            return Score(
                value="incorrect", 
                explanation="No keyword extraction output to score"
            )
        
        try:
            # Parse the keyword extraction result
            result = json.loads(state.output.completion)
            
            # Count keywords from all categories
            total_keywords = 0
            category_counts = {}
            
            # Standard keyword categories from KeywordsExtractionOutput
            keyword_categories = [
                "keywords",
                "methodology_keywords", 
                "application_keywords",
                "technology_keywords"
            ]
            
            for category in keyword_categories:
                if category in result and isinstance(result[category], list):
                    count = len(result[category])
                    category_counts[category] = count
                    total_keywords += count
            
            if total_keywords > 0:
                category_breakdown = ", ".join([
                    f"{cat.replace('_', ' ')}: {count}" 
                    for cat, count in category_counts.items() 
                    if count > 0
                ])
                explanation = f"Extracted {total_keywords} total keywords ({category_breakdown})"
            else:
                explanation = "No keywords extracted"
            
            return Score(
                value=total_keywords,
                explanation=explanation
            )
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return Score(
                value="incorrect", 
                explanation=f"Error parsing keyword extraction result: {str(e)}"
            )
    
    return score

class KeywordsExtractionOutput(BaseModel):
    """Pydantic model for structured keywords extraction output."""
    model_config = {"extra": "forbid"}
    
    keywords: List[str] = Field(description="Most relevant keywords capturing the research content")
    methodology_keywords: List[str] = Field(description="Keywords related to research methodologies or approaches")
    application_keywords: List[str] = Field(description="Keywords related to target applications or outcomes")
    technology_keywords: List[str] = Field(description="Keywords related to technologies or tools mentioned")


@hooks(name="KeywordsExtractionHook", description="Hook to save keywords extraction results as JSON files")
class KeywordsExtractionHook(Hooks):
    """Hook to save keywords extraction results as JSON files."""


    async def on_sample_end(self, data: SampleEnd) -> None:
        """Save keywords extraction results."""

        output_text = data.sample.output.completion
        result_json = json.loads(output_text)

        EXTRACTED_KEYWORDS_DIR.mkdir(parents=True, exist_ok=True)

        grant_id = data.sample.id
        grant_id_sanitized = grant_id.replace("/", "_")

        output_file = RESULTS_DIR / f"{grant_id_sanitized}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, indent=2, ensure_ascii=False)

    async def on_task_end(self, data: TaskEnd) -> None:
        # results_dir = EXTRACTED_KEYWORDS_DIR
        keywords = []
        for file in EXTRACTED_KEYWORDS_DIR.glob("*.json"):
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['id'] = file.stem.replace("_", "/")
                keywords.append(data)
        with open(EXTRACTED_KEYWORDS_PATH, 'w', encoding='utf-8') as f:
            json.dump(keywords, f, indent=2, ensure_ascii=False)


def record_to_sample(record: dict) -> Sample:
    return Sample(
        id=record["id"],
        input=f""""
**Title**: {record["title"]}
**Summary**: 
{record["grant_summary"]}
                """,
        metadata={
            "title": record["title"],
            "summary": record["grant_summary"],
            "funding_amount": record["funding_amount"],
            "funder": record["funder"],
            "start_year": record["start_year"],
            "end_year": record["end_year"],
        }
    )


@task
def extract() -> Task:
    return Task(
        dataset=json_dataset(
            str(GRANTS_FILE),
            record_to_sample
        ),
        solver=[
            system_message(str(PROMPTS_DIR / "extract.txt")),
            generate()
        ],
        scorer=[
            keywords_counter()
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="keywords_extraction",
                json_schema=json_schema(KeywordsExtractionOutput),
                strict=True
            )
        ),
        hooks=["KeywordsExtractionHook"],
    )
