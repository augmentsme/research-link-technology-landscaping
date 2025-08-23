
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


from metric import total

@scorer(metrics=[total()])
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
            
            # Count keywords from the flat structure
            total_keywords = 0
            type_counts = {}
            
            if "keywords" in result and isinstance(result["keywords"], list):
                keywords_list = result["keywords"]
                total_keywords = len(keywords_list)
                
                # Count by type if available
                for keyword in keywords_list:
                    if isinstance(keyword, dict) and "type" in keyword:
                        kw_type = keyword["type"]
                        type_counts[kw_type] = type_counts.get(kw_type, 0) + 1
            
            if total_keywords > 0:
                if type_counts:
                    type_breakdown = ", ".join([
                        f"{kw_type}: {count}" 
                        for kw_type, count in sorted(type_counts.items())
                    ])
                    explanation = f"Extracted {total_keywords} total keywords ({type_breakdown})"
                else:
                    explanation = f"Extracted {total_keywords} total keywords"
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

class Keyword(BaseModel):
    """Individual keyword with context and identifier."""
    model_config = {"extra": "forbid"}
    term: str = Field(description="The actual keyword or phrase")
    type: str = Field(description="Type of keyword: 'general', 'methodology', 'application', or 'technology'")
    description: str = Field(description="Short description explaining the context and relevance of this keyword within the research")

class KeywordsExtractionOutput(BaseModel):
    """Pydantic model for structured keywords extraction output."""
    model_config = {"extra": "forbid"}
    
    keywords: List[Keyword] = Field(description="List of all extracted keywords with their types, descriptions, and identifiers")


@hooks(name="KeywordsExtractionHook", description="Hook to save keywords extraction results as JSON files")
class KeywordsExtractionHook(Hooks):
    """Hook to save keywords extraction results as JSON files."""


    async def on_sample_end(self, data: SampleEnd) -> None:
        """Save keywords extraction results with auto-generated IDs."""

        output_text = data.sample.output.completion
        result_json = json.loads(output_text)

        grant_id = data.sample.id
        
        # Auto-generate IDs for keywords with grant ID prefix
        if "keywords" in result_json and isinstance(result_json["keywords"], list):
            for idx, keyword in enumerate(result_json["keywords"]):
                if isinstance(keyword, dict):
                    keyword["id"] = f"{grant_id}_{idx}"

        EXTRACTED_KEYWORDS_DIR.mkdir(parents=True, exist_ok=True)

        grant_id_sanitized = grant_id.replace("/", "_")

        output_file = EXTRACTED_KEYWORDS_DIR / f"{grant_id_sanitized}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, indent=2, ensure_ascii=False)

    async def on_task_end(self, data: TaskEnd) -> None:
        """Flatten all keywords from all grants into a single top-level array."""
        all_keywords = []
        
        for file in EXTRACTED_KEYWORDS_DIR.glob("*.json"):
            with open(file, 'r', encoding='utf-8') as f:
                grant_data = json.load(f)
                
                # Extract keywords from this grant and add to the flattened list
                if "keywords" in grant_data and isinstance(grant_data["keywords"], list):
                    all_keywords.extend(grant_data["keywords"])
        
        # Write flattened keywords array to the final output
        with open(EXTRACTED_KEYWORDS_PATH, 'w', encoding='utf-8') as f:
            json.dump(all_keywords, f, indent=2, ensure_ascii=False)


def record_to_sample(record: dict) -> Sample:
    return Sample(
        id=record["id"],
        input=f"""**Grant ID**: {record["id"]}
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
