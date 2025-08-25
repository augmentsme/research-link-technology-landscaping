
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset, FieldSpec, Sample
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.solver import system_message, generate, user_message
from inspect_ai.util import json_schema


from config import GRANTS_FILE, PROMPTS_DIR, EXTRACTED_KEYWORDS_DIR, EXTRACTED_KEYWORDS_PATH

import json
from typing import List
from enum import Enum
from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks
from inspect_ai.solver import TaskState
from pydantic import BaseModel, Field
from inspect_ai.scorer import scorer, Score, Target, metric, Metric, SampleScore


from metric import total
import re

from nltk.stem import PorterStemmer

    



def normalize_keyword_term(term: str) -> str:
    """
    Normalize keyword terms for deduplication using NLTK stemming and lexical variations.
    
    Args:
        term: The original keyword term
        
    Returns:
        Normalized term for comparison
    """
    # Convert to lowercase
    normalized = term.lower().strip()

    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = normalized.replace('-', ' ').replace('_', ' ')
    
    words = normalized.split()
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_normalized = ' '.join(stemmed_words)
    return stemmed_normalized



def collect(keywords_dir, output_path):
    keyword_map = {}  # normalized_term -> keyword_data
    
    # Get all files and sort them to ensure consistent processing order
    files = sorted(keywords_dir.glob("*.json"))
    
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            grant_data = json.load(f)
            # Reverse the sanitization to get the true grant ID
            grant_id = file.stem.replace("_", "/")  # Unsanitize filename to get true grant ID
            
            for keyword in grant_data.get("keywords", []):
                if isinstance(keyword, dict) and "term" in keyword:
                    normalized_term = normalize_keyword_term(keyword["term"])
                    
                    if normalized_term in keyword_map:
                        # Keyword already exists, merge entries
                        existing = keyword_map[normalized_term]
                        # Take type and description from the latest variant (current keyword)
                        existing["type"] = keyword.get("type", existing.get("type"))
                        existing["description"] = keyword.get("description", existing.get("description"))
                        existing["term"] = keyword["term"]  # Update to latest term variant
                        # Add grant ID if not already present
                        if grant_id not in existing["grants"]:
                            existing["grants"].append(grant_id)
                    else:
                        # New keyword, create entry
                        keyword_map[normalized_term] = {
                            "term": keyword["term"],  # Keep original term from first occurrence
                            "type": keyword.get("type"),
                            "description": keyword.get("description"),
                            "grants": [grant_id]
                        }

    # Convert back to list format
    all_keywords = list(keyword_map.values())

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_keywords, f, ensure_ascii=False)

@scorer(metrics=[total()])
def keywords_counter():

    
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

from models import KeywordType, Keyword, KeywordsExtractionOutput

@hooks(name="KeywordsExtractionHook", description="Hook to save keywords extraction results as JSON files")
class KeywordsExtractionHook(Hooks):
    """Hook to save keywords extraction results as JSON files."""


    async def on_sample_end(self, data: SampleEnd) -> None:
        """Save keywords extraction results with grant references."""

        output_text = data.sample.output.completion
        result_json = json.loads(output_text)

        grant_id = data.sample.id
        
        # Populate grants field for keywords
        if "keywords" in result_json and isinstance(result_json["keywords"], list):
            for keyword in result_json["keywords"]:
                if isinstance(keyword, dict):
                    # Initialize grants field with current grant ID
                    if "grants" not in keyword:
                        keyword["grants"] = [grant_id]

        EXTRACTED_KEYWORDS_DIR.mkdir(parents=True, exist_ok=True)

        grant_id_sanitized = grant_id.replace("/", "_")

        output_file = EXTRACTED_KEYWORDS_DIR / f"{grant_id_sanitized}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, ensure_ascii=False)

    async def on_task_end(self, data: TaskEnd) -> None:
        """Flatten all keywords from all grants into a single top-level array."""
        collect(EXTRACTED_KEYWORDS_DIR, EXTRACTED_KEYWORDS_PATH)

def finished_grants():
    return [grant.stem.replace("_", "/") for grant in EXTRACTED_KEYWORDS_DIR.glob("*.json")]


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
    finished = finished_grants()
    return Task(
        dataset=json_dataset(
            str(GRANTS_FILE),
            record_to_sample
        ).filter(lambda s: s.id not in finished),
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
