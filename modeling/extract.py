
import json
import re
from enum import Enum
from typing import List

import chromadb
import typer
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, Sample, json_dataset
from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.scorer import (Metric, SampleScore, Score, Target, metric,
                               scorer)
from inspect_ai.solver import TaskState, generate, system_message
from inspect_ai.util import json_schema
from nltk.stem import PorterStemmer

from config import CONFIG
from database import get_db_manager
from grants import record_to_sample
from metric import total


# def normalize_keyword_term(term: str) -> str:
#     """
#     Normalize keyword terms for deduplication using NLTK stemming and lexical variations.
    
#     Args:
#         term: The original keyword term
        
#     Returns:
#         Normalized term for comparison
#     """
#     # Convert to lowercase
#     normalized = term.lower().strip()

#     normalized = re.sub(r'\s+', ' ', normalized)
#     normalized = normalized.replace('-', ' ').replace('_', ' ')
    
#     words = normalized.split()
#     stemmer = PorterStemmer()
#     stemmed_words = [stemmer.stem(word) for word in words]
#     stemmed_normalized = ' '.join(stemmed_words)
#     return stemmed_normalized



# def collect(keywords_dir, output_path):
#     keyword_map = {}  # normalized_term -> keyword_data
    
#     # Get all files and sort them to ensure consistent processing order
#     files = sorted(keywords_dir.glob("*.json"))
    
#     for file in files:
#         with open(file, 'r', encoding='utf-8') as f:
#             grant_data = json.load(f)
#             # Reverse the sanitization to get the true grant ID
#             grant_id = file.stem.replace("_", "/")  # Unsanitize filename to get true grant ID
            
#             for keyword in grant_data.get("keywords", []):
#                 if isinstance(keyword, dict) and "term" in keyword:
#                     normalized_term = normalize_keyword_term(keyword["term"])
                    
#                     if normalized_term in keyword_map:
#                         # Keyword already exists, merge entries
#                         existing = keyword_map[normalized_term]
#                         # Take type and description from the latest variant (current keyword)
#                         existing["type"] = keyword.get("type", existing.get("type"))
#                         existing["description"] = keyword.get("description", existing.get("description"))
#                         existing["term"] = keyword["term"]  # Update to latest term variant
#                         # Add grant ID if not already present
#                         if grant_id not in existing["grants"]:
#                             existing["grants"].append(grant_id)
#                     else:
#                         # New keyword, create entry
#                         keyword_map[normalized_term] = {
#                             "term": keyword["term"],  # Keep original term from first occurrence
#                             "type": keyword.get("type"),
#                             "description": keyword.get("description"),
#                             "grants": [grant_id]
#                         }

#     # Convert back to list format
#     all_keywords = list(keyword_map.values())

#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(all_keywords, f, ensure_ascii=False)

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

from models import Keyword, KeywordsExtractionOutput, KeywordType


class KeywordsExtractionHook(Hooks):
    """Hook to save keywords extraction results to ChromaDB."""
    
    def __init__(self):
        self.db_manager = get_db_manager()

    async def on_sample_end(self, data: SampleEnd) -> None:
        """Save keywords extraction results to ChromaDB."""
        try:
            output_text = data.sample.output.completion
            result_json = json.loads(output_text)
            grant_id = data.sample.id
            
            # Store keywords in ChromaDB
            self.db_manager.store_keywords(grant_id, result_json)
            
        except Exception as e:
            print(f"❌ Error storing keywords for grant {data.sample.id}: {e}")

    async def on_task_end(self, data: TaskEnd) -> None:
        """Task completed - keywords are now stored in ChromaDB."""
        total_keywords = self.db_manager.count_documents("extracted_keywords")
        print(f"✅ Task completed. Total keywords in ChromaDB: {total_keywords}")

def finished_grants():
    """Get list of grants that have been processed from ChromaDB."""
    try:
        db_manager = get_db_manager()
        
        # Get all unique grant IDs from the keywords collection  
        all_keywords = db_manager.get_all_keywords()
        grant_ids = list(set(keyword.get('grants', [None])[0] for keyword in all_keywords if keyword.get('grants')))
        return [gid for gid in grant_ids if gid is not None]
    except Exception as e:
        print(f"Warning: Could not access ChromaDB keywords: {e}")
        return []


def get_keywords_for_categorisation():
    """Get all keywords from ChromaDB for categorisation task."""
    try:
        db_manager = get_db_manager()
        return db_manager.get_all_keywords()
    except Exception as e:
        print(f"Error retrieving keywords from ChromaDB: {e}")
        return []


# Keywords extraction functionality has been integrated into the unified CLI (cli.py)
# Use: python cli.py data extract-keywords --help


@task
def extract() -> Task:
    finished = finished_grants()
    return Task(
        dataset=json_dataset(
            str(CONFIG.grants_file),
            record_to_sample
        ).filter(lambda s: s.id not in finished),
        solver=[
            system_message(str(CONFIG.prompts_dir / "extract.txt")),
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
