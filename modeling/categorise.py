import json
import math
import time
from pathlib import Path

import chromadb
import numpy as np
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, MemoryDataset, Sample, json_dataset
from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks
from inspect_ai.model import GenerateConfig, ResponseSchema, get_model
from inspect_ai.scorer import (INCORRECT, NOANSWER, Metric, SampleScore, Score,
                               Target, mean, metric, scorer)
from inspect_ai.solver import (TaskState, chain_of_thought, generate,
                               prompt_template, solver, system_message,
                               use_tools, user_message)
from inspect_ai.util import json_schema
from pydantic import BaseModel, Field

from config import CONFIG
from database import get_db_manager
from metric import total
from models import Category, CategoryList



@scorer(metrics=[mean()])
def keywords_coverage():

    async def score(state: TaskState, target: Target) -> Score:
        """Score based on keyword coverage discrepancy."""
        result = json.loads(state.output.completion)
        categories_list = result["categories"]

        # num_covered_keywords = sum([len(category["keywords"]) for category in categories_list if "keywords" in category])
        output_terms = [kw for category in categories_list for kw in category['keywords']]
        # print(state.metadata)
        input_keywords = state.metadata['keywords']
        input_terms = [kw['term'] for kw in input_keywords]

        covered_terms = set(input_terms).intersection(output_terms)
        missed_terms = set(input_terms).difference(output_terms)
        missed_terms = [term for term in input_terms if term in missed_terms]

        return Score(value=len(covered_terms) / len(input_terms), explanation=f"Coverage Score (missing {missed_terms})", metadata={"missing": list(missed_terms)})

    return score



@hooks(name="CategoriseOutputHook", description="Hook to save categorisation results to ChromaDB")
class CategoriseOutputHook(Hooks):
    def __init__(self):
        self.missing = []
        self.unknown = []
        self.output = []
        self.db_manager = get_db_manager()
        
    async def on_sample_end(self, data: SampleEnd) -> None:
        """Save keywords extraction results."""
        try:
            result_json = json.loads(data.sample.output.completion)
            self.output.extend(result_json['categories'])
            self.missing.extend(data.sample.scores['keywords_coverage'].metadata['missing'])
            for category in result_json['categories']:
                if category['name'] == "Unknown":
                    self.unknown.extend(category['keywords'])
        except Exception as e:
            print(f"❌ Error processing sample end: {e}")

    async def on_task_end(self, data: TaskEnd) -> None:
        """Aggregate and save all categorisation results to ChromaDB."""
        try:
            aggregated_result = {
                "categories": self.output,
                "missing_keywords": self.missing,
                "unknown_keywords": self.unknown
            }

            # Store in ChromaDB using unified manager
            self.db_manager.store_categories(aggregated_result)
            print("✅ Categorization results stored in ChromaDB")
        except Exception as e:
            print(f"❌ Error storing categorization results: {e}")

def get_num_batches() -> int:
    """
    Calculate the number of batches based on total available keywords and batch size.
    
    Returns:
        int: Number of batches needed to process all keywords
    """
    try:
        db_manager = get_db_manager()
        keywords = db_manager.get_all_keywords()
        total_keywords = len(keywords)
        
        # Calculate number of batches needed
        num_batches = max(1, math.ceil(total_keywords / CONFIG.batch_size))
        return min(num_batches, 50)  # Cap at 50 batches for practical reasons
    except Exception as e:
        print(f"Error calculating number of batches: {e}")
        return 1
    





def load_for_codes_as_prompt():
    """Load FOR codes and return formatted template string for context."""
    if not CONFIG.for_codes_cleaned_path.exists():
        raise FileNotFoundError(f"FOR codes file not found at {CONFIG.for_codes_cleaned_path}")
    
    with open(CONFIG.for_codes_cleaned_path, 'r', encoding='utf-8') as f:
        for_codes = json.load(f)
    
    # Build formatted template string
    template_lines = []
    template_lines.append("Available FOR Code Divisions:")
    template_lines.append("=" * 50)
    
    # Extract and format top-level divisions (2-digit codes)
    for code in sorted(for_codes.keys()):
        if len(code) == 2:  # Top-level divisions are 2-digit codes
            data = for_codes[code]
            template_lines.append(f"\nFOR Code {code}: {data['name']}")
            template_lines.append("-" * (len(f"FOR Code {code}: {data['name']}")))
            
            description = data.get('definition', {}).get('description', '')
            if description:
                template_lines.append(f"Description: {description}")
            
            includes = data.get('definition', {}).get('includes', [])
            if includes:
                template_lines.append("Includes:")
                for item in includes:
                    # Clean up the item (remove trailing semicolons, etc.)
                    clean_item = item.rstrip(';').rstrip(',').strip()
                    template_lines.append(f"  • {clean_item}")
    
    return "\n".join(template_lines)

def keywords_to_sample(keywords, idx):
    entries = []
    for kw in keywords:
        entry = f"**{kw['term']}** ({kw.get('type', 'general')}): {kw.get('description', 'No description available')}"
        entries.append(entry)
    return Sample(id=f"categorise_batch{idx}", input="\n".join(entries), metadata={"keywords": keywords})

def load_dataset(batch_size: int, keywords_type: str):
    """
    Load keywords from ChromaDB iteratively without sampling.
    Include FOR codes context for categorization.
    
    Args:
        batch_size: Number of keywords per batch (used for chunking the full dataset)
        keywords_type: Type of keywords to use
    """
    from typing import List

    from models import Keyword

    # Load keywords from ChromaDB
    db_manager = get_db_manager()
    keywords_data = db_manager.get_all_keywords()

    if keywords_type is None:
        filtered = keywords_data
    else:
        filtered = list(filter(lambda kw: kw.get('type') == keywords_type, keywords_data))

    batches = np.array_split(np.array(filtered), len(filtered) // batch_size + 1)
    samples = list(map(keywords_to_sample, batches, range(len(batches))))

    return MemoryDataset(samples)

# batch_size=BATCH_SIZE
# keywords_type=CONFIG.keywords_type
@task
def categorise(batch_size: int = None, keywords_type: str = None, num_batches: int = None):
    """
    Categorise keywords into research categories with comprehensive scoring.
    
    This task processes keywords iteratively in batches without sampling or shuffling,
    ensuring all keywords are processed in a consistent order. The LLM determines the 
    appropriate number of categories based on natural groupings in the keyword data.
    
    Args:
        batch_size: Number of keywords per batch (used for chunking the dataset, default: from config)
        keywords_type: Type of keywords to use (default: from config)
        num_batches: Number of batches to generate (default: auto-calculated from total keywords)
    """
    # Use config defaults if not provided
    if batch_size is None:
        batch_size = CONFIG.batch_size
    if keywords_type is None:
        keywords_type = CONFIG.keywords_type
        
    # Auto-calculate num_batches if not provided
    if num_batches is None:
        num_batches = get_num_batches()

    # Create dataset from ChromaDB
    dataset = load_dataset(batch_size=batch_size, keywords_type=keywords_type)
    
    import pandas as pd
    return Task(
        dataset=dataset,
        solver=[
            system_message(str(CONFIG.prompts_dir / "categorise.txt")),
            system_message(f"\nFOR Code Context: \n{load_for_codes_as_prompt()}"),
            generate(),
        ],
        scorer=[
            keywords_coverage()
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="Categories",
                json_schema=json_schema(CategoryList),
                strict=True
            ),
        ),
        hooks=["CategoriseOutputHook"]
    )

