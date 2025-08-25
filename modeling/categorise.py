from inspect_ai import Task, task
import numpy as np
from models import Category, CategoryList
from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks
from inspect_ai.dataset import json_dataset, FieldSpec, Sample, MemoryDataset
from config import CATEGORY_PATH, PROMPTS_DIR, FOR_CODES_CLEANED_PATH, BATCH_SIZE, KEYWORDS_TYPE, CATEGORY_PROPOSAL_PATH
from config import KEYWORDS_PATH, EXTRACTED_KEYWORDS_PATH
import time
import math
import json
from inspect_ai.solver import system_message, generate, user_message, TaskState, use_tools, chain_of_thought, solver, prompt_template
from inspect_ai.model import GenerateConfig, ResponseSchema, get_model
from inspect_ai.util import json_schema
from inspect_ai.scorer import scorer, Score, Target, metric, Metric, SampleScore, INCORRECT, NOANSWER, mean

import json 
from metric import total
from pydantic import BaseModel, Field
from pathlib import Path

# @metric
# def collect_missing() -> Metric:
#     def metric_impl(scores: list[SampleScore]) -> float:
#         return [item.score.metadata["missing"] for item in scores]
#     return metric_impl
# @metric
# def collect_unknown() -> Metric:
#     def metric_impl(scores: list[SampleScore]) -> float:
#         return [item.score.metadata["unknown"] for item in scores]
#     return metric_impl

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



@hooks(name="CategoriseOutputHook", description="Hook to save categorisation results as JSON files")
class CategoriseOutputHook(Hooks):
    def __init__(self):
        self.missing = []
        self.unknown = []
        self.output = []
    async def on_sample_end(self, data: SampleEnd) -> None:
        """Save keywords extraction results."""

        result_json = json.loads(data.sample.output.completion)
        self.output.extend(result_json['categories'])
        self.missing.extend(data.sample.scores['keywords_coverage'].metadata['missing'])
        for category in result_json['categories']:
            if category['name'] == "Unknown":
                self.unknown.extend(category['keywords'])

    async def on_task_end(self, data: TaskEnd) -> None:
        """Aggregate and save all categorisation results at the end of the task."""

        aggregated_result = {
            "categories": self.output,
            "missing_keywords": self.missing,
            "unknown_keywords": self.unknown
        }

        with open(CATEGORY_PROPOSAL_PATH, 'w', encoding='utf-8') as f:
            json.dump(aggregated_result, f, ensure_ascii=False, indent=2)

def get_num_batches() -> int:
    """
    Calculate the number of batches based on total available keywords and batch size.
    
    Returns:
        int: Number of batches needed to process all keywords
    """
    keywords_file = KEYWORDS_PATH
    


    with open(keywords_file, 'r', encoding='utf-8') as f:
        keywords_data = json.load(f)
    
    # Count total keywords based on the structure
    if isinstance(keywords_data, list):
        # New flattened structure - array of keyword objects
        total_keywords = len([kw for kw in keywords_data if isinstance(kw, dict) and 'term' in kw])
    else:
        # Fallback for old structure
        total_keywords = len(keywords_data.get('keywords', []))
    
    # Calculate number of batches needed
    num_batches = max(1, math.ceil(total_keywords / BATCH_SIZE))
    return min(num_batches, 50)  # Cap at 50 batches for practical reasons
    





def load_for_codes():
    """Load FOR codes and return formatted template string for context."""
    if not FOR_CODES_CLEANED_PATH.exists():
        raise FileNotFoundError(f"FOR codes file not found at {FOR_CODES_CLEANED_PATH}")
    
    with open(FOR_CODES_CLEANED_PATH, 'r', encoding='utf-8') as f:
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

def load_dataset(keywords_path, batch_size: int, keywords_type: str):
    """
    Load keywords from the extracted keywords dataset iteratively without sampling.
    Include FOR codes context for categorization.
    
    Args:
        batch_size: Number of keywords per batch (used for chunking the full dataset)
        keywords_type: Type of keywords to use
        num_batches: Number of batches to generate
    """
    from models import Keyword
    from typing import List
    with open(keywords_path, 'r', encoding='utf-8') as f:
        keywords_data: List[Keyword] = json.load(f)

    if keywords_type is None:
        filtered = keywords_data
    else:
        filtered = list(filter(lambda kw: kw.get('type') == keywords_type, keywords_data))

    batches = np.array_split(np.array(filtered), len(filtered) // batch_size + 1)
    samples = list(map(keywords_to_sample, batches, range(len(batches))))

    return MemoryDataset(samples)

# batch_size=BATCH_SIZE
# keywords_type=KEYWORDS_TYPE
@task
def categorise(batch_size: int = BATCH_SIZE, keywords_type: str = KEYWORDS_TYPE, num_batches: int = None):
    """
    Categorise keywords into research categories with comprehensive scoring.
    
    This task processes keywords iteratively in batches without sampling or shuffling,
    ensuring all keywords are processed in a consistent order. The LLM determines the 
    appropriate number of categories based on natural groupings in the keyword data.
    
    Args:
        batch_size: Number of keywords per batch (used for chunking the dataset, default: 1000)
        keywords_type: Type of keywords to use (default: "keywords")
        num_batches: Number of batches to generate (default: auto-calculated from total keywords)
    """
    # Auto-calculate num_batches if not provided
    import shutil
    if not KEYWORDS_PATH.exists():
        shutil.copy(EXTRACTED_KEYWORDS_PATH, KEYWORDS_PATH)

    if num_batches is None:
        num_batches = get_num_batches()

    # Create dataset with metadata
    dataset = load_dataset(keywords_path=KEYWORDS_PATH, batch_size=batch_size, keywords_type=keywords_type)
    import pandas as pd
    return Task(
        dataset=dataset,
        solver=[
            system_message(str(PROMPTS_DIR / "categorise.txt")),
            system_message(f"\nFOR Code Context: \n{load_for_codes()}"),
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

