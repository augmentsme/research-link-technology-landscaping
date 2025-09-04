from inspect_ai import Task, task
import shutil

import jsonlines
import config
from pathlib import Path
import utils
import pandas as pd
import numpy as np
from models import Category, CategoryList
from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks
from inspect_ai.dataset import json_dataset, FieldSpec, Sample, MemoryDataset
import json
from inspect_ai.solver import system_message, generate
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.util import json_schema

import json 
from scorer import keywords_coverage


@hooks(name="CategoriseOutputHook", description="Hook to save categorisation results as JSON files")
class CategoriseOutputHook(Hooks):
    def __init__(self):
        self.output_write = jsonlines.open(config.Categories.category_proposal_path, 'w')
        self.unknown_write = jsonlines.open(config.Categories.unknown_keywords_path, 'w')
        self.missing_write = jsonlines.open(config.Categories.missing_keywords_path, 'w')
    async def on_sample_end(self, data: SampleEnd) -> None:
        """Save keywords extraction results."""

        result_json = json.loads(data.sample.output.completion)
        self.output_write.write_all(result_json['categories'])
        self.missing_write.write_all(data.sample.scores['keywords_coverage'].metadata['missing'])
        for category in result_json['categories']:
            if category['name'] == "Unknown":
                self.unknown_write.write_all(category['keywords'])

    async def on_task_end(self, data: TaskEnd) -> None:
        """Aggregate and save all categorisation results at the end of the task."""
        self.output_write.close()
        self.unknown_write.close()
        self.missing_write.close()



def load_for_codes():
    """Load FOR codes and return formatted template string for context."""
    if not config.FOR_CODES_CLEANED_PATH.exists():
        raise FileNotFoundError(f"FOR codes file not found at {config.FOR_CODES_CLEANED_PATH}")
    
    with open(config.FOR_CODES_CLEANED_PATH, 'r', encoding='utf-8') as f:
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
    return Sample(id=f"categorise_batch{idx}", input="\n".join(entries), metadata={"keywords": [kw['term'] for kw in keywords]})

def load_dataset(keywords_path: Path, batch_size: int, keywords_type: str):
    """
    Load keywords from the extracted keywords dataset iteratively without sampling.
    Include FOR codes context for categorization.
    
    Args:
        batch_size: Number of keywords per batch (used for chunking the full dataset)
        keywords_type: Type of keywords to use
        num_batches: Number of batches to generate
    """

    with open(keywords_path, 'r', encoding='utf-8') as f:
        keywords_data = json.load(f)

    if keywords_type is None:
        filtered = keywords_data
    else:
        filtered = list(filter(lambda kw: kw.get('type') == keywords_type, keywords_data))

    batches = np.array_split(np.array(filtered), len(filtered) // batch_size + 1)
    samples = list(map(keywords_to_sample, batches, range(len(batches))))

    return MemoryDataset(samples)


@task
def categorise(keywords_path = config.Keywords.keywords_path, batch_size: int = config.Categories.batch_size, keywords_type: str = config.Categories.keywords_type):
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

    if not keywords_path.exists():
        shutil.copy(config.Keywords.extracted_keywords_path, keywords_path)

    dataset = load_dataset(keywords_path=keywords_path, batch_size=batch_size, keywords_type=keywords_type)

    return Task(
        dataset=dataset,
        solver=[
            system_message(str(config.PROMPTS_DIR / "categorise.txt")),
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

