
from inspect_ai._util.answer import answer_character, answer_index
from inspect_ai import Task, task
from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks
from inspect_ai.dataset import json_dataset, FieldSpec, Sample, MemoryDataset
from inspect_ai.solver._multiple_choice import parse_answers
from config import CATEGORY_PATH, REFINED_CATEGORY_PATH, COMPREHENSIVE_TAXONOMY_PATH, GRANTS_FILE, CLASSIFICATION_PATH
from inspect_ai.solver import system_message, generate, user_message, multiple_choice
from inspect_ai.scorer import model_graded_fact, answer, choice
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.util import json_schema

import json 
from pathlib import Path 

grant_template = lambda grant: f"""
Which category does this grant belong to?

**Title**: {grant["title"]}
**Summary**: 
{grant["grant_summary"]}
"""

category_template = lambda category: f"""
**Name**: {category["name"]}
**Description**: {category["description"]}
"""

refined_category_template = lambda category: f"""
**Name**: {category["name"]}
**Description**: {category["description"]}
**Subcategories**: {', '.join(category.get('subcategories', []))}
"""

def load_choices(category_file_path: Path):
    
    if not category_file_path.exists():
        raise FileNotFoundError(f"Category file not found at {category_file_path}")
    
    with open(category_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different data structures
    if isinstance(data, dict) and 'categories' in data:
        categories = data['categories']
    elif isinstance(data, list):
        categories = data
    else:
        raise ValueError(f"Unexpected data structure in {category_file_path}")

    choices = []
    for category in categories:
        choices.append(category_template(category))
    
    return choices

def load_comprehensive_taxonomy_choices(level: str = 'base'):
    """
    Load choices from comprehensive taxonomy file for a specific level.
    
    Args:
        level: Taxonomy level to use ('coarsened', 'base', 'refined')
    
    Returns:
        List of formatted choice strings for the specified level
    """
    if not COMPREHENSIVE_TAXONOMY_PATH.exists():
        raise FileNotFoundError(f"Comprehensive taxonomy file not found at {COMPREHENSIVE_TAXONOMY_PATH}")
    
    with open(COMPREHENSIVE_TAXONOMY_PATH, 'r', encoding='utf-8') as f:
        taxonomy_data = json.load(f)
    
    # Extract categories for the specified level
    level_categories = [
        cat for cat in taxonomy_data.get('categories', []) 
        if cat.get('level') == level
    ]
    
    choices = []
    for category in level_categories:
        # Use category_template for all levels since we only need name and description
        choices.append(category_template(category))
    
    return choices

def record_to_sample(record: dict, choices: list[str]) -> Sample:
    return Sample(
        input=grant_template(record),
        choices=choices,
        metadata={"grant_id": record.get("id", ""), "title": record.get("title", "")}
    )


@hooks(name="ClassificationOutputHook", description="Hook to save classification results as JSON files")
class ClassificationOutputHook(Hooks):
    
    def __init__(self):
        self.classification_results = []
        
    async def on_sample_end(self, data: SampleEnd) -> None:
        """Collect classification results for each sample."""
        # Parse the model's answers to get selected choice letters
        selected_answers = parse_answers(data.sample, multiple_correct=True)
        # Convert answer letters to actual category names
        selected_categories = []
        for answer_letter in selected_answers:
            choice_index = answer_index(answer_letter)
            if choice_index < len(data.sample.choices):
                choice_text = data.sample.choices[choice_index]
                # Parse the category name from the formatted text (e.g., "**Name**: Category Name")
                if "**Name**: " in choice_text:
                    category_name = choice_text.split("**Name**: ")[1].split("\n")[0]
                    selected_categories.append(category_name)
        
        sample_result = {
            "grant_id": data.sample.metadata.get("grant_id", "") if data.sample.metadata else "",
            "title": data.sample.metadata.get("title", "") if data.sample.metadata else "",
            "taxonomy_level": data.sample.metadata["taxonomy_level"],
            "input": data.sample.input,
            "selected_categories": selected_categories
        }

        self.classification_results.append(sample_result)
        
    async def on_task_end(self, data: TaskEnd) -> None:
        """Save aggregated classification results to JSON file."""
        
        # Save to classification.json
        with open(CLASSIFICATION_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.classification_results, f, ensure_ascii=False)
        

        # subcategory_count = sum(len(result.get('selected_subcategories', [])) for result in self.classification_results)



    
@task
def classify(taxonomy_level: str = 'base') -> Task:
    """
    Classify grants using comprehensive taxonomy with specified granularity level.
    
    Args:
        taxonomy_level: Level of granularity ('coarsened', 'base', 'refined')
    
    Returns:
        Task for classifying grants into categories at the specified level.
    """
    # Load choices from comprehensive taxonomy
    choices = load_comprehensive_taxonomy_choices(taxonomy_level)
    
    # Create record_to_sample function with the loaded choices
    def record_to_sample_with_choices(record: dict) -> Sample:
        return Sample(
            input=grant_template(record),
            choices=choices,
            metadata={
                "grant_id": record.get("id", ""), 
                "title": record.get("title", ""),
                "taxonomy_level": taxonomy_level
            }
        )

    return Task(
        dataset=json_dataset(str(GRANTS_FILE), record_to_sample_with_choices),
        solver=[
            multiple_choice(multiple_correct=True, cot=False),
        ],
        hooks=["ClassificationOutputHook"]
    )
