
from inspect_ai._util.answer import answer_character, answer_index
from inspect_ai import Task, task
from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks
from inspect_ai.dataset import json_dataset, FieldSpec, Sample, MemoryDataset
from inspect_ai.solver._multiple_choice import parse_answers
from config import LOGS_DIR, KEYWORDS_PATH, PROMPTS_DIR, CATEGORY_PATH, REFINED_CATEGORY_PATH, GRANTS_FILE, RESULTS_DIR
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
        data = json.load(f)  # Now expecting an array directly
    
    # Check if these are refined categories (have subcategories field)
    has_subcategories = any('subcategories' in category for category in data)
    
    choices = []
    for category in data:
        if has_subcategories:
            choices.append(refined_category_template(category))
        else:
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
        # parse_answers(data)
        choices = data.sample.choices
        sample_result = {
            "grant_id": data.sample.metadata.get("grant_id", "") if data.sample.metadata else "",
            "title": data.sample.metadata.get("title", "") if data.sample.metadata else "",
            "input": data.sample.input,
            "selected_categories": [],
            "selected_subcategories": []
        }


        
        self.classification_results.append(sample_result)
        
    async def on_task_end(self, data: TaskEnd) -> None:
        """Save aggregated classification results to JSON file."""
        
        # Save to classification.json
        classification_path = RESULTS_DIR / "classification.json"
        with open(classification_path, 'w', encoding='utf-8') as f:
            json.dump(self.classification_results, f, indent=2, ensure_ascii=False)
        

        # subcategory_count = sum(len(result.get('selected_subcategories', [])) for result in self.classification_results)



@task
def classify(category_path: str = CATEGORY_PATH) -> Task:
    """
    Classify grants into categories using multiple choice selection.
    
    Args:
        category_path: Path to category file. Defaults to refined categories.
    
    Returns:
        Task for classifying grants into categories.
    """
    
    # Load choices based on the selected category file
    choices = load_choices(Path(category_path))
    
    # Create record_to_sample function with the loaded choices
    def record_to_sample_with_choices(record: dict) -> Sample:
        return Sample(
            input=grant_template(record),
            choices=choices,
            metadata={"grant_id": record.get("id", ""), "title": record.get("title", "")}
        )

    return Task(
        dataset=json_dataset(str(GRANTS_FILE), record_to_sample_with_choices),
        solver=[
            multiple_choice(multiple_correct=False, cot=False),


        ],
        hooks=["ClassificationOutputHook"],
        scorer=[
            answer("letter")
        ]
    )
