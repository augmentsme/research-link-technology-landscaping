
from inspect_ai._util.answer import answer_character, answer_index
from inspect_ai import Task, task
from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks
from inspect_ai.dataset import json_dataset, FieldSpec, Sample, MemoryDataset
from inspect_ai.solver._multiple_choice import parse_answers
# from config import CATEGORY_PATH, GRANTS_FILE, CLASSIFICATION_PATH
from inspect_ai.solver import system_message, generate, user_message, multiple_choice
from inspect_ai.scorer import model_graded_fact, answer, choice
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.util import json_schema

import json 
from pathlib import Path

import config 

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
        selected_answers = parse_answers(data.sample, multiple_correct=True)
        selected_categories = []
        for answer_letter in selected_answers:
            choice_index = answer_index(answer_letter)
            if choice_index < len(data.sample.choices):
                choice_text = data.sample.choices[choice_index]
                if "**Name**: " in choice_text:
                    category_name = choice_text.split("**Name**: ")[1].split("\n")[0]
                    selected_categories.append(category_name)
        
        sample_result = {
            "grant_id": data.sample.metadata.get("grant_id", "") if data.sample.metadata else "",
            "title": data.sample.metadata.get("title", "") if data.sample.metadata else "",
            # "taxonomy_level": data.sample.metadata["taxonomy_level"],
            "input": data.sample.input,
            "selected_categories": selected_categories
        }

        self.classification_results.append(sample_result)
        
    async def on_task_end(self, data: TaskEnd) -> None:
        
        # Save to classification.json
        with open(config.CLASSIFICATION_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.classification_results, f, ensure_ascii=False)
        

        # subcategory_count = sum(len(result.get('selected_subcategories', [])) for result in self.classification_results)



    
@task
def classify() -> Task:
    """
    Classify grants using categories specified in `CATEGORY_PATH`.

    Returns:
        Task for classifying grants into categories loaded from `CATEGORY_PATH`.
    """
    # Load choices from CATEGORY_PATH
    choices = load_choices(config.Categories.categories_path)

    # Create record_to_sample function with the loaded choices
    def record_to_sample_with_choices(record: dict) -> Sample:
        return Sample(
            input=grant_template(record),
            choices=choices,
            metadata={
                "grant_id": record.get("id", ""),
                "title": record.get("title", "")
            }
        )

    return Task(
        dataset=json_dataset(str(config.Grants.grants_path), record_to_sample_with_choices),
        solver=[
            multiple_choice(multiple_correct=True, cot=False),
        ],
        hooks=["ClassificationOutputHook"]
    )
