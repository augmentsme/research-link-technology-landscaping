from inspect_ai import Task, task
from inspect_ai.hooks import Hooks, TaskEnd, hooks, SampleEnd
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import system_message, generate
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.util import json_schema
from pydantic import BaseModel, Field
import json
from pathlib import Path
from config import CATEGORY_PROPOSAL_PATH, CATEGORY_PATH, PROMPTS_DIR

from models import Category, CategoryList


@hooks(name="MergeCategoriesHook", description="Hook to combine FOR group merge results into a single categories.json")
class MergeCategoriesHook(Hooks):
    def __init__(self):
        self.results = []
    async def on_sample_end(self, data: SampleEnd):
        result: CategoryList = json.loads(data.sample.output.completion)
        self.results.extend(result['categories'])

    async def on_task_end(self, data: TaskEnd) -> None:

        with open(CATEGORY_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False)


def load_category_files_dataset() -> MemoryDataset:
    """Load and group category files by FOR code into separate dataset samples.
    
    Returns:
        MemoryDataset: Dataset containing one sample per FOR group with categories to merge
    """

    for_groups = {}
    with open(CATEGORY_PROPOSAL_PATH, 'r', encoding='utf-8') as f:
        content = json.load(f)


    categories = content["categories"]

    for category in categories:
        for_code_value = category["for_code"]
        for_groups[for_code_value].append(category)
    
    samples = []
    for for_code, categories in sorted(for_groups.items()):
        # Create the categories structure for this FOR group
        for_content = {"categories": categories}
        
        # Create a sample for this FOR group
        sample = Sample(
            id=f"merge_for_{for_code}",
            input=json.dumps(for_content, ensure_ascii=False),
            metadata={
                "for_code": for_code,
                "total_categories": len(categories),
            }
        )
        samples.append(sample)

    
    return MemoryDataset(samples)


@task
def merge() -> Task:
    """Task to merge categorisation outputs within each FOR group into a single `categories.json`.

    The task groups categories by FOR code and creates separate merge requests for each FOR group,
    ensuring that categories are only merged within the same research domain. The hook then 
    combines all FOR group results into a single final taxonomy file.
    """
    # Load dataset grouped by FOR codes
    dataset = load_category_files_dataset()

    return Task(
        dataset=dataset,
        solver=[
            system_message(str(PROMPTS_DIR / "merge.txt")),
            generate()
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="MergedCategories",
                json_schema=json_schema(CategoryList),
                strict=True
            )
        ),
        hooks=["MergeCategoriesHook"]
    )
