from inspect_ai import Task, task
from inspect_ai.hooks import Hooks, TaskEnd, hooks
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import system_message, generate
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.util import json_schema
from pydantic import BaseModel, Field
import json
from pathlib import Path
from config import CATEGORY_DIR, CATEGORY_PATH, PROMPTS_DIR


class Category(BaseModel):
    """A flexible research category linked to FOR codes."""
    model_config = {"extra": "forbid"}
    name: str = Field(description="Name of the category")
    description: str = Field(description="A few sentences describing what this category is about, including its scope, focus areas, and the types of research or technologies it encompasses")
    keywords: list[str] = Field(description="List of keywords associated with this category")
    for_code: str = Field(description="The 2-digit FOR division code this category falls under (e.g., '30', '46', '51')")
    for_division_name: str = Field(description="The name of the FOR division this category belongs to")

class MergedCategoryList(BaseModel):
    """A list of merged research categories."""
    model_config = {"extra": "forbid"}
    categories: list[Category] = Field(description="List of merged research categories with duplicates combined")


@hooks(name="MergeCategoriesHook", description="Hook to combine FOR group merge results into a single categories.json")
class MergeCategoriesHook(Hooks):

    async def on_task_end(self, data: TaskEnd) -> None:
        """Combine all FOR group merge results into a single categories.json file.

        Behavior:
        - Collects category outputs from all samples (one per FOR group)
        - Combines them into a single unified taxonomy
        - Writes the merged object {"categories": [...]} to CATEGORY_PATH
        """
        all_merged_categories = []
        
        # Collect categories from all sample outputs
        for sample in data.dataset.samples:
            if sample.output and sample.output.completion:
                try:
                    result = json.loads(sample.output.completion)
                    if isinstance(result, dict) and "categories" in result:
                        categories = result["categories"]
                        if isinstance(categories, list):
                            all_merged_categories.extend(categories)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Failed to parse output from sample {sample.id}: {e}")
                    continue
        
        # Write final merged output
        merged_obj = {"categories": all_merged_categories}
        CATEGORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CATEGORY_PATH, 'w', encoding='utf-8') as f:
            json.dump(merged_obj, f, indent=2, ensure_ascii=False)


def load_category_files_dataset() -> MemoryDataset:
    """Load and group category files by FOR code into separate dataset samples.
    
    Returns:
        MemoryDataset: Dataset containing one sample per FOR group with categories to merge
    """
    CATEGORY_DIR.mkdir(parents=True, exist_ok=True)
    
    # Group categories by FOR code
    for_groups = {}
    total_files = 0
    
    # Load and group all categories by FOR code
    for category_file in sorted(CATEGORY_DIR.glob("*.json")):
        try:
            with open(category_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            # Extract categories from the file
            categories_in_file = []
            if isinstance(content, dict) and "categories" in content:
                categories_in_file = content["categories"]
            elif isinstance(content, list):
                categories_in_file = content
            elif isinstance(content, dict):
                categories_in_file = [content]
            
            # Group categories by their FOR code
            for category in categories_in_file:
                if isinstance(category, dict) and "for_code" in category:
                    for_code = category["for_code"]
                    if for_code not in for_groups:
                        for_groups[for_code] = []
                    for_groups[for_code].append(category)
            
            total_files += 1
            
        except Exception as e:
            print(f"Warning: Failed to load {category_file}: {e}")
            continue
    
    # Create samples for each FOR group
    samples = []
    for for_code, categories in sorted(for_groups.items()):
        # Create the categories structure for this FOR group
        for_content = {"categories": categories}
        
        # Create a sample for this FOR group
        sample = Sample(
            id=f"merge_for_{for_code}",
            input=json.dumps(for_content, indent=2, ensure_ascii=False),
            metadata={
                "for_code": for_code,
                "total_categories": len(categories),
                "source_files": total_files,
                "source_dir": str(CATEGORY_DIR)
            }
        )
        samples.append(sample)
    
    # If no categories found, create an empty sample
    if not samples:
        samples = [Sample(
            id="no_categories", 
            input='{"categories": []}',
            metadata={"total_categories": 0, "source_files": 0, "source_dir": str(CATEGORY_DIR)}
        )]
    
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
                json_schema=json_schema(MergedCategoryList),
                strict=True
            )
        ),
        hooks=["MergeCategoriesHook"]
    )
