from inspect_ai import Task, task
from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks
from inspect_ai.dataset import json_dataset, FieldSpec, Sample, MemoryDataset
from config import PROMPTS_DIR, CATEGORY_PATH, REFINED_CATEGORY_PATH
from inspect_ai.solver import system_message, generate, user_message, TaskState
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.util import json_schema
from inspect_ai.scorer import scorer, Score, Target, metric, Metric, SampleScore
import json 
from pathlib import Path
from metric import count
from pydantic import BaseModel, Field



@scorer(metrics=[count()])
def refined_categories_counter():
    """
    Scorer that counts the total number of refined categories generated in the refine task.
    
    This provides metrics for refinement effectiveness,
    counting the total number of detailed, specific categories created.
    
    Returns:
        Score with total refined category count as the metric
    """
    
    async def score(state: TaskState, target: Target) -> Score:
        """Score based on total generated refined category count."""
        
        if not state.output or not state.output.completion:
            return Score(
                value="incorrect", 
                explanation="No refine output to score"
            )
        
        try:
            # Parse the refinement result - expecting an object with 'categories' key
            result = json.loads(state.output.completion)
            
            # Check if result has the expected structure
            if not isinstance(result, dict) or "categories" not in result or not isinstance(result["categories"], list):
                return Score(
                    value=0,
                    explanation="Expected an object with a top-level 'categories' key containing a list"
                )
            
            categories_list = result["categories"]
            total_refined_categories = len(categories_list)
            
            # Count total parent categories referenced
            total_parent_categories = 0
            unique_parents = set()
            for category in categories_list:
                if "parent_category" in category and category["parent_category"]:
                    unique_parents.add(category["parent_category"])
            total_parent_categories = len(unique_parents)
            
            if total_refined_categories > 0:
                explanation = f"Generated {total_refined_categories} refined categories derived from {total_parent_categories} parent categories"
            else:
                explanation = "No refined categories generated"
            
            return Score(
                value=total_refined_categories,
                explanation=explanation
            )
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return Score(
                value="incorrect", 
                explanation=f"Error parsing refinement result: {str(e)}"
            )
    
    return score

class RefinedCategory(BaseModel):
    """A detailed, specific research category derived from a broader parent category."""
    model_config = {"extra": "forbid"}
    name: str = Field(description="Name of the refined, specific category")
    description: str = Field(description="A detailed description of this specific category, explaining its narrow focus, specialized applications, and how it represents a distinct sub-area within the broader parent category")
    parent_category: str = Field(description="Name of the parent category from which this refined category is derived")

class RefinedCategoryList(BaseModel):
    """A list of refined research categories."""
    model_config = {"extra": "forbid"}
    categories: list[RefinedCategory] = Field(description="List of refined research categories")

@hooks(name="RefineOutputHook", description="Hook to save refined categorisation results as JSON files")
class RefineOutputHook(Hooks):

    async def on_sample_end(self, data: SampleEnd) -> None:
        """Save refined categories results."""

        output_text = data.sample.output.completion
        result_json = json.loads(output_text)
        
        # Save to refined_categories.json
        REFINED_CATEGORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        with open(REFINED_CATEGORY_PATH, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, indent=2, ensure_ascii=False)

def load_dataset():
    """Load the output from the categorise task."""
    
    with open(CATEGORY_PATH, 'r', encoding='utf-8') as f:
        categories = json.load(f)  # Now expecting an array directly
    
    # Get the number of base categories for ratio calculations
    base_categories_count = len(categories['categories'])
    
    # Create input text with all categories
    input_text = "Here are the research categories to be refined into more detailed, specific subcategories:\n\n"
    # print(categories)
    for i, category in enumerate(categories['categories'], 1):
        input_text += f"{i}. **{category['name']}**: {category['description']}\n\n"
    
    return MemoryDataset([Sample(
        id="refine",
        input=input_text,
        metadata={"base_categories_count": base_categories_count}
    )])

@task
def refine(refinement_ratio: float = 0.5):
    """
    Task to refine broad categories into more detailed, specific subcategories.
    
    This task takes the output from the 'categorise' task and produces a more granular,
    detailed view of research domains by breaking down broad categories into
    more specific, focused research areas suitable for detailed analysis and specialization.
    
    Args:
        refinement_ratio: Ratio of refined categories to base categories. 
                         Examples:
                         - 0.5: Each base category produces 0.5 refined categories on average (fewer refined than base)
                         - 1.0: Each base category produces 1 refined category on average (equal numbers)
                         - 2.0: Each base category produces 2 refined categories on average (more refined than base)
                         
                         With 275 base categories:
                         - ratio 0.5 → ~138 refined categories
                         - ratio 1.0 → ~275 refined categories  
                         - ratio 2.0 → ~550 refined categories
    """
    # Load dataset to get base categories count
    dataset = load_dataset()
    sample = next(iter(dataset))
    base_categories_count = sample.metadata["base_categories_count"]
    
    # Calculate target refined categories based on ratio
    target_refined_categories = int(base_categories_count * refinement_ratio)
    
    # Calculate bounds (±15% for refinement to allow for natural variation in subcategory count)
    lower_bound = int(target_refined_categories * 0.85)
    upper_bound = int(target_refined_categories * 1.15)
    
    return Task(
        dataset=dataset,
        solver=[
            system_message(str(PROMPTS_DIR / "refine.txt"), 
                          target_refined_categories=target_refined_categories,
                          lower_bound=lower_bound,
                          upper_bound=upper_bound,
                          refinement_ratio=refinement_ratio,
                          base_categories_count=base_categories_count),
            generate(),
        ],
        scorer=[
            refined_categories_counter()
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="RefinedCategories",
                json_schema=json_schema(RefinedCategoryList),
                strict=True
            )
        ),
        hooks=["RefineOutputHook"]
    )
