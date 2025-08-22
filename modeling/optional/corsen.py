from inspect_ai import Task, task
from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks
from inspect_ai.dataset import json_dataset, FieldSpec, Sample, MemoryDataset
from config import PROMPTS_DIR, CATEGORY_PATH, COARSENED_CATEGORY_PATH
from inspect_ai.solver import system_message, generate, user_message, TaskState
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.util import json_schema
from inspect_ai.scorer import scorer, Score, Target, metric, Metric, SampleScore
import json 
from pathlib import Path
from metric import count
from pydantic import BaseModel, Field



@scorer(metrics=[count()])
def coarsened_categories_counter():
    """
    Scorer that counts the total number of coarsened categories generated in the corsen task.
    
    This provides metrics for coarsening effectiveness,
    counting the total number of high-level abstract categories created.
    
    Returns:
        Score with total coarsened category count as the metric
    """
    
    async def score(state: TaskState, target: Target) -> Score:
        """Score based on total generated coarsened category count."""
        
        if not state.output or not state.output.completion:
            return Score(
                value="incorrect", 
                explanation="No corsen output to score"
            )
        
        try:
            # Parse the coarsening result - expecting an object with 'categories' key
            result = json.loads(state.output.completion)
            
            # Check if result has the expected structure
            if not isinstance(result, dict) or "categories" not in result or not isinstance(result["categories"], list):
                return Score(
                    value=0,
                    explanation="Expected an object with a top-level 'categories' key containing a list"
                )
            
            categories_list = result["categories"]
            total_coarsened_categories = len(categories_list)
            
            # Count total subcategories across all coarsened categories
            total_subcategories = 0
            for category in categories_list:
                if "subcategories" in category and isinstance(category["subcategories"], list):
                    total_subcategories += len(category["subcategories"])
            
            if total_coarsened_categories > 0:
                explanation = f"Generated {total_coarsened_categories} coarsened categories encompassing {total_subcategories} subcategories"
            else:
                explanation = "No coarsened categories generated"
            
            return Score(
                value=total_coarsened_categories,
                explanation=explanation
            )
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return Score(
                value="incorrect", 
                explanation=f"Error parsing coarsening result: {str(e)}"
            )
    
    return score

class CoarsenedCategory(BaseModel):
    """A high-level, abstract research category."""
    model_config = {"extra": "forbid"}
    name: str = Field(description="Name of the coarsened, high-level category")
    description: str = Field(description="A comprehensive description of this abstract category, explaining its broad scope, overarching themes, and how it encompasses multiple specific research areas")
    subcategories: list[str] = Field(description="List of more specific category names that fall under this high-level category")

class CoarsenedCategoryList(BaseModel):
    """A list of coarsened research categories."""
    model_config = {"extra": "forbid"}
    categories: list[CoarsenedCategory] = Field(description="List of coarsened research categories")

@hooks(name="CorsenOutputHook", description="Hook to save coarsened categorisation results as JSON files")
class CorsenOutputHook(Hooks):

    async def on_sample_end(self, data: SampleEnd) -> None:
        """Save coarsened categories results."""

        output_text = data.sample.output.completion
        result_json = json.loads(output_text)
        
        # Save to coarsened_categories.json
        COARSENED_CATEGORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        with open(COARSENED_CATEGORY_PATH, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, indent=2, ensure_ascii=False)

def load_dataset():
    """Load the output from the categorise task."""
    
    with open(CATEGORY_PATH, 'r', encoding='utf-8') as f:
        categories_data = json.load(f)  # Expecting an object with 'categories' key
    
    # Extract the categories list from the wrapper object
    categories = categories_data.get('categories', [])
    
    # Create input text with all categories
    input_text = "Here are the detailed research categories to be coarsened into higher-level abstractions:\n\n"
    
    for i, category in enumerate(categories, 1):
        input_text += f"{i}. **{category['name']}**: {category['description']}\n\n"
    
    return MemoryDataset([Sample(
        id="corsen",
        input=input_text,
    )])

@task
def corsen(target_coarsened_categories: int = 20):
    """
    Task to coarsen detailed categories into higher-level, more abstract categories.
    
    This task takes the output from the 'categorise' task and produces a more strategic,
    high-level view of research domains by consolidating related categories into
    broader, more abstract themes suitable for strategic planning and overview purposes.
    
    Args:
        target_coarsened_categories: Target number of high-level coarsened categories to generate (default: 20)
    """
    # Calculate bounds (±10%)
    lower_bound = int(target_coarsened_categories * 0.9)
    upper_bound = int(target_coarsened_categories * 1.1)
    
    return Task(
        dataset=load_dataset(),
        solver=[
            system_message(str(PROMPTS_DIR / "corsen.txt"), 
                          target_coarsened_categories=target_coarsened_categories,
                          lower_bound=lower_bound,
                          upper_bound=upper_bound),
            generate(),
        ],
        scorer=[
            coarsened_categories_counter()
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="CoarsenedCategories",
                json_schema=json_schema(CoarsenedCategoryList),
                strict=True
            )
        ),
        hooks=["CorsenOutputHook"]
    )
