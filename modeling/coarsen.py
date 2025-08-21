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
            # Parse the coarsening result - expecting an array of coarsened categories
            result = json.loads(state.output.completion)
            
            # Count coarsened categories - result should be an array
            if isinstance(result, list):
                total_coarsened_categories = len(result)
                
                # Count total subcategories across all coarsened categories
                total_subcategories = 0
                for category in result:
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
            else:
                return Score(
                    value=0,
                    explanation="Expected array of coarsened categories but got different format"
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
        categories = json.load(f)  # Now expecting an array directly
    
    # Get the number of base categories for ratio calculations
    base_categories_count = len(categories['categories'])
    
    # Create input text with all categories
    input_text = "Here are the detailed research categories to be coarsened into higher-level abstractions:\n\n"
    
    for i, category in enumerate(categories['categories'], 1):
        input_text += f"{i}. **{category['name']}**: {category['description']}\n\n"

    return MemoryDataset([Sample(
        id="corsen",
        input=input_text,
        metadata={"base_categories_count": base_categories_count}
    )])

@task
def coarsen(coarsening_ratio: float = 0.1):
    """
    Task to coarsen detailed categories into higher-level, more abstract categories.
    
    This task takes the output from the 'categorise' task and produces a more strategic,
    high-level view of research domains by consolidating related categories into
    broader, more abstract themes suitable for strategic planning and overview purposes.
    
    Args:
        coarsening_ratio: Ratio of coarsened categories to base categories.
                         Examples:
                         - 0.05: Every ~20 base categories are grouped into 1 coarsened category
                         - 0.1: Every ~10 base categories are grouped into 1 coarsened category  
                         - 0.2: Every ~5 base categories are grouped into 1 coarsened category
                         
                         With 275 base categories:
                         - ratio 0.05 → ~14 coarsened categories (strategic overview)
                         - ratio 0.1 → ~28 coarsened categories (high-level themes)
                         - ratio 0.2 → ~55 coarsened categories (broad domains)
    """
    # Load dataset to get base categories count
    dataset = load_dataset()
    sample = next(iter(dataset))
    base_categories_count = sample.metadata["base_categories_count"]
    
    # Calculate target coarsened categories based on ratio
    target_coarsened_categories = max(1, int(base_categories_count * coarsening_ratio))
    
    # Calculate bounds (±10%)
    lower_bound = max(1, int(target_coarsened_categories * 0.9))
    upper_bound = int(target_coarsened_categories * 1.1)
    
    return Task(
        dataset=dataset,
        solver=[
            system_message(str(PROMPTS_DIR / "coarsen.txt"), 
                          target_coarsened_categories=target_coarsened_categories,
                          lower_bound=lower_bound,
                          upper_bound=upper_bound,
                          coarsening_ratio=coarsening_ratio,
                          base_categories_count=base_categories_count),
            generate(),
        ],
        scorer=[
            coarsened_categories_counter()
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="CoarsenedCategories",
                json_schema=json_schema(list[CoarsenedCategory]),
                strict=True
            )
        ),
        hooks=["CorsenOutputHook"]
    )
