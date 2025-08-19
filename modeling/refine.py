from inspect_ai import Task, task
from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks
from inspect_ai.dataset import json_dataset, FieldSpec, Sample, MemoryDataset
from config import LOGS_DIR, KEYWORDS_PATH, PROMPTS_DIR, CATEGORY_PATH, REFINED_CATEGORY_PATH
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
    counting the total number of high-level abstract categories created.
    
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
            # Parse the refinement result - expecting an array of refined categories
            result = json.loads(state.output.completion)
            
            # Count refined categories - result should be an array
            if isinstance(result, list):
                total_refined_categories = len(result)
                
                # Count total subcategories across all refined categories
                total_subcategories = 0
                for category in result:
                    if "subcategories" in category and isinstance(category["subcategories"], list):
                        total_subcategories += len(category["subcategories"])
                
                if total_refined_categories > 0:
                    explanation = f"Generated {total_refined_categories} refined categories encompassing {total_subcategories} subcategories"
                else:
                    explanation = "No refined categories generated"
                
                return Score(
                    value=total_refined_categories,
                    explanation=explanation
                )
            else:
                return Score(
                    value=0,
                    explanation="Expected array of refined categories but got different format"
                )
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return Score(
                value="incorrect", 
                explanation=f"Error parsing refinement result: {str(e)}"
            )
    
    return score

class RefinedCategory(BaseModel):
    """A high-level, abstract research category."""
    model_config = {"extra": "forbid"}
    name: str = Field(description="Name of the refined, high-level category")
    description: str = Field(description="A comprehensive description of this abstract category, explaining its broad scope, overarching themes, and how it encompasses multiple specific research areas")
    subcategories: list[str] = Field(description="List of more specific category names that fall under this high-level category")

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
    
    # Create input text with all categories
    input_text = "Here are the detailed research categories to be refined into higher-level abstractions:\n\n"
    
    for i, category in enumerate(categories, 1):
        input_text += f"{i}. **{category['name']}**: {category['description']}\n\n"
    
    return MemoryDataset([Sample(
        id="refine",
        input=input_text,
    )])

@task
def refine(target_refined_categories: int = 20):
    """
    Task to refine detailed categories into higher-level, more abstract categories.
    
    This task takes the output from the 'categorise' task and produces a more strategic,
    high-level view of research domains by consolidating related categories into
    broader, more abstract themes suitable for strategic planning and overview purposes.
    
    Args:
        target_refined_categories: Target number of high-level refined categories to generate (default: 20)
    """
    # Calculate bounds (±10%)
    lower_bound = int(target_refined_categories * 0.9)
    upper_bound = int(target_refined_categories * 1.1)
    
    return Task(
        dataset=load_dataset(),
        solver=[
            system_message(str(PROMPTS_DIR / "refine.txt"), 
                          target_refined_categories=target_refined_categories,
                          lower_bound=lower_bound,
                          upper_bound=upper_bound),
            generate(),
        ],
        scorer=[
            refined_categories_counter()
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="RefinedCategories",
                json_schema=json_schema(list[RefinedCategory]),
                strict=True
            )
        ),
        hooks=["RefineOutputHook"]
    )
