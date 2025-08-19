

from inspect_ai import Task, task
from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks
from inspect_ai.dataset import json_dataset, FieldSpec, Sample, MemoryDataset
from config import LOGS_DIR, KEYWORDS_PATH, PROMPTS_DIR, CATEGORY_PATH
from inspect_ai.solver import system_message, generate, user_message, TaskState
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.util import json_schema
from inspect_ai.scorer import scorer, Score, Target, metric, Metric, SampleScore
import json 
from metric import count
from pydantic import BaseModel, Field
import random


@scorer(metrics=[count()])
def categories_counter():
    """
    Scorer that counts the total number of categories generated in the categorise task.
    
    This provides metrics for categorization volume and effectiveness,
    counting the total number of research categories created.
    
    Returns:
        Score with total category count as the metric
    """
    
    async def score(state: TaskState, target: Target) -> Score:
        """Score based on total generated category count."""
        
        if not state.output or not state.output.completion:
            return Score(
                value="incorrect", 
                explanation="No categorisation output to score"
            )
        
        try:
            # Parse the categorisation result - expecting an array of categories
            result = json.loads(state.output.completion)
            
            # Count categories - result should be an array
            if isinstance(result, list):
                total_categories = len(result)
                
                # Count total keywords across all categories
                total_keywords = 0
                for category in result:
                    if "keywords" in category and isinstance(category["keywords"], list):
                        total_keywords += len(category["keywords"])
                
                if total_categories > 0:
                    explanation = f"Generated {total_categories} categories with {total_keywords} total associated keywords"
                else:
                    explanation = "No categories generated"
                
                return Score(
                    value=total_categories,
                    explanation=explanation
                )
            else:
                return Score(
                    value=0,
                    explanation="Expected array of categories but got different format"
                )
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return Score(
                value="incorrect", 
                explanation=f"Error parsing categorisation result: {str(e)}"
            )
    
    return score

class Category(BaseModel):
    """A flexible research category."""
    model_config = {"extra": "forbid"}
    name: str = Field(description="Name of the category")
    description: str = Field(description="A few sentences describing what this category is about, including its scope, focus areas, and the types of research or technologies it encompasses")
    keywords: list[str] = Field(description="List of keywords associated with this category")

@hooks(name="CategoriseOutputHook", description="Hook to save categorisation results as JSON files")
class CategoriseOutputHook(Hooks):

    async def on_sample_end(self, data: SampleEnd) -> None:
        """Save keywords extraction results."""

        output_text = data.sample.output.completion
        result_json = json.loads(output_text)
        
        with open(CATEGORY_PATH, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, indent=2, ensure_ascii=False)

def load_dataset(sample_size: int = 10000, random_seed: int = 42, keywords_type="keywords"):
    """
    Load and sample keywords from the extracted keywords dataset.
    
    Args:
        sample_size: Number of unique keywords to sample for categorization
        random_seed: Random seed for reproducible sampling
    """
    
    if not KEYWORDS_PATH.exists():
        raise FileNotFoundError(f"Keywords file not found at {KEYWORDS_PATH}. Please run the 'extract' task first.")
    
    with open(KEYWORDS_PATH, 'r', encoding='utf-8') as f:
        records = json.load(f)

    # Collect all unique keywords across all categories
    all_keywords = set()
    # if keywords_type is None:
    #     for record in records:
    #         for category in ['keywords', 'methodology_keywords', 'application_keywords', 'technology_keywords']:
    #             if category in record and isinstance(record[category], list):
    #                 all_keywords.update(record[category])

    for record in records:
        if keywords_type is None:
            for category in ['keywords', 'methodology_keywords', 'application_keywords', 'technology_keywords']:
                if category in record and isinstance(record[category], list):
                    all_keywords.update(record[category])
        else:
            all_keywords.update(record[keywords_type])

    # Convert to sorted list for consistent ordering
    sorted_keywords = sorted(list(all_keywords))
    
    # Sample keywords if we have more than requested
    if len(sorted_keywords) > sample_size:
        random.seed(random_seed)
        sampled_keywords = random.sample(sorted_keywords, sample_size)
        # Shuffle the sampled keywords to randomize their order
        random.shuffle(sampled_keywords)
    else:
        sampled_keywords = sorted_keywords
        # Shuffle all keywords if using the full set
        random.seed(random_seed)
        random.shuffle(sampled_keywords)
    
    return MemoryDataset([Sample(
        id="categorise",
        input=", ".join(sampled_keywords),
    )])


@task
def categorise(sample_size: int = 5000, random_seed: int = 42, target_categories: int = 50, keywords_type="keywords"):
    """
    Categorise sampled keywords into research categories.
    
    This task samples a subset of extracted keywords to ensure manageable
    token count and focused categorization results.
    
    Args:
        sample_size: Number of unique keywords to sample for categorization (default: 5000)
        random_seed: Random seed for reproducible sampling (default: 42)
        target_categories: Target number of categories to generate (default: 50)
        keywords_type: Type of keywords to use (default: "keywords")
    """
    # Calculate bounds (±10%)
    lower_bound = int(target_categories * 0.9)
    upper_bound = int(target_categories * 1.1)
    
    return Task(
        dataset=load_dataset(sample_size=sample_size, random_seed=random_seed, keywords_type=keywords_type),
        solver=[
            system_message(str(PROMPTS_DIR / "categorise.txt"), 
                          target_categories=target_categories,
                          lower_bound=lower_bound,
                          upper_bound=upper_bound),
            generate(),
        ],
        scorer=[
            categories_counter()
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="Categories",
                json_schema=json_schema(list[Category]),
                strict=True
            )
        ),
        hooks=["CategoriseOutputHook"]
    )

