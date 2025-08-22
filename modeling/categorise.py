from inspect_ai import Task, task
from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks
from inspect_ai.dataset import json_dataset, FieldSpec, Sample, MemoryDataset
from config import KEYWORDS_FINAL_PATH, PROMPTS_DIR, CATEGORY_PATH, FOR_CODES_CLEANED_PATH, NUM_KEYWORDS_PER_CATEGORY, SAMPLE_SIZE, KEYWORDS_TYPE, CATEGORY_DIR, NUM_SAMPLES
import time
from inspect_ai.solver import system_message, generate, user_message, TaskState, use_tools, chain_of_thought
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.util import json_schema
from inspect_ai.scorer import scorer, Score, Target, metric, Metric, SampleScore, INCORRECT, NOANSWER

import json 
from metric import count
from pydantic import BaseModel, Field
import random
from pathlib import Path


@scorer(metrics=[count()])
def categories_discrepancy():
    """
    Scorer that computes the discrepancy between expected and actual number of categories.
    
    This provides metrics for categorization accuracy by measuring how close
    the generated category count is to the target number.
    
    Returns:
        Score with negative discrepancy as the metric (higher is better, 0 is perfect)
    """
    
    async def score(state: TaskState, target: Target) -> Score:
        """Score based on category count discrepancy from target."""
        
        if not state.output or not state.output.completion:
            return Score(
                value=NOANSWER, 
                explanation="No categorisation output to score"
            )
        
        try:
            # Expect strictly the CategoryList shape: an object with a 'categories' key
            result = json.loads(state.output.completion)
            if not isinstance(result, dict) or "categories" not in result or not isinstance(result["categories"], list):
                return Score(
                    value=NOANSWER,
                    explanation="Expected an object with a top-level 'categories' key containing a list"
                )

            categories_list = result["categories"]
            
            target_categories = state.metadata['num_target_categories']
            
            # Count categories and keywords
            actual_categories = len(categories_list)
            total_keywords = 0
            for_code_assignments = {}
            missing_for_codes = 0

            for category in categories_list:
                if "keywords" in category and isinstance(category["keywords"], list):
                    total_keywords += len(category["keywords"])

                for_code = category.get("for_code")
                if for_code:
                    for_code_assignments[for_code] = for_code_assignments.get(for_code, 0) + 1
                else:
                    missing_for_codes += 1

            # Calculate discrepancy (negative so higher scores are better)
            discrepancy = abs(actual_categories - target_categories)
            score_value = -discrepancy
            
            # Create explanation text
            for_distribution = ", ".join([
                f"FOR {code}: {count} categories" for code, count in sorted(for_code_assignments.items())
            ])

            explanation_parts = [
                f"Generated {actual_categories} categories (target: {target_categories}, discrepancy: {discrepancy})",
                f"Total keywords: {total_keywords}"
            ]
            
            if for_distribution:
                explanation_parts.append(f"FOR code distribution: {for_distribution}")
            if missing_for_codes > 0:
                explanation_parts.append(f"WARNING: {missing_for_codes} categories missing FOR codes")

            explanation = ". ".join(explanation_parts)

            return Score(value=score_value, explanation=explanation)
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return Score(
                value=NOANSWER, 
                explanation=f"Error parsing categorisation result: {str(e)}"
            )
    
    return score


@scorer(metrics=[count()])
def keywords_coverage():
    """
    Scorer that computes the discrepancy between input keywords and keywords covered by categories.
    
    This measures how well the categorization covers the input keywords by comparing
    the number of unique keywords in the input with those assigned to categories.
    
    Returns:
        Score with negative coverage discrepancy (higher is better, 0 means perfect coverage)
    """
    
    async def score(state: TaskState, target: Target) -> Score:
        """Score based on keyword coverage discrepancy."""
        
        if not state.output or not state.output.completion:
            return Score(
                value=NOANSWER, 
                explanation="No categorisation output to score"
            )
        
        try:
            # Parse the categorization result
            result = json.loads(state.output.completion)
            if not isinstance(result, dict) or "categories" not in result or not isinstance(result["categories"], list):
                return Score(
                    value=NOANSWER,
                    explanation="Expected an object with a top-level 'categories' key containing a list"
                )

            categories_list = result["categories"]
            
            # Extract all keywords from the input
            input_text = state.input
            # The input contains keywords followed by FOR context, separated by "\n\nFOR Division Context:"
            keywords_part = input_text.split("\n\nFOR Division Context:")[0]
            input_keywords = set(keyword.strip() for keyword in keywords_part.split(","))
            
            # Extract all keywords from the categorization result
            covered_keywords = set()
            for category in categories_list:
                if "keywords" in category and isinstance(category["keywords"], list):
                    covered_keywords.update(keyword.strip() for keyword in category["keywords"])
            
            # Calculate coverage metrics
            total_input_keywords = len(input_keywords)
            total_covered_keywords = len(covered_keywords)
            
            # Find intersection (keywords that are both in input and covered)
            correctly_covered = input_keywords.intersection(covered_keywords)
            
            # Find keywords that are covered but not in input (potential hallucinations)
            extra_keywords = covered_keywords - input_keywords
            
            # Find keywords that are in input but not covered (missed keywords)
            missed_keywords = input_keywords - covered_keywords
            
            # Calculate coverage percentage
            coverage_percentage = len(correctly_covered) / total_input_keywords * 100 if total_input_keywords > 0 else 0
            
            # Score based on coverage (negative missed keywords so higher is better)
            score_value = -len(missed_keywords)
            
            explanation_parts = [
                f"Input keywords: {total_input_keywords}",
                f"Covered keywords: {total_covered_keywords}",
                f"Correctly covered: {len(correctly_covered)} ({coverage_percentage:.1f}%)",
                f"Missed keywords: {len(missed_keywords)} ({missed_keywords})",
                f"Extra keywords: {len(extra_keywords)}"
            ]
            
            explanation = ". ".join(explanation_parts)

            return Score(value=score_value, explanation=explanation)
                
        except (json.JSONDecodeError, KeyError, TypeError, IndexError) as e:
            return Score(
                value=NOANSWER, 
                explanation=f"Error analyzing keyword coverage: {str(e)}"
            )
    
    return score

class Category(BaseModel):
    """A flexible research category linked to FOR codes."""
    model_config = {"extra": "forbid"}
    name: str = Field(description="Name of the category")
    description: str = Field(description="A few sentences describing what this category is about, including its scope, focus areas, and the types of research or technologies it encompasses")
    keywords: list[str] = Field(description="List of keywords associated with this category")
    for_code: str = Field(description="The 2-digit FOR division code this category falls under (e.g., '30', '46', '51')")
    for_division_name: str = Field(description="The name of the FOR division this category belongs to")

class CategoryList(BaseModel):
    """A list of research categories."""
    model_config = {"extra": "forbid"}
    categories: list[Category] = Field(description="List of research categories")

@hooks(name="CategoriseOutputHook", description="Hook to save categorisation results as JSON files")
class CategoriseOutputHook(Hooks):

    async def on_sample_end(self, data: SampleEnd) -> None:
        """Save keywords extraction results."""

        output_text = data.sample.output.completion
        try:
            result_json = json.loads(output_text)
        except Exception:
            # If parsing fails, skip saving
            return

        # Ensure the category output directory exists
        CATEGORY_DIR.mkdir(parents=True, exist_ok=True)

        # Derive filename from sample id (fall back to timestamp if missing)
        sample_id = getattr(data.sample, 'id', None) or f"sample_{int(time.time())}"
        out_file = CATEGORY_DIR / f"{sample_id}.json"

        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, indent=2, ensure_ascii=False)

def load_for_codes():
    """Load FOR codes and return top-level divisions."""
    if not FOR_CODES_CLEANED_PATH.exists():
        raise FileNotFoundError(f"FOR codes file not found at {FOR_CODES_CLEANED_PATH}")
    
    with open(FOR_CODES_CLEANED_PATH, 'r', encoding='utf-8') as f:
        for_codes = json.load(f)
    
    # Extract top-level divisions (2-digit codes)
    top_level_divisions = {}
    for code, data in for_codes.items():
        if len(code) == 2:  # Top-level divisions are 2-digit codes
            top_level_divisions[code] = {
                'code': code,
                'name': data['name'],
                'description': data.get('definition', {}).get('description', '')
            }
    
    return top_level_divisions

def load_dataset(sample_size: int, random_seed: int, keywords_type: str, num_samples: int = 1):
    """
    Load and sample keywords from the extracted keywords dataset.
    Include FOR codes context for categorization.
    
    Args:
        sample_size: Number of unique keywords to sample for categorization
        random_seed: Random seed for reproducible sampling
        keywords_type: Type of keywords to use
        num_samples: Number of samples to generate
    """
    
    # Choose input file based on whether to use harmonized terms

    keywords_file = KEYWORDS_FINAL_PATH
    
    if not keywords_file.exists():
        raise FileNotFoundError(f"Keywords file not found at {keywords_file}. Please run the 'extract' task first.")
    
    with open(keywords_file, 'r', encoding='utf-8') as f:
        records = json.load(f)

    # Collect all unique keywords across all categories
    all_keywords = set()

    for record in records:
        if keywords_type is None:
            for category in ['keywords', 'methodology_keywords', 'application_keywords', 'technology_keywords']:
                if category in record and isinstance(record[category], list):
                    all_keywords.update(record[category])
        else:
            if keywords_type in record and isinstance(record[keywords_type], list):
                all_keywords.update(record[keywords_type])

    # Convert to sorted list for consistent ordering
    sorted_keywords = sorted(list(all_keywords))
    
    # Prepare multiple samples
    samples: list[Sample] = []

    # Load FOR codes for context once
    for_divisions = load_for_codes()

    for i in range(num_samples):
        # For each sample, generate a reproducible sample of keywords
        sample_seed = random_seed + i
        
        if len(sorted_keywords) > sample_size:
            random.seed(sample_seed)
            sampled_keywords = random.sample(sorted_keywords, sample_size)
            # Shuffle the sampled keywords to randomize their order
            random.shuffle(sampled_keywords)
        else:
            sampled_keywords = list(sorted_keywords)
            # Shuffle all keywords if using the full set
            random.seed(sample_seed)
            random.shuffle(sampled_keywords)

        # Create input that includes both keywords and FOR code context
        keywords_text = ", ".join(sampled_keywords)
        for_context = "\n\nFOR Division Context:\n" + "\n".join([
            f"{code}: {data['name']}" for code, data in sorted(for_divisions.items(), key=lambda x: int(x[0]))
        ])

        full_input = keywords_text + for_context

        samples.append(Sample(id=f"categorise_sample{i}", input=full_input))

    return MemoryDataset(samples)


@task
def categorise(num_keywords_per_category: int = NUM_KEYWORDS_PER_CATEGORY, sample_size: int = SAMPLE_SIZE, keywords_type: str = KEYWORDS_TYPE, random_seed: int = 42, num_samples: int = NUM_SAMPLES):
    """
    Categorise sampled keywords into research categories with comprehensive scoring.
    
    This task samples a subset of extracted keywords to ensure manageable
    token count and focused categorization results. It uses two scorers:
    1. categories_discrepancy: Measures how close the number of generated categories is to the target
    2. keywords_coverage: Measures how well the categories cover the input keywords
    
    Args:
        num_keywords_per_category: Target number of keywords per category (default: 10)
        sample_size: Number of unique keywords to sample for categorization (default: 1000)
        random_seed: Random seed for reproducible sampling (default: 42)
        keywords_type: Type of keywords to use (default: "keywords")
        num_samples: Number of samples to generate (default: 10)
    """
    # Calculate target number of categories based on sample size and keywords per category
    num_target_categories = max(1, sample_size // num_keywords_per_category)
    
    # Calculate bounds (±10%)
    lower_bound = int(num_target_categories * 0.9)
    upper_bound = int(num_target_categories * 1.1)
    
    # Create dataset with metadata
    dataset = load_dataset(sample_size=sample_size, random_seed=random_seed, 
                          keywords_type=keywords_type, num_samples=num_samples)
    
    # Add target categories to sample metadata
    for sample in dataset.samples:
        sample.metadata = {"num_target_categories": num_target_categories}
    
    return Task(
        dataset=dataset,
        solver=[
            system_message(str(PROMPTS_DIR / "categorise.txt"), 
                          num_target_categories=num_target_categories,
                          lower_bound=lower_bound,
                          upper_bound=upper_bound),
            generate(),
        ],
        scorer=[
            categories_discrepancy(),
            keywords_coverage()
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="Categories",
                json_schema=json_schema(CategoryList),
                strict=True
            ),
        ),
        hooks=["CategoriseOutputHook"]
    )

