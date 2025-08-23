from inspect_ai import Task, task
from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks
from inspect_ai.dataset import json_dataset, FieldSpec, Sample, MemoryDataset
from config import PROMPTS_DIR, CATEGORY_PATH, FOR_CODES_CLEANED_PATH, NUM_KEYWORDS_PER_CATEGORY, BATCH_SIZE, KEYWORDS_TYPE, CATEGORY_DIR
from config import KEYWORDS_PATH
import time
import math
from inspect_ai.solver import system_message, generate, user_message, TaskState, use_tools, chain_of_thought
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.util import json_schema
from inspect_ai.scorer import scorer, Score, Target, metric, Metric, SampleScore, INCORRECT, NOANSWER

import json 
from metric import total
from pydantic import BaseModel, Field
import random
from pathlib import Path


@scorer(metrics=[total()])
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


@scorer(metrics=[total()])
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
            
            # Parse keywords from the new format: **term** (type): description
            input_keywords = set()
            for line in keywords_part.split('\n'):
                if line.startswith('**') and '**' in line[2:]:
                    # Extract term between ** markers
                    end_marker = line.find('**', 2)
                    if end_marker > 2:
                        term = line[2:end_marker].strip()
                        input_keywords.add(term)
            
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

def get_num_batches() -> int:
    """
    Calculate the number of batches based on total available keywords and batch size.
    
    Returns:
        int: Number of batches needed to process all keywords
    """
    keywords_file = KEYWORDS_PATH
    
    if not keywords_file.exists():
        # Default fallback if keywords file doesn't exist yet
        return 10
    
    try:
        with open(keywords_file, 'r', encoding='utf-8') as f:
            keywords_data = json.load(f)
        
        # Count total keywords based on the structure
        if isinstance(keywords_data, list):
            # New flattened structure - array of keyword objects
            total_keywords = len([kw for kw in keywords_data if isinstance(kw, dict) and 'term' in kw])
        else:
            # Fallback for old structure
            total_keywords = len(keywords_data.get('keywords', []))
        
        # Calculate number of batches needed
        num_batches = max(1, math.ceil(total_keywords / BATCH_SIZE))
        return min(num_batches, 50)  # Cap at 50 batches for practical reasons
        
    except (json.JSONDecodeError, FileNotFoundError, KeyError):
        # Default fallback if file cannot be read
        return 10


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

def load_dataset(batch_size: int, random_seed: int, keywords_type: str, num_batches: int = 1):
    """
    Load and sample keywords from the extracted keywords dataset.
    Include FOR codes context for categorization.
    
    Args:
        batch_size: Number of unique keywords to sample for categorization
        random_seed: Random seed for reproducible sampling
        keywords_type: Type of keywords to use
        num_batches: Number of batches to generate
    """
    
    # Choose input file based on whether to use harmonized terms
    
    keywords_file = KEYWORDS_PATH
    
    if not keywords_file.exists():
        raise FileNotFoundError(f"Keywords file not found at {keywords_file}. Please run the 'extract' task first.")
    
    with open(keywords_file, 'r', encoding='utf-8') as f:
        keywords_data = json.load(f)

    # Handle the new flattened structure (top-level array of keywords)
    all_keywords = []  # Changed from set to list to preserve keyword objects
    

    if keywords_type is None:
        # Include all keywords regardless of type
        for keyword in keywords_data:
            if isinstance(keyword, dict) and 'term' in keyword:
                all_keywords.append(keyword)
    else:
        # Filter by specific type (use the type directly as specified in input file)
        for keyword in keywords_data:
            if isinstance(keyword, dict) and 'term' in keyword and keyword.get('type') == keywords_type:
                all_keywords.append(keyword)


    # Remove duplicates based on term (preserve first occurrence)
    seen_terms = set()
    unique_keywords = []
    for kw in all_keywords:
        if kw['term'] not in seen_terms:
            seen_terms.add(kw['term'])
            unique_keywords.append(kw)

    # Convert to sorted list for consistent ordering
    sorted_keywords = sorted(unique_keywords, key=lambda x: x['term'])
    
    # Prepare multiple samples
    samples: list[Sample] = []

    # Load FOR codes for context once
    for_divisions = load_for_codes()

    for i in range(num_batches):
        # For each batch, generate a reproducible sample of keywords
        batch_seed = random_seed + i
        
        if len(sorted_keywords) > batch_size:
            random.seed(batch_seed)
            sampled_keywords = random.sample(sorted_keywords, batch_size)
            # Shuffle the sampled keywords to randomize their order
            random.shuffle(sampled_keywords)
        else:
            sampled_keywords = list(sorted_keywords)
            # Shuffle all keywords if using the full set
            random.seed(batch_seed)
            random.shuffle(sampled_keywords)

        # Create input that includes keywords with descriptions and FOR code context
        keywords_entries = []
        for kw in sampled_keywords:
            entry = f"**{kw['term']}** ({kw.get('type', 'general')}): {kw.get('description', 'No description available')}"
            keywords_entries.append(entry)
        
        keywords_text = "\n".join(keywords_entries)
        for_context = "\n\nFOR Division Context:\n" + "\n".join([
            f"{code}: {data['name']}" for code, data in sorted(for_divisions.items(), key=lambda x: int(x[0]))
        ])

        full_input = keywords_text + for_context

        samples.append(Sample(id=f"categorise_batch{i}", input=full_input))

    return MemoryDataset(samples)


@task
def categorise(num_keywords_per_category: int = NUM_KEYWORDS_PER_CATEGORY, batch_size: int = BATCH_SIZE, keywords_type: str = KEYWORDS_TYPE, random_seed: int = 42, num_batches: int = None):
    """
    Categorise sampled keywords into research categories with comprehensive scoring.
    
    This task samples a subset of extracted keywords to ensure manageable
    token count and focused categorization results. It uses two scorers:
    1. categories_discrepancy: Measures how close the number of generated categories is to the target
    2. keywords_coverage: Measures how well the categories cover the input keywords
    
    Args:
        num_keywords_per_category: Target number of keywords per category (default: 10)
        batch_size: Number of unique keywords to sample for categorization (default: 1000)
        random_seed: Random seed for reproducible sampling (default: 42)
        keywords_type: Type of keywords to use (default: "keywords")
        num_batches: Number of batches to generate (default: auto-calculated from total keywords)
    """
    # Auto-calculate num_batches if not provided
    if num_batches is None:
        num_batches = get_num_batches()
    
    # Calculate target number of categories based on batch size and keywords per category
    num_target_categories = max(1, batch_size // num_keywords_per_category)
    
    # Calculate bounds (±10%)
    lower_bound = int(num_target_categories * 0.9)
    upper_bound = int(num_target_categories * 1.1)
    
    # Create dataset with metadata
    dataset = load_dataset(batch_size=batch_size, random_seed=random_seed, 
                          keywords_type=keywords_type, num_batches=num_batches)
    
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

