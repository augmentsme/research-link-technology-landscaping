"""
Keywords Extraction, Taxonomy Creation, and Grant Classification Tasks for Research Grants

This module implements Inspect AI tasks for:
1. Extracting relevant keywords from research grant titles and summaries
2. Creating and refining taxonomies through iterative improvement cycles
3. Classifying research grants into the created taxonomy
4. Loading and processing evaluation results
"""
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from inspect_ai import Task, task, hooks
from inspect_ai.dataset import Sample, json_dataset, FieldSpec
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.solver import system_message, generate, user_message
from inspect_ai.util import json_schema

# Import from modeling package
from modeling.solvers import simple_iterative_taxonomy_refinement
from modeling.scorers import keywords_counter
from modeling.models import KeywordsExtractionOutput, Taxonomy, ClassificationOutput
# from modeling.prompts import (
#     EXTRACTION_SYSTEM_MESSAGE, TAXONOMY_SYSTEM_MESSAGE, CLASSIFICATION_SYSTEM_MESSAGE
# )
from modeling.utils import (
    load_taxonomy,
    create_classification_samples,
    filter_existing_samples
)


# Configuration constants - hardcoded to remove config.yaml dependency
ROOT_DIR = Path("/Users/luhancheng/Desktop/research-link-technology-landscaping")
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
PROMPTS_DIR = ROOT_DIR / "modeling" / "PromptTemplates"
GRANTS_FILE = DATA_DIR / "active_grants.json"







@task
def classify() -> Task:
    """
    Inspect AI task for classifying research grants into the created taxonomy.
    
    This task takes a taxonomy (created by the refine_taxonomy task) and classifies
    each research grant into the appropriate category and sub-category based on
    the grant's title, summary, and research focus. It automatically skips samples
    that have already been classified to avoid redundant work.
    
    Returns:
        Configured Inspect AI Task
    """
    
    def dataset():
        # Load the taxonomy from the refine_taxonomy task
        taxonomy = load_taxonomy(logs_dir=LOGS_DIR)
        if not taxonomy:
            raise FileNotFoundError("No taxonomy found. Run the refine_taxonomy task first.")
        
        # Load grants data
        grants = load_grants_data(GRANTS_FILE)
        
        # Create classification samples
        all_samples = create_classification_samples(grants, taxonomy)
        
        # Filter out samples that have already been classified
        results_dir = LOGS_DIR / "results" / "classify"
        filtered_samples, skipped_count = filter_existing_samples(
            all_samples, results_dir, "_classification.json"
        )
        
        if skipped_count > 0:
            print(f"⏭️  Skipping {skipped_count} already classified samples, processing {len(filtered_samples)} remaining")
        else:
            print(f"🚀 Processing {len(filtered_samples)} samples for classification")
            
        return filtered_samples
    
    return Task(
        dataset=dataset(),
        solver=[
            system_message(PROMPTS_DIR / "system" / "grant_classification.txt"),
            generate(),
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="grant_classification",
                json_schema=json_schema(ClassificationOutput),
                strict=True
            )
        ),
        hooks=["GrantClassificationHook", "AggregatedResultsHook"]
    )


@task
def refine(
    refinement_factor: float = 0.1,
    num_keywords: int = 100,
    max_iterations: int = 2,
    use_keywords: bool = True,
    keyword_type: Optional[str] = "keywords"
) -> Task:
    """
    Decomposed task that creates and refines taxonomies through group->merge cycles.
    
    This task can either:
    1. Create taxonomies from extracted keywords (use_keywords=True) - keywords are treated as the lowest level taxonomy
    2. Refine taxonomies from grant descriptions (use_keywords=False)
    
    The task chains multiple single-iteration solvers together to perform multiple refinement cycles.
    Each solver performs exactly one group->merge cycle.
    
    Args:
        refinement_factor: Controls number of output categories (output = refinement_factor * input) (default: 0.5)
        num_keywords: Number of keywords/items to use (default: 100)
        max_iterations: Number of refinement iterations to chain together (default: 1)
        use_keywords: Whether to use extracted keywords as base (default: True) 
        keyword_type: Specific keyword type to filter for (default: None = all types)
        
    Returns:
        Configured Inspect AI Task with chained refinement solvers
    """
    
    def dataset():
        if use_keywords:
            # Load keywords from extracted data
            from modeling.utils import load_extracted_keywords
                
            all_keywords = load_extracted_keywords(logs_dir=LOGS_DIR, keyword_type=keyword_type)
            
            # Handle both old dict format and new Pydantic model format
            if hasattr(all_keywords, 'keywords'):
                # New Pydantic KeywordsExtractionOutput model
                if keyword_type:
                    # If a specific type was requested, we already got filtered dict
                    flat_keywords = [kw for sublist in all_keywords.values() for kw in sublist]
                else:
                    # Get all keywords from all categories
                    flat_keywords = (
                        all_keywords.keywords + 
                        all_keywords.methodology_keywords + 
                        all_keywords.application_keywords + 
                        all_keywords.technology_keywords
                    )
            else:
                # Old dict format (when keyword_type filter is used)
                flat_keywords = [kw for sublist in all_keywords.values() for kw in sublist]
            
            unique_keywords = list(set(flat_keywords))  # Remove duplicates
            
            print(f"🔄 Starting decomposed taxonomy refinement")
            print(f"📊 Using {len(unique_keywords)} unique keywords from {len(flat_keywords)} total")
            if keyword_type:
                print(f"🏷️ Filtered for keyword type: {keyword_type}")
        else:
            # Load grants data directly
            grants = load_grants_data(GRANTS_FILE)
            unique_keywords = [f"{grant.title}: {grant.grant_summary}" for grant in grants[:100] if grant.grant_summary]  # Limit for manageability and filter out None summaries
            print(f"🔄 Starting decomposed taxonomy refinement")
            print(f"📊 Using {len(unique_keywords)} grant descriptions")
        
        # Create refinement sample
        sample_metadata = {
            "task_type": "decomposed_refinement",
            "num_keywords": len(unique_keywords),
            "use_keywords": use_keywords,
            "max_iterations": max_iterations,
            "refinement_factor": refinement_factor,
            "expected_output_categories": int(len(unique_keywords[:num_keywords]) * refinement_factor)
        }
        
        if keyword_type:
            sample_metadata["keyword_type_filter"] = keyword_type
        
        sample = Sample(
            input=json.dumps({
                "keywords": unique_keywords[:num_keywords],
                "refinement_factor": refinement_factor,
                "expected_output_categories": int(len(unique_keywords[:num_keywords]) * refinement_factor)
            }),
            metadata=sample_metadata,
            id="decomposed_taxonomy_refinement"
        )
        
        return [sample]
    
    # Import the refine solver
    from modeling.solvers import refine_solver
    
    # Create a chain of refine_solver instances for multiple iterations
    solver_chain = []
    for i in range(max_iterations):
        print(f"Adding refine solver {i+1}/{max_iterations} to chain with refinement_factor={refinement_factor}")
        solver_chain.append(refine_solver(refinement_factor=refinement_factor))
    
    # No need for a finalize_results solver since final taxonomy is no different from other iterations
    
    return Task(
        dataset=dataset(),
        solver=solver_chain,
        # scorer=[
        #     taxonomy_category_counter()
        # ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="taxonomy",
                json_schema=json_schema(Taxonomy),  # Use simple Taxonomy model
                strict=True
            )
        ),
        hooks=["IterativeTaxonomyHook"]
    )

