"""
Keywords Harmonisation Task for Research Grants

This module implements an Inspect AI task that harmonizes keywords 
extracted from research grants by consolidating variants and synonyms 
into standardized terms.
"""

import yaml
from typing import List, Dict, Any
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.solver import system_message, generate
from inspect_ai.util import json_schema
from pydantic import BaseModel, Field

from modeling.ioutils import collect_keywords_from_evaluation_results


class KeywordsHarmonisationOutput(BaseModel):
    """Pydantic model for structured keywords harmonisation output."""
    harmonised_keywords: List[str] = Field(description="List of harmonised/standardized keywords")
    keyword_mappings: Dict[str, str] = Field(description="Direct mapping from original keyword to its harmonised form")
    merged_groups: Dict[str, List[str]] = Field(description="Groups showing which original keywords were merged into each harmonised keyword")
    unchanged_keywords: List[str] = Field(description="Keywords that required no harmonisation")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}, using defaults")
        return {}


def create_harmonisation_prompt(keywords: List[str], config: Dict[str, Any]) -> str:
    """
    Create a harmonisation prompt using config parameters.
    
    Args:
        keywords: List of keywords to harmonise
        config: Configuration dictionary with harmonisation parameters
        
    Returns:
        Formatted prompt string
    """
    harmonisation_config = config.get('modeling', {}).get('harmonisation', {})
    
    # Get configurable parameters
    instructions = harmonisation_config.get('instructions', [
        "Identify and merge similar keywords, variants, and synonyms",
        "Standardize terminology using commonly accepted scientific terms",
        "Resolve inconsistencies in naming conventions",
        "Preserve semantic richness while reducing redundancy"
    ])
    
    # Build instructions text
    instructions_text = "\n".join([f"- {instruction}" for instruction in instructions])
    
    keywords_text = "\n".join([f"- {kw}" for kw in sorted(set(keywords))])
    
    prompt = f"""I have extracted {len(set(keywords))} research keywords from grant data that need harmonisation. Please consolidate these keywords by identifying variants, synonyms, and similar terms that should be merged.

**Instructions:**
{instructions_text}

**Keywords to harmonise:**
{keywords_text}

Please provide:
1. **harmonised_keywords**: Final list of standardized keywords after merging similar terms
2. **keyword_mappings**: Complete mapping showing what each original keyword maps to
3. **merged_groups**: Groups showing which original keywords were consolidated into each harmonised keyword (only include groups with 2+ original keywords)
4. **unchanged_keywords**: Keywords that required no changes

Focus on semantic similarity while preserving technical precision. Don't create higher-level categories - just clean up variants and synonyms."""

    return prompt


@task
def harmonise_research_keywords(
    data_dir: str = "/home/lcheng/oz318/research-link-technology-landscaping/data",
    output_file: str = "harmonised_keywords.json",
    config_file: str = "config.yaml"
) -> Task:
    """
    Inspect AI task for harmonising research keywords by consolidating variants and synonyms.
    
    This creates a single comprehensive query with ALL keywords extracted from the previous step,
    allowing the LLM to identify and merge similar terms, spelling variants, and synonyms
    while preserving the technical specificity and meaning of the original research keywords.
    
    Args:
        data_dir: Directory containing the data files
        output_file: Output file for saving harmonisation results
        config_file: Path to the configuration YAML file
        
    Returns:
        Configured Inspect AI Task
    """
    
    def dataset():
        # Load configuration
        config = load_config(config_file)
        
        # Collect all keywords from evaluation results
        all_keywords = collect_keywords_from_evaluation_results()
        
        # Flatten all keywords into a single comprehensive list
        all_keywords_flat = []
        category_info = {}
        
        for category, keywords in all_keywords.items():
            all_keywords_flat.extend(keywords)
            category_info[category] = len(keywords)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in all_keywords_flat:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        # Create prompt using config parameters
        prompt = create_harmonisation_prompt(unique_keywords, config)
        
        sample = Sample(
            input=prompt,
            metadata={
                "total_keywords": len(unique_keywords),
                "original_categories": category_info,
                "data_dir": data_dir,
                "output_file": output_file
            }
        )
        
        return [sample]
    
    # Load config for system message
    config = load_config(config_file)
    harmonisation_config = config.get('modeling', {}).get('harmonisation', {})
    system_msg = harmonisation_config.get('system_message', """You are an expert research analyst specializing in keyword harmonisation and standardization.

Your goal is to harmonise research keywords by identifying and merging variants, synonyms, and different expressions of the same concepts while preserving technical precision and meaning.

Focus on harmonisation that merges true synonyms and variants while preserving important technical distinctions between different concepts.""")
    
    return Task(
        dataset=dataset(),
        solver=[
            system_message(system_msg),
            generate()
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="keywords_harmonisation",
                json_schema=json_schema(KeywordsHarmonisationOutput)
            )
        )
    )