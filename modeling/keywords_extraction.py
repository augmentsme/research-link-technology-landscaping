"""
Keywords Extraction Task for Research Grants

This module implements an Inspect AI task that extracts relevant keywords
from research grant titles and summaries using LLM-based analysis.
"""

import json
import yaml
from typing import List, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.solver import system_message, generate
from inspect_ai.util import json_schema
from pydantic import BaseModel, Field


class KeywordsExtractionOutput(BaseModel):
    """Pydantic model for structured keywords extraction output."""
    keywords: List[str] = Field(description="Most relevant keywords capturing the research content")
    methodology_keywords: List[str] = Field(description="Keywords related to research methodologies or approaches")
    application_keywords: List[str] = Field(description="Keywords related to target applications or outcomes")
    technology_keywords: List[str] = Field(description="Keywords related to technologies or tools mentioned")


@dataclass
class Grant:
    """Data class representing a research grant."""
    id: str
    title: str
    grant_summary: Optional[str]
    funder: Optional[str]
    funding_amount: Optional[float]
    funding_scheme: Optional[str]
    status: Optional[str]


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}, using defaults")
        return {}


def load_grants_data(file_path: str = "/fred/oz318/luhanc/research-link-technology-landscaping/data/active_grants.json") -> List[Grant]:
    """
    Load grants data from JSON file.
    
    Args:
        file_path: Path to the grants JSON file
        
    Returns:
        List of Grant objects
    """
    grants = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for grant_data in data:
            # Only process grants with meaningful content
            if grant_data.get('title') and grant_data.get('grant_summary'):
                grant = Grant(
                    id=grant_data.get('id', ''),
                    title=grant_data.get('title', ''),
                    grant_summary=grant_data.get('grant_summary', ''),
                    funder=grant_data.get('funder', ''),
                    funding_amount=grant_data.get('funding_amount'),
                    funding_scheme=grant_data.get('funding_scheme', ''),
                    status=grant_data.get('status', '')
                )
                grants.append(grant)
    
    except FileNotFoundError:
        print(f"Warning: Could not find grants file at {file_path}")
    except Exception as e:
        print(f"Error loading grants data: {e}")
    
    return grants


def create_keywords_extraction_prompt(grant: Grant, config: Dict[str, Any]) -> str:
    """
    Create a prompt for extracting keywords from a grant using config parameters.
    
    Args:
        grant: Grant object containing title and summary
        config: Configuration dictionary with extraction parameters
        
    Returns:
        Formatted prompt string
    """
    extraction_config = config.get('modeling', {}).get('extraction', {})
    
    # Get configurable parameters
    keywords_per_category = extraction_config.get('keywords_per_category', 8)
    min_keywords = extraction_config.get('min_keywords_per_category', 5)
    max_keywords = extraction_config.get('max_keywords_per_category', 10)
    focus_areas = extraction_config.get('focus_areas', [
        "emerging research domains and interdisciplinary areas",
        "novel methodologies and cutting-edge approaches",
        "innovative technologies and emerging tools"
    ])
    keyword_categories = extraction_config.get('keyword_categories', [
        {"name": "keywords", "description": "most relevant keywords capturing the research content"},
        {"name": "methodology_keywords", "description": "keywords related to research methodologies or approaches"},
        {"name": "application_keywords", "description": "keywords related to target applications or outcomes"},
        {"name": "technology_keywords", "description": "keywords related to technologies or tools mentioned"}
    ])
    
    # Build focus areas text
    focus_text = "\n".join([f"- {area}" for area in focus_areas])
    
    # Build categories text
    categories_text = ""
    for i, category in enumerate(keyword_categories, 1):
        categories_text += f"{i}. {category['description'].title()}\n"
    
    prompt = f"""You are a research analyst tasked with extracting relevant keywords from research grant information to identify emerging research trends and innovative technologies.

Please analyze the following research grant and extract keywords that capture:
{focus_text}

Grant Title: {grant.title}

Grant Summary: {grant.grant_summary}

Extract {keywords_per_category} keywords for each category (minimum {min_keywords}, maximum {max_keywords}). Focus on meaningful, specific keywords that reflect emerging research directions and innovations rather than generic terms.

Categories to extract:
{categories_text}

Prioritize:
- Novel technical terms and emerging methodologies
- Cutting-edge technologies and tools
- Emerging interdisciplinary research areas
- Innovative applications and use cases
- New scientific concepts and terminology that indicate research frontiers"""

    return prompt


def create_grant_samples(grants: List[Grant], config: Dict[str, Any]) -> List[Sample]:
    """
    Convert grants into Inspect AI samples for evaluation.
    
    Args:
        grants: List of Grant objects
        config: Configuration dictionary
        
    Returns:
        List of Sample objects for Inspect AI
    """
    samples = []
    
    for grant in grants:
        # Create the input prompt using config parameters
        input_text = create_keywords_extraction_prompt(grant, config)
        
        # Create metadata for tracking
        metadata = {
            "grant_id": grant.id,
            "funder": grant.funder,
            "funding_amount": grant.funding_amount,
            "funding_scheme": grant.funding_scheme,
            "status": grant.status
        }
        
        # Create sample - no target since this is generative extraction
        sample = Sample(
            input=input_text,
            metadata=metadata,
            id=grant.id
        )
        
        samples.append(sample)
    
    return samples


@task
def extract_grant_keywords(
    grants_file: str = "/fred/oz318/luhanc/research-link-technology-landscaping/data/active_grants.json",
    config_file: str = "config.yaml"
) -> Task:
    """
    Inspect AI task for extracting keywords from research grants with configurable prompts.
    
    This task identifies keywords that highlight innovative technologies, novel methodologies,
    emerging research domains, and cutting-edge applications to support trend analysis
    and discovery of research frontiers.
    
    Args:
        grants_file: Path to the grants JSON file
        config_file: Path to the configuration YAML file
        
    Returns:
        Configured Inspect AI Task
    """
    
    def dataset():
        # Load configuration
        config = load_config(config_file)
        
        # Load grants data
        print(f"Loading grants from {grants_file}...")
        grants = load_grants_data(grants_file)
        print(f"Loaded {len(grants)} grants with meaningful content")
        
        # Convert to samples using config
        samples = create_grant_samples(grants, config)
        
        return samples
    
    # Load config for system message
    config = load_config(config_file)
    extraction_config = config.get('modeling', {}).get('extraction', {})
    system_msg = extraction_config.get('system_message', """You are an expert research analyst with deep knowledge across multiple academic disciplines and a keen eye for emerging research trends. 
Your task is to extract meaningful keywords from research grant information that would be useful for identifying emerging research domains, novel methodologies, innovative technologies, and cutting-edge applications.

Provide accurate, specific, and well-categorized keywords that capture the innovative and emerging aspects of the research.""")
    
    return Task(
        dataset=dataset(),  # Call the function to get the samples
        solver=[
            system_message(system_msg),
            generate()
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="keywords_extraction",
                json_schema=json_schema(KeywordsExtractionOutput)
            )
        )
    )



