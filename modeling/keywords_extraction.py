"""
Keywords Extraction Task for Research Grants

This module implements an Inspect AI task that extracts relevant keywords
from research grant titles and summaries using LLM-based analysis.
"""

import json
from typing import List, Optional
from dataclasses import dataclass
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.solver import system_message, generate
from inspect_ai.util import json_schema
from pydantic import BaseModel, Field


class KeywordsExtractionOutput(BaseModel):
    """Pydantic model for structured keywords extraction output."""
    keywords: List[str] = Field(description="5-10 most relevant keywords capturing the research content")
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


def create_keywords_extraction_prompt(grant: Grant) -> str:
    """
    Create a prompt for extracting keywords from a grant.
    
    Args:
        grant: Grant object containing title and summary
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are a research analyst tasked with extracting relevant keywords from research grant information to identify emerging research trends and innovative technologies.

Please analyze the following research grant and extract keywords that capture:
1. The main research domain/field (especially emerging interdisciplinary areas)
2. Key methodologies or approaches (particularly novel or cutting-edge methods)
3. Important technologies or tools mentioned (focus on emerging/innovative technologies)
4. Target applications or outcomes (especially novel applications or emerging needs)
5. Any specific scientific terms or concepts (prioritize new, emerging, or trending terminology)

Grant Title: {grant.title}

Grant Summary: {grant.grant_summary}

Extract 5-10 most relevant keywords for each category. Focus on meaningful, specific keywords that reflect emerging research directions and innovations rather than generic terms. Prioritize:
- Novel technical terms and emerging methodologies
- Cutting-edge technologies and tools
- Emerging interdisciplinary research areas
- Innovative applications and use cases
- New scientific concepts and terminology that indicate research frontiers"""

    return prompt


def create_grant_samples(grants: List[Grant]) -> List[Sample]:
    """
    Convert grants into Inspect AI samples for evaluation.
    
    Args:
        grants: List of Grant objects
        
    Returns:
        List of Sample objects for Inspect AI
    """
    samples = []
    
    for grant in grants:
        # Create the input prompt
        input_text = create_keywords_extraction_prompt(grant)
        
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
    grants_file: str = "/fred/oz318/luhanc/research-link-technology-landscaping/data/active_grants.json"
) -> Task:
    """
    Inspect AI task for extracting keywords from research grants with a focus on emerging research trends.
    
    This task identifies keywords that highlight innovative technologies, novel methodologies,
    emerging research domains, and cutting-edge applications to support trend analysis
    and discovery of research frontiers.
    
    Args:
        grants_file: Path to the grants JSON file
        
    Returns:
        Configured Inspect AI Task
    """
    
    def dataset():
        # Load grants data
        print(f"Loading grants from {grants_file}...")
        grants = load_grants_data(grants_file)
        print(f"Loaded {len(grants)} grants with meaningful content")
        
        # Convert to samples
        samples = create_grant_samples(grants)
        
        return samples
    
    return Task(
        dataset=dataset(),  # Call the function to get the samples
        solver=[
            system_message("""You are an expert research analyst with deep knowledge across multiple academic disciplines and a keen eye for emerging research trends. 
Your task is to extract meaningful keywords from research grant information that would be useful for:
- Identifying emerging research domains and interdisciplinary areas
- Discovering novel methodologies and cutting-edge approaches
- Tracking innovative technologies and emerging tools
- Finding related research projects working on similar frontiers
- Understanding emerging research trends and future directions

Focus on extracting keywords that highlight what's new, innovative, and emerging in the research landscape. Prioritize:
- Technical terms that represent novel concepts or emerging fields
- Methodologies that are cutting-edge or represent new approaches
- Technologies that are innovative or represent emerging tools
- Applications that address new challenges or emerging needs
- Scientific terminology that indicates research at the frontiers of knowledge

Provide accurate, specific, and well-categorized keywords that capture the innovative and emerging aspects of the research."""),
            generate()
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="keywords_extraction",
                json_schema=json_schema(KeywordsExtractionOutput)
            )
        )
    )



