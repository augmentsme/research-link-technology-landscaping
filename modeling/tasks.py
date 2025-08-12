"""
Keywords Extraction, Taxonomy Creation, and Grant Classification Tasks for Research Grants

This module implements Inspect AI tasks for:
1. Extracting relevant keywords from research grant titles and summaries
2. Creating a 2-level taxonomy from extracted keywords
3. Classifying research grants into the created taxonomy
4. Loading and processing evaluation results
"""

import json
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import pandas as pd
import os
from dataclasses import dataclass
from collections import defaultdict
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.solver import system_message, generate, use_tools
from inspect_ai.util import json_schema
from inspect_ai.analysis import messages_df, evals_df
from inspect_ai.tool import think
from pydantic import BaseModel, Field


# Configuration constants - hardcoded to remove config.yaml dependency
ROOT_DIR = Path("/Users/luhancheng/Desktop/research-link-technology-landscaping")
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
GRANTS_FILE = DATA_DIR / "active_grants.json"



# Keywords extraction configuration
KEYWORDS_PER_CATEGORY = 8
MAX_KEYWORDS_PER_CATEGORY = 10
MIN_KEYWORDS_PER_CATEGORY = 5

EXTRACTION_SYSTEM_MESSAGE = """You are an expert research analyst with deep knowledge across multiple academic disciplines and a keen eye for emerging research trends. 
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

Provide accurate, specific, and well-categorized keywords that capture the innovative and emerging aspects of the research."""

FOCUS_AREAS = [
    "emerging research domains and interdisciplinary areas",
    "novel methodologies and cutting-edge approaches",
    "innovative technologies and emerging tools",
    "novel applications and emerging needs",
    "new scientific concepts and terminology"
]

KEYWORD_CATEGORIES = [
    {"name": "keywords", "description": "most relevant keywords capturing the research content"},
    {"name": "methodology_keywords", "description": "keywords related to research methodologies or approaches"},
    {"name": "application_keywords", "description": "keywords related to target applications or outcomes"},
    {"name": "technology_keywords", "description": "keywords related to technologies or tools mentioned"}
]

# Taxonomy creation configuration
TAXONOMY_SYSTEM_MESSAGE = """You are an expert in research taxonomy design and knowledge organization with deep understanding of research domains across disciplines.

Your task is to create a 2-level taxonomy from research keywords that will help organize and classify research grants. The taxonomy should:
1. Capture the main research domains and areas represented in the keywords
2. Provide a hierarchical structure with top-level categories and sub-categories
3. Be comprehensive enough to classify diverse research grants
4. Use clear, standardized terminology from research literature

Guidelines:
- Create broad top-level categories that represent major research domains
- Under each top-level category, create more specific sub-categories
- Ensure categories are mutually exclusive where possible
- Use terminology that researchers would recognize and understand
- Balance comprehensiveness with usability (not too many categories)

Remember: This taxonomy will be used to classify research grants, so it should reflect the diversity and structure of research domains."""

TAXONOMY_INSTRUCTIONS = [
    "Create broad top-level categories representing major research domains",
    "Develop specific sub-categories under each top-level category",
    "Ensure categories are mutually exclusive where possible",
    "Use standardized terminology from research literature",
    "Balance comprehensiveness with usability",
    "Consider the diversity of research domains represented in the keywords"
]

# Grant classification configuration  
CLASSIFICATION_SYSTEM_MESSAGE = """You are an expert research analyst with deep knowledge across multiple academic disciplines and extensive experience in research classification.

Your task is to classify research grants into a predefined 2-level taxonomy. For each grant, you need to:
1. Analyze the grant title and summary to understand the research domain and focus
2. Identify the most appropriate top-level category and sub-category from the taxonomy
3. Provide clear reasoning for your classification decisions
4. Handle cases where grants might span multiple categories

Guidelines:
- Focus on the primary research domain and methodology of the grant
- Consider the main applications and technologies mentioned
- Use the grant's explicit research objectives to guide classification
- When grants span multiple areas, choose the most prominent or primary focus
- Provide consistent classification based on research content, not just keywords

Remember: Accurate classification helps in understanding research trends and identifying related projects across the research landscape."""

CLASSIFICATION_INSTRUCTIONS = [
    "Analyze grant title and summary to understand research domain",
    "Identify the most appropriate top-level category and sub-category",
    "Focus on primary research domain and methodology",
    "Consider main applications and technologies",
    "Use explicit research objectives to guide classification",
    "Choose primary focus when grants span multiple areas",
    "Provide consistent classification based on research content"
]


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

    def asdict(self):
        return {
            "id": self.id,
            "title": self.title,
            "grant_summary": self.grant_summary,
            "funder": self.funder,
            "funding_amount": self.funding_amount,
            "funding_scheme": self.funding_scheme,
            "status": self.status
        }


class KeywordsExtractionOutput(BaseModel):
    """Pydantic model for structured keywords extraction output."""
    model_config = {"extra": "forbid"}
    
    keywords: List[str] = Field(description="Most relevant keywords capturing the research content")
    methodology_keywords: List[str] = Field(description="Keywords related to research methodologies or approaches")
    application_keywords: List[str] = Field(description="Keywords related to target applications or outcomes")
    technology_keywords: List[str] = Field(description="Keywords related to technologies or tools mentioned")




class SubCategory(BaseModel):
    """A sub-category within a top-level category."""
    model_config = {"extra": "forbid"}
    
    name: str = Field(description="Name of the sub-category")
    description: str = Field(description="Brief description of what this sub-category covers")


class TopLevelCategory(BaseModel):
    """A top-level category in the taxonomy."""
    model_config = {"extra": "forbid"}
    
    name: str = Field(description="Name of the top-level category") 
    description: str = Field(description="Brief description of what this category covers")
    subcategories: List[SubCategory] = Field(description="List of sub-categories under this top-level category")


class TaxonomyOutput(BaseModel):
    """Pydantic model for 2-level taxonomy creation output."""
    model_config = {"extra": "forbid"}
    
    taxonomy: List[TopLevelCategory] = Field(description="List of top-level categories with their sub-categories")


class GrantClassification(BaseModel):
    """Classification result for a single grant."""
    model_config = {"extra": "forbid"}
    
    top_level_category: str = Field(description="The top-level category assigned to this grant")
    subcategory: str = Field(description="The sub-category assigned to this grant")
    reasoning: str = Field(description="Brief explanation for why this classification was chosen")


class ClassificationOutput(BaseModel):
    """Pydantic model for grant classification output."""
    model_config = {"extra": "forbid"}
    
    classification: GrantClassification = Field(description="Classification result for the grant")




def load_extracted_keywords(logs_dir) -> Dict[str, List[str]]:

    evals = evals_df(logs_dir)
    task_evals = evals[evals.task_name == "extract"]
    log_paths = task_evals.log.iloc[0]
    
    all_keywords = {"keywords": [], "methodology_keywords": [], "application_keywords": [], "technology_keywords": []}
    
    # Load messages from all matching log files
    messages = messages_df(log_paths)
    assistant_messages = messages[messages.role == "assistant"]
    # return assistant_messages

    for _, row in assistant_messages.iterrows():
        content = row['content']
        keywords_result = json.loads(content)
        for category, keywords in keywords_result.items():
            all_keywords[category].extend(keywords)
    
    return dict(all_keywords)



def load_taxonomy(logs_dir) -> Optional[Dict[str, Any]]:
    """
    Load the 2-level taxonomy from the identify task evaluation results.
    
    Args:
        logs_dir: Directory to search for identify evaluation files
        
    Returns:
        Dictionary containing the taxonomy structure, or None if not found
    """
    try:
        evals = evals_df(logs_dir)
        identify_evals = evals[evals.task_name == "identify"]
        if identify_evals.empty:
            return None
            
        logpath = identify_evals.log.iloc[0]
        messages = messages_df(logpath)
        content = messages[messages.role == "assistant"].content.item()
        taxonomy_result = json.loads(content)
        
        return taxonomy_result
    except Exception as e:
        print(f"Error loading taxonomy: {e}")
        return None


def load_grant_classifications(logs_dir) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Load grant classifications from the classify task evaluation results.
    
    Args:
        logs_dir: Directory to search for classify evaluation files
        
    Returns:
        Dictionary mapping grant IDs to their classifications, or None if not found
    """
    try:
        evals = evals_df(logs_dir)
        classify_evals = evals[evals.task_name == "classify"]
        if classify_evals.empty:
            return None
            
        logpath = classify_evals.log.iloc[0]
        messages = messages_df(logpath)
        assistant_messages = messages[messages.role == "assistant"]
        
        classifications = {}
        for _, row in assistant_messages.iterrows():
            content = row['content']
            classification_result = json.loads(content)
            
            # Extract grant ID from the sample metadata if available
            # This would need to be implemented based on how the samples are structured
            # For now, we'll use the row index or sample ID
            sample_id = row.get('sample_id', f"sample_{row.name}")
            classifications[sample_id] = classification_result
            
        return classifications
    except Exception as e:
        print(f"Error loading grant classifications: {e}")
        return None



def load_grants_data(file_path: str, as_dataframe=False) -> List[Grant]:
    """
    Load grants data from JSON file.
    
    Args:
        file_path: Path to the grants JSON file
        
    Returns:
        List of Grant objects
    """
    grants = []
    
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
    if as_dataframe:
        # Convert to DataFrame if requested
        return pd.DataFrame([grant.asdict() for grant in grants]).set_index("id")
    return grants


def create_keywords_extraction_prompt(grant: Grant) -> str:
    """
    Create a prompt for extracting keywords from a grant.
    
    Args:
        grant: Grant object containing title and summary
        
    Returns:
        Formatted prompt string
    """
    # Use hardcoded constants
    keywords_per_category = KEYWORDS_PER_CATEGORY
    min_keywords = MIN_KEYWORDS_PER_CATEGORY
    max_keywords = MAX_KEYWORDS_PER_CATEGORY
    focus_areas = FOCUS_AREAS
    keyword_categories = KEYWORD_CATEGORIES
    
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

IMPORTANT: You MUST respond with valid JSON only. Do not include explanations, schema definitions, or any other text outside the JSON structure.

REQUIRED JSON SCHEMA - Follow this structure exactly:
{{
    "keywords": ["keyword1", "keyword2", ...],
    "methodology_keywords": ["method1", "method2", ...], 
    "application_keywords": ["app1", "app2", ...],
    "technology_keywords": ["tech1", "tech2", ...]
}}

All four fields (keywords, methodology_keywords, application_keywords, technology_keywords) are REQUIRED and must be arrays of strings.

If the grant information is insufficient or unclear, return empty arrays for each category but maintain the required JSON structure.

Extract {keywords_per_category} keywords for each category (minimum {min_keywords}, maximum {max_keywords}). Focus on meaningful, specific keywords that reflect emerging research directions and innovations rather than generic terms.

Categories to extract:
{categories_text}

Prioritize:
- Novel technical terms and emerging methodologies
- Cutting-edge technologies and tools
- Emerging interdisciplinary research areas
- Innovative applications and use cases
- New scientific concepts and terminology that indicate research frontiers

CRITICAL: Your response must be valid JSON that strictly follows the required schema. No additional text, explanations, or deviations from the schema are allowed."""

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
        # Create the input prompt using hardcoded constants
        input_text = create_keywords_extraction_prompt(grant)
        
        # Create metadata for tracking
        metadata = {
            "grant_id": grant.id,
            "funder": grant.funder,
            "funding_amount": grant.funding_amount,
            "funding_scheme": grant.funding_scheme,
            "status": grant.status,
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
def extract() -> Task:
    """
    Inspect AI task for extracting keywords from research grants.
    
    This task identifies keywords that highlight innovative technologies, novel methodologies,
    emerging research domains, and cutting-edge applications to support trend analysis
    and discovery of research frontiers.
        
    Returns:
        Configured Inspect AI Task
    """
    
    def dataset():
        grants = load_grants_data(GRANTS_FILE)
        samples = create_grant_samples(grants)
        
        return samples
    
    return Task(
        dataset=dataset(),  # Call the function to get the samples
        solver=[
            system_message(EXTRACTION_SYSTEM_MESSAGE),
            generate()
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="keywords_extraction",
                json_schema=json_schema(KeywordsExtractionOutput),
                strict=True
            )
        )
    )





def create_taxonomy_creation_prompt(keywords: List[str]) -> str:
    """
    Create a prompt for creating a 2-level taxonomy from extracted keywords.
    
    Args:
        keywords: List of keywords to create taxonomy from
        
    Returns:
        Formatted prompt string
    """
    
    # Build instructions text
    instructions_text = "\n".join([f"- {instruction}" for instruction in TAXONOMY_INSTRUCTIONS])
    
    # Create keyword list organized by category if available
    unique_keywords = sorted(set(keywords))
    keywords_text = "\n".join([f"- {kw}" for kw in unique_keywords])
    
    prompt = f"""I have extracted {len(unique_keywords)} research keywords from grant data. Your task is to create a comprehensive 2-level taxonomy that can be used to classify research grants into meaningful categories.

**Instructions for taxonomy creation:**
{instructions_text}

**Keywords to analyze:**
{keywords_text}

IMPORTANT: You MUST respond with valid JSON only. Do not include explanations or any other text outside the JSON structure.

REQUIRED JSON SCHEMA - Follow this structure exactly:
{{
    "taxonomy": [
        {{
            "name": "Computer Science and AI",
            "description": "Research in computing, artificial intelligence, and related technologies",
            "subcategories": [
                {{
                    "name": "Machine Learning",
                    "description": "Research focused on machine learning algorithms and applications"
                }},
                {{
                    "name": "Data Science",
                    "description": "Research in data analysis, big data, and data mining"
                }}
            ]
        }},
        {{
            "name": "Life Sciences",
            "description": "Research in biology, medicine, and life sciences",
            "subcategories": [
                {{
                    "name": "Biomedical Research", 
                    "description": "Medical research and healthcare applications"
                }},
                {{
                    "name": "Molecular Biology",
                    "description": "Research at the molecular level of biological processes"
                }}
            ]
        }}
    ]
}}

Requirements:
- Create 6-10 top-level categories that capture major research domains
- Each top-level category should have 3-8 sub-categories  
- Categories should be comprehensive enough to classify diverse research grants
- Use clear, standardized terminology from research literature
- Ensure categories are mutually exclusive where possible
- Sub-categories should be specific enough to be meaningful for classification

CRITICAL: Your response must be valid JSON that strictly follows the required schema. No additional text, explanations, or deviations from the schema are allowed."""

    return prompt


def create_taxonomy_creation_prompt_from_grants(grants: List[Grant]) -> str:
    """
    Create a prompt for creating a 2-level taxonomy by analyzing all grant titles and summaries.

    Args:
        grants: List of Grant objects

    Returns:
        Formatted prompt string
    """
    # Build instructions text
    instructions_text = "\n".join([f"- {instruction}" for instruction in TAXONOMY_INSTRUCTIONS])

    # Build grants text
    grant_entries = []
    for i, g in enumerate(grants, start=1):
        title = (g.title or "").strip()
        summary = (g.grant_summary or "").strip()
        grant_entries.append(f"{i}. Title: {title}\n   Summary: {summary}")
    grants_text = "\n\n".join(grant_entries)

    prompt = f"""I have a collection of {len(grants)} research grants (titles and summaries). Your task is to analyze these grants and create a comprehensive 2-level taxonomy that can be used to classify research grants into meaningful categories.

**Instructions for taxonomy creation:**
{instructions_text}

**Grants to analyze:**
{grants_text}

IMPORTANT: You MUST respond with valid JSON only. Do not include explanations or any other text outside the JSON structure.

REQUIRED JSON SCHEMA - Follow this structure exactly:
{{
    "taxonomy": [
        {{
            "name": "Computer Science and AI",
            "description": "Research in computing, artificial intelligence, and related technologies",
            "subcategories": [
                {{
                    "name": "Machine Learning",
                    "description": "Research focused on machine learning algorithms and applications"
                }},
                {{
                    "name": "Data Science",
                    "description": "Research in data analysis, big data, and data mining"
                }}
            ]
        }},
        {{
            "name": "Life Sciences",
            "description": "Research in biology, medicine, and life sciences",
            "subcategories": [
                {{
                    "name": "Biomedical Research",
                    "description": "Medical research and healthcare applications"
                }},
                {{
                    "name": "Molecular Biology",
                    "description": "Research at the molecular level of biological processes"
                }}
            ]
        }}
    ]
}}

Requirements:
- Create 6-10 top-level categories that capture major research domains
- Each top-level category should have 3-8 sub-categories
- Categories should be comprehensive enough to classify diverse research grants
- Use clear, standardized terminology from research literature
- Ensure categories are mutually exclusive where possible
- Sub-categories should be specific enough to be meaningful for classification

CRITICAL: Your response must be valid JSON that strictly follows the required schema. No additional text, explanations, or deviations from the schema are allowed."""

    return prompt


@task
def identify(use_keywords: bool = True) -> Task:
    """
    Inspect AI task for creating a 2-level taxonomy from either extracted keywords or directly from grants.

    Args:
        use_keywords: When True (default), build taxonomy from extracted keywords. When False, analyze all grants to build taxonomy.

    Returns:
        Configured Inspect AI Task
    """

    def dataset():
        if use_keywords:
            # Maintain current behavior: build taxonomy from extracted keywords
            all_keywords = load_extracted_keywords(logs_dir=LOGS_DIR)
            flat_keywords = [kw for sublist in all_keywords.values() for kw in sublist]
            prompt = create_taxonomy_creation_prompt(flat_keywords)
            sample = Sample(
                input=prompt,
                metadata={
                    "mode": "keywords",
                    "total_keywords": len(flat_keywords),
                    "unique_keywords": len(set(flat_keywords)),
                }
            )
            return [sample]
        else:
            # New behavior: build taxonomy by analyzing all grants directly
            grants = load_grants_data(GRANTS_FILE)
            prompt = create_taxonomy_creation_prompt_from_grants(grants)
            sample = Sample(
                input=prompt,
                metadata={
                    "mode": "grants",
                    "total_grants": len(grants),
                }
            )
            return [sample]

    return Task(
        dataset=dataset(),
        solver=[
            system_message(TAXONOMY_SYSTEM_MESSAGE),
            generate(),
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="taxonomy_creation",
                json_schema=json_schema(TaxonomyOutput),
                strict=True
            )
        )
    )


@task
def classify() -> Task:
    """
    Inspect AI task for classifying research grants into the created taxonomy.
    
    This task takes the 2-level taxonomy created by the identify task and classifies
    each research grant into the appropriate category and sub-category based on
    the grant's title, summary, and research focus.
    
    Returns:
        Configured Inspect AI Task
    """
    
    def dataset():
        # Load the taxonomy from the identify task
        taxonomy = load_taxonomy(logs_dir=LOGS_DIR)
        if not taxonomy:
            raise ValueError("No taxonomy found. Run identify task first to create taxonomy.")
        
        # Load grants data
        grants = load_grants_data(GRANTS_FILE)
        
        samples = []
        for grant in grants:
            # Create classification prompt for this grant
            prompt = create_grant_classification_prompt(grant, taxonomy)
            
            # Create metadata for tracking
            metadata = {
                "grant_id": grant.id,
                "funder": grant.funder,
                "funding_amount": grant.funding_amount,
                "funding_scheme": grant.funding_scheme,
                "status": grant.status,
                "taxonomy_categories": len(taxonomy.get("taxonomy", [])) if taxonomy else 0
            }
            
            sample = Sample(
                input=prompt,
                metadata=metadata,
                id=grant.id
            )
            
            samples.append(sample)
        
        return samples
    
    return Task(
        dataset=dataset(),
        solver=[
            system_message(CLASSIFICATION_SYSTEM_MESSAGE),
            generate(),
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="grant_classification",
                json_schema=json_schema(ClassificationOutput),
                strict=True
            )
        )
    )


def create_grant_classification_prompt(grant: Grant, taxonomy: Dict[str, Any]) -> str:
    """
    Create a prompt for classifying a grant into the created taxonomy.
    
    Args:
        grant: Grant object containing title and summary
        taxonomy: The taxonomy structure from the identify task
        
    Returns:
        Formatted prompt string
    """
    
    # Build instructions text
    instructions_text = "\n".join([f"- {instruction}" for instruction in CLASSIFICATION_INSTRUCTIONS])
    
    # Build taxonomy text for reference
    taxonomy_text = ""
    if "taxonomy" in taxonomy:
        for top_cat in taxonomy["taxonomy"]:
            taxonomy_text += f"\n{top_cat['name']}: {top_cat['description']}\n"
            for sub_cat in top_cat.get('subcategories', []):
                taxonomy_text += f"  - {sub_cat['name']}: {sub_cat['description']}\n"
    
    prompt = f"""You are a research analyst tasked with classifying a research grant into a predefined 2-level taxonomy.

**Classification Guidelines:**
{instructions_text}

**Available Taxonomy:**
{taxonomy_text}

**Grant to Classify:**

Grant Title: {grant.title}

Grant Summary: {grant.grant_summary}

IMPORTANT: You MUST respond with valid JSON only. Do not include explanations or any other text outside the JSON structure.

REQUIRED JSON SCHEMA - Follow this structure exactly:
{{
    "classification": {{
        "top_level_category": "Computer Science and AI",
        "subcategory": "Machine Learning", 
        "reasoning": "This grant focuses on developing new machine learning algorithms for medical image analysis, making it primarily a computer science/AI project with machine learning as the specific approach."
    }}
}}

Requirements:
- Choose exactly one top-level category and one sub-category from the provided taxonomy
- The top-level category and sub-category must exist in the taxonomy
- Provide clear reasoning (2-3 sentences) explaining why this classification was chosen
- Focus on the primary research domain and methodology described in the grant
- When grants span multiple areas, choose the most prominent focus

CRITICAL: Your response must be valid JSON that strictly follows the required schema. No additional text, explanations, or deviations from the schema are allowed."""

    return prompt



