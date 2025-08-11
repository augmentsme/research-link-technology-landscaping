"""
Keywords Extraction and Harmonisation Tasks for Research Grants

This module implements Inspect AI tasks for:
1. Extracting relevant keywords from research grant titles and summaries
2. Harmonising extracted keywords by consolidating variants and synonyms
3. Loading and processing evaluation results
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

# Keywords harmonisation configuration
HARMONISATION_SYSTEM_MESSAGE = """You are an expert in research terminology and knowledge organization with deep understanding of scientific vocabulary across disciplines.

Your task is to harmonise a list of research keywords by:
1. Identifying and merging similar keywords, variants, and synonyms
2. Standardizing terminology using the most commonly accepted scientific terms
3. Resolving inconsistencies in naming conventions
4. Preserving the semantic richness while reducing redundancy

Guidelines:
- Merge keywords that are clearly variants of the same concept
- Use the most standard, widely-accepted term when merging
- Preserve technical precision - don't over-generalize
- Maintain the granularity level appropriate for research analysis
- Keep domain-specific terminology when it adds value

Remember: This is about cleaning up keyword variants, NOT about creating higher-level topic categories or abstractions."""

HARMONISATION_INSTRUCTIONS = [
    "Identify and merge similar keywords, variants, and synonyms",
    "Standardize terminology using commonly accepted scientific terms",
    "Resolve inconsistencies in naming conventions",
    "Preserve semantic richness while reducing redundancy",
    "Use the most standard, widely-accepted term when merging",
    "Preserve technical precision - don't over-generalize",
    "Maintain appropriate granularity level for research analysis"
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




class KeywordMapping(BaseModel):
    """Individual keyword mapping from original indices to harmonised keyword."""
    model_config = {"extra": "forbid"}
    
    original_index: List[int] = Field(description="List of indices of original keywords that map to this harmonised keyword")
    harmonised: str = Field(description="The harmonised/standardized keyword")


class KeywordsHarmonisationOutput(BaseModel):
    """Pydantic model for structured keywords harmonisation output."""
    model_config = {"extra": "forbid"}
    
    keyword_mappings: List[KeywordMapping] = Field(description="List of mappings from original keyword indices to harmonised keywords")


class KeywordGroup(BaseModel):
    """Individual keyword group containing indices that should be harmonised together."""
    model_config = {"extra": "forbid"}
    
    indices: List[int] = Field(description="List of keyword indices that should be harmonised together")


class KeywordGroupsOutput(BaseModel):
    """Pydantic model for keyword grouping output."""
    model_config = {"extra": "forbid"}
    
    keyword_groups: List[KeywordGroup] = Field(description="List of groups of keyword indices that should be harmonised together")


class GroupHarmonisationOutput(BaseModel):
    """Pydantic model for harmonising a single group of keywords."""
    model_config = {"extra": "forbid"}
    
    harmonised_keyword: str = Field(description="The final harmonised keyword for this group")




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



def load_harmonised_keywords(logs_dir) -> Optional[Dict[str, Any]]:
    """
    Load harmonised keywords from Inspect AI evaluation result files.
    
    Args:
        logs_dir: Directory to search for harmonisation evaluation files
        
    Returns:
        Dictionary containing harmonised keywords data with string-based mappings, or None if not found
    """
    evals = evals_df(logs_dir)
    logpath = evals[evals.task_name == "harmonise"].log.iloc[0]
    messages = messages_df(logpath)
    content = messages[messages.role == "assistant"].content.item()
    harmonisation_result = json.loads(content)
    
    # Convert index-based mappings to string-based mappings for compatibility
    if "keyword_mappings" in harmonisation_result:
        # Get the original keywords in the same order they were indexed
        all_keywords = load_extracted_keywords(logs_dir)
        flat_keywords = [kw for sublist in all_keywords.values() for kw in sublist]
        unique_keywords = sorted(set(flat_keywords))
        
        # Convert index mappings to string mappings
        index_mappings = harmonisation_result["keyword_mappings"]
        
        string_mappings = {}
        # Handle new format: [{"original_index": [0, 1, 5], "harmonised": "machine learning"}, ...]
        if isinstance(index_mappings, list) and len(index_mappings) > 0:
            first_mapping = index_mappings[0]
            if isinstance(first_mapping, dict) and "harmonised" in first_mapping:
                # New format with direct harmonised keywords
                for mapping in index_mappings:
                    original_indices = mapping["original_index"]
                    harmonised_keyword = mapping["harmonised"]
                    for original_idx in original_indices:
                        if 0 <= original_idx < len(unique_keywords):
                            original_keyword = unique_keywords[original_idx]
                            string_mappings[original_keyword] = harmonised_keyword
            elif isinstance(first_mapping, dict) and "harmonised_index" in first_mapping:
                # Old format with harmonised_index
                harmonised_keywords = harmonisation_result.get("harmonised_keywords", [])
                for mapping in index_mappings:
                    original_idx = mapping["original_index"]
                    harmonised_idx = mapping["harmonised_index"]
                    if 0 <= original_idx < len(unique_keywords) and 0 <= harmonised_idx < len(harmonised_keywords):
                        original_keyword = unique_keywords[original_idx]
                        harmonised_keyword = harmonised_keywords[harmonised_idx]
                        string_mappings[original_keyword] = harmonised_keyword
            else:
                # Even older dict format: {"0": 0, "1": 2, ...}
                harmonised_keywords = harmonisation_result.get("harmonised_keywords", [])
                for original_idx_str, harmonised_idx in index_mappings.items():
                    original_idx = int(original_idx_str)
                    if 0 <= original_idx < len(unique_keywords) and 0 <= harmonised_idx < len(harmonised_keywords):
                        original_keyword = unique_keywords[original_idx]
                        harmonised_keyword = harmonised_keywords[harmonised_idx]
                        string_mappings[original_keyword] = harmonised_keyword
        
        harmonisation_result["keyword_mappings"] = string_mappings
    
    return harmonisation_result


def load_keyword_groups(logs_dir) -> Optional[List[List[int]]]:
    """
    Load keyword groups from the identify task evaluation results.
    
    Args:
        logs_dir: Directory to search for evaluation files
        
    Returns:
        List of lists where each sublist contains indices of keywords that should be harmonised together
    """
    try:
        evals = evals_df(logs_dir)
        group_evals = evals[evals.task_name == "identify"]
        if group_evals.empty:
            return None
            
        logpath = group_evals.log.iloc[0]
        messages = messages_df(logpath)
        content = messages[messages.role == "assistant"].content.item()
        groups_result = json.loads(content)
        
        if "keyword_groups" in groups_result:
            return [group["indices"] for group in groups_result["keyword_groups"]]
        
        return None
    except Exception as e:
        print(f"Error loading keyword groups: {e}")
        return None


def load_group_harmonisations(logs_dir) -> Optional[Dict[str, str]]:
    """
    Load harmonised keywords from the harmonise task evaluation results.
    
    Args:
        logs_dir: Directory to search for evaluation files
        
    Returns:
        Dictionary mapping original keywords to harmonised keywords
    """
    try:
        evals = evals_df(logs_dir)
        harmonise_group_evals = evals[evals.task_name == "harmonise"]
        if harmonise_group_evals.empty:
            return None
            
        # Load original keywords for mapping
        all_keywords = load_extracted_keywords(logs_dir)
        flat_keywords = [kw for sublist in all_keywords.values() for kw in sublist]
        unique_keywords = sorted(set(flat_keywords))
        
        # Load keyword groups
        keyword_groups = load_keyword_groups(logs_dir)
        if not keyword_groups:
            return None
        
        # Load harmonisation results for each group
        string_mappings = {}
        
        log_path = harmonise_group_evals.log.iloc[0]
        messages = messages_df(log_path)
        
        # Group messages by sample
        user_messages = messages[messages.role == "user"]
        assistant_messages = messages[messages.role == "assistant"]
        
        # Process each sample (each group)
        for i in range(len(assistant_messages)):
            try:
                # Get assistant response
                assistant_row = assistant_messages.iloc[i]
                content = assistant_row['content']
                harmonisation_result = json.loads(content)
                
                # Get corresponding user message to find metadata
                user_row = user_messages.iloc[i]
                sample_metadata = user_row.get('metadata', {}) if hasattr(user_row, 'get') else {}
                
                # Try different ways to access metadata
                if not sample_metadata and hasattr(user_row, 'to_dict'):
                    user_dict = user_row.to_dict()
                    sample_metadata = user_dict.get('metadata', {})
                
                # If metadata access doesn't work, use the group index from the pattern
                group_indices = sample_metadata.get('group_indices', [])
                if not group_indices and i < len(keyword_groups):
                    group_indices = keyword_groups[i]
                
                if "harmonised_keyword" in harmonisation_result and group_indices:
                    harmonised_keyword = harmonisation_result["harmonised_keyword"]
                    
                    # Map all original keywords in this group to the harmonised keyword
                    for idx in group_indices:
                        if 0 <= idx < len(unique_keywords):
                            original_keyword = unique_keywords[idx]
                            string_mappings[original_keyword] = harmonised_keyword
                            
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        return string_mappings
        
    except Exception as e:
        print(f"Error loading group harmonisations: {e}")
        return None


def combine_two_step_harmonisation_results(logs_dir) -> Optional[Dict[str, Any]]:
    """
    Combine the results from the two-step harmonisation process (identify + harmonise).
    
    Args:
        logs_dir: Directory containing evaluation results
        
    Returns:
        Dictionary in the same format as the original harmonise task for compatibility
    """
    try:
        # Load results from both steps
        keyword_groups = load_keyword_groups(logs_dir)
        group_harmonisations = load_group_harmonisations(logs_dir)
        
        if not keyword_groups or not group_harmonisations:
            return None
        
        # Load original keywords for compatibility
        all_keywords = load_extracted_keywords(logs_dir)
        flat_keywords = [kw for sublist in all_keywords.values() for kw in sublist]
        unique_keywords = sorted(set(flat_keywords))
        
        # Build the result in the same format as the original harmonise task
        keyword_mappings = []
        
        for group_indices in keyword_groups:
            if not group_indices:
                continue
                
            # Get the harmonised keyword for this group
            # Use the first keyword in the group to look up the harmonised result
            if group_indices[0] < len(unique_keywords):
                first_keyword = unique_keywords[group_indices[0]]
                harmonised_keyword = group_harmonisations.get(first_keyword)
                
                if harmonised_keyword:
                    mapping = KeywordMapping(
                        original_index=group_indices,
                        harmonised=harmonised_keyword
                    )
                    keyword_mappings.append(mapping)
        
        return {
            "keyword_mappings": group_harmonisations,  # String-based mapping for compatibility
            "original_keyword_mappings": [mapping.dict() for mapping in keyword_mappings]  # Index-based for analysis
        }
        
    except Exception as e:
        print(f"Error combining two-step harmonisation results: {e}")
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





def create_harmonisation_prompt(keywords: List[str]) -> str:
    """
    Create a harmonisation prompt.
    
    Args:
        keywords: List of keywords to harmonise
        
    Returns:
        Formatted prompt string
    """
    
    # Build instructions text
    instructions_text = "\n".join([f"- {instruction}" for instruction in HARMONISATION_INSTRUCTIONS])
    
    # Create indexed keywords list
    unique_keywords = sorted(set(keywords))
    keywords_text = "\n".join([f"{i}: {kw}" for i, kw in enumerate(unique_keywords)])
    
    prompt = f"""I have extracted {len(unique_keywords)} research keywords from grant data that need harmonisation. Please consolidate these keywords by identifying variants, synonyms, and similar terms that should be merged.

**Instructions:**
{instructions_text}

**Keywords to harmonise (with indices):**
{keywords_text}

IMPORTANT: You MUST respond with valid JSON only. Do not include explanations or any other text outside the JSON structure.

REQUIRED JSON SCHEMA - Follow this structure exactly:
{{
    "keyword_mappings": [
        {{"original_index": [0, 1, 5], "harmonised": "machine learning"}},
        {{"original_index": [2, 8], "harmonised": "artificial intelligence"}},
        {{"original_index": [3], "harmonised": "neural networks"}},
        ...
    ]
}}

The field is REQUIRED:
- keyword_mappings: Array of mapping objects, each containing:
  - original_index: Array of integers representing indices of original keywords that map to this harmonised keyword
  - harmonised: String representing the final standardized/harmonised keyword

Each original keyword index (0 to {len(unique_keywords)-1}) must appear exactly once across all original_index arrays.
Group similar/synonymous keywords together by including their indices in the same original_index array.

For example:
- If keywords at indices 0, 1, and 5 are variants of "machine learning", use: {{"original_index": [0, 1, 5], "harmonised": "machine learning"}}
- If keyword at index 3 stands alone as "neural networks", use: {{"original_index": [3], "harmonised": "neural networks"}}
- Every index from 0 to {len(unique_keywords)-1} must be included exactly once across all mappings

Focus on semantic similarity while preserving technical precision. Don't create higher-level categories - just clean up variants and synonyms.

CRITICAL: Your response must be valid JSON that strictly follows the required schema. No additional text, explanations, or deviations from the schema are allowed."""

    return prompt


def create_keyword_grouping_prompt(keywords: List[str]) -> str:
    """
    Create a prompt for identifying groups of keywords that should be harmonised together.
    
    Args:
        keywords: List of keywords to group
        
    Returns:
        Formatted prompt string
    """
    
    # Build instructions text
    instructions_text = "\n".join([f"- {instruction}" for instruction in HARMONISATION_INSTRUCTIONS])
    
    # Create indexed keywords list
    unique_keywords = sorted(set(keywords))
    keywords_text = "\n".join([f"{i}: {kw}" for i, kw in enumerate(unique_keywords)])
    
    prompt = f"""I have extracted {len(unique_keywords)} research keywords from grant data. Your task is to identify groups of keywords that should be harmonised together (variants, synonyms, and similar terms that represent the same concept).

**Instructions for grouping:**
{instructions_text}

**Keywords to group (with indices):**
{keywords_text}

IMPORTANT: You MUST respond with valid JSON only. Do not include explanations or any other text outside the JSON structure.

REQUIRED JSON SCHEMA - Follow this structure exactly:
{{
    "keyword_groups": [
        {{"indices": [0, 1, 5]}},
        {{"indices": [2, 8, 12]}},
        {{"indices": [3]}},
        ...
    ]
}}

The field is REQUIRED:
- keyword_groups: Array of group objects, each containing:
  - indices: Array of integers representing keyword indices that should be harmonised together

Rules for grouping:
- Each keyword index (0 to {len(unique_keywords)-1}) must appear exactly once across all groups
- Group keywords that are variants, synonyms, or represent the same concept
- Keywords that are unique and don't have variants should be in their own group with a single index
- Focus on semantic similarity while preserving technical precision
- Don't create higher-level categories - just identify which keywords represent the same underlying concept

For example:
- If keywords at indices 0, 1, and 5 are variants of the same concept, use: {{"indices": [0, 1, 5]}}
- If keyword at index 3 is unique, use: {{"indices": [3]}}
- Every index from 0 to {len(unique_keywords)-1} must be included exactly once across all groups

CRITICAL: Your response must be valid JSON that strictly follows the required schema. No additional text, explanations, or deviations from the schema are allowed."""

    return prompt


def create_group_harmonisation_prompt(keywords_with_indices: List[Tuple[int, str]]) -> str:
    """
    Create a prompt for harmonising a specific group of keywords.
    
    Args:
        keywords_with_indices: List of (index, keyword) tuples for the group
        
    Returns:
        Formatted prompt string
    """
    
    # Build instructions text
    instructions_text = "\n".join([f"- {instruction}" for instruction in HARMONISATION_INSTRUCTIONS])
    
    # Create keywords list for this group
    keywords_text = "\n".join([f"{idx}: {kw}" for idx, kw in keywords_with_indices])
    
    prompt = f"""You have been given a group of {len(keywords_with_indices)} research keywords that have been identified as variants, synonyms, or similar terms that should be harmonised into a single standardised keyword.

**Instructions for harmonisation:**
{instructions_text}

**Keywords in this group:**
{keywords_text}

IMPORTANT: You MUST respond with valid JSON only. Do not include explanations or any other text outside the JSON structure.

REQUIRED JSON SCHEMA - Follow this structure exactly:
{{
    "harmonised_keyword": "final standardised keyword"
}}

Choose the most appropriate, standardised, and widely-accepted term that best represents all the keywords in this group. Consider:
- Scientific precision and accuracy
- Common usage in research literature
- Technical specificity
- Domain conventions

CRITICAL: Your response must be valid JSON that strictly follows the required schema. No additional text, explanations, or deviations from the schema are allowed."""

    return prompt



@task
def identify() -> Task:
    """
    Inspect AI task for identifying groups of keywords that should be harmonised together.
    
    This is the first step of a two-step harmonisation process that identifies which keywords
    are variants, synonyms, or represent the same concept and should be grouped together.
    This approach helps avoid token limit issues by separating grouping from harmonisation.
    
    Returns:
        Configured Inspect AI Task
    """
    
    def dataset():
        all_keywords = load_extracted_keywords(logs_dir=LOGS_DIR)
        flat_keywords = [kw for sublist in all_keywords.values() for kw in sublist]

        prompt = create_keyword_grouping_prompt(flat_keywords)
        sample = Sample(
            input=prompt,
            metadata={
                "total_keywords": len(flat_keywords),
                "unique_keywords": len(set(flat_keywords)),
            }
        )
        
        return [sample]
    
    return Task(
        dataset=dataset(),
        solver=[
            system_message(HARMONISATION_SYSTEM_MESSAGE),
            generate(),
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="keyword_groups",
                json_schema=json_schema(KeywordGroupsOutput),
                strict=True
            )
        )
    )


@task
def harmonise() -> Task:
    """
    Inspect AI task for harmonising individual groups of keywords.
    
    This is the second step of a two-step harmonisation process that takes the groups
    identified by the identify task and determines the final harmonised keyword
    for each group. This can be run per group to avoid token limits.
    
    Returns:
        Configured Inspect AI Task
    """
    
    def dataset():
        # Load extracted keywords and identified groups
        all_keywords = load_extracted_keywords(logs_dir=LOGS_DIR)
        flat_keywords = [kw for sublist in all_keywords.values() for kw in sublist]
        unique_keywords = sorted(set(flat_keywords))
        
        keyword_groups = load_keyword_groups(logs_dir=LOGS_DIR)
        if not keyword_groups:
            raise ValueError("No keyword groups found. Run identify task first.")
        
        samples = []
        for group_idx, group_indices in enumerate(keyword_groups):
            # Get keywords for this group
            keywords_with_indices = [(idx, unique_keywords[idx]) for idx in group_indices 
                                   if 0 <= idx < len(unique_keywords)]
            
            if not keywords_with_indices:
                continue
                
            # Create prompt for this group
            prompt = create_group_harmonisation_prompt(keywords_with_indices)
            
            sample = Sample(
                input=prompt,
                metadata={
                    "group_index": group_idx,
                    "group_indices": group_indices,
                    "group_size": len(keywords_with_indices),
                },
                id=f"group_{group_idx}"
            )
            
            samples.append(sample)
        
        return samples
    
    return Task(
        dataset=dataset(),
        solver=[
            system_message(HARMONISATION_SYSTEM_MESSAGE),
            generate(),
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="group_harmonisation",
                json_schema=json_schema(GroupHarmonisationOutput),
                strict=True
            )
        )
    )

