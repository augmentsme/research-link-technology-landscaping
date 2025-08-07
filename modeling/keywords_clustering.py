"""
Keywords Clustering Task for Research Grants

This module implements an Inspect AI task that clusters and harmonizes keywords 
extracted from research grants into coherent topics.
"""

from typing import List, Dict
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.solver import system_message, generate
from inspect_ai.util import json_schema
from pydantic import BaseModel, Field

from modeling.ioutils import collect_keywords_from_evaluation_results, TopicSchema


class KeywordsClusteringOutput(BaseModel):
    """Pydantic model for structured keywords clustering output."""
    topics: List[TopicSchema] = Field(description="List of identified research topics with their keywords")
    unassigned_keywords: List[str] = Field(description="Keywords that don't fit well into any topic")
    topic_relationships: Dict[str, List[str]] = Field(description="Relationships between topics", default_factory=dict)
    keyword_to_topic_mapping: Dict[str, str] = Field(description="Direct mapping from each keyword to its assigned topic", default_factory=dict)


@task
def cluster_research_keywords(
    data_dir: str = "/home/lcheng/oz318/research-link-technology-landscaping/data",
    output_file: str = "keyword_clusters.json"
) -> Task:
    """
    Inspect AI task for harmonizing and clustering research keywords into emergent research areas.
    
    This creates a single comprehensive query with ALL keywords extracted from the previous step,
    allowing the LLM to harmonize similar terms and identify specific emergent research areas
    while preserving the technical specificity and cutting-edge nature of the research.
    
    Args:
        data_dir: Directory containing the data files
        output_file: Output file for saving clustering results
        
    Returns:
        Configured Inspect AI Task
    """
    
    def dataset():
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
        
        # Create a comprehensive prompt with all keywords
        keywords_list_str = ", ".join(f'"{kw}"' for kw in unique_keywords)
        
        prompt = f"""You are an expert research analyst tasked with harmonizing and organizing research keywords to identify emergent research areas and trends.

You have been given {len(unique_keywords)} unique keywords extracted from research grants across multiple domains:

{keywords_list_str}

Your task is to:
1. Harmonize similar keywords by grouping variants of the same concept (e.g., "AI", "artificial intelligence", "machine intelligence")
2. Identify emergent and specific research areas rather than broad traditional disciplines
3. Preserve the specificity and novelty of the original keywords - avoid overly generic topic names
4. Create 12-20 focused research clusters that capture emerging trends and specific methodologies
5. Maintain the granularity that reflects current research directions and innovations
6. Create a complete mapping from each keyword to its harmonized cluster

Guidelines for keyword harmonization:
- Group spelling variants, synonyms, and different expressions of the same concept
- Preserve technical specificity - prefer "deep reinforcement learning" over "machine learning"
- Focus on emergent research areas rather than traditional broad disciplines
- Use the most representative or specific term from the cluster as the topic name
- Keep clusters focused and specific - avoid merging distinct research areas
- Prefer compound/specific terms over single broad words (e.g., "quantum computing" not "computing")
- If keywords represent truly distinct concepts, keep them separate even if clusters are small
- Maintain the cutting-edge nature of the research - capture what's new and emerging

Examples of good harmonization:
- "deep learning", "neural networks", "artificial neural networks" → "Deep Learning and Neural Networks"
- "CRISPR", "gene editing", "genome editing" → "CRISPR and Gene Editing"
- "quantum computing", "quantum algorithms", "quantum information" → "Quantum Computing"

The goal is to identify what research areas are actively emerging and being funded, not to categorize into traditional academic disciplines. Preserve the specificity that shows current research trends and innovations.

Original keyword categories for context: {category_info}"""
        
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
    
    return Task(
        dataset=dataset(),
        solver=[
            system_message("""You are an expert research analyst specializing in keyword harmonization and emerging research trend identification.

Your goal is to harmonize research keywords while preserving their specificity and identifying emergent research areas that will be used for:
1. Tracking emerging research trends and innovations
2. Identifying cutting-edge research directions receiving funding
3. Supporting discovery of novel research areas and methodologies
4. Creating precise linkages between related research projects

You must provide:
- Specific, technical topic names that reflect current research directions (not broad academic disciplines)
- Harmonization of keyword variants while preserving technical precision
- Complete keyword assignments that maintain the granularity of emerging research
- A direct keyword-to-topic mapping for all assigned keywords
- Representative primary keywords that capture the most specific/emerging terms

Focus on creating harmonized clusters that are:
- Technically precise and reflect current research language
- Specific enough to capture emerging trends and methodologies
- Preserving the cutting-edge nature of funded research
- Focused on what's new and innovative rather than traditional categories
- Maintaining sufficient granularity to distinguish between different research approaches

Avoid overly broad categorizations - the goal is to see what specific research areas are being actively funded and developed."""),
            generate()
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="keywords_clustering",
                json_schema=json_schema(KeywordsClusteringOutput)
            )
        )
    )