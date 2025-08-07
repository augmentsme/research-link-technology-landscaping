"""
I/O Utilities for Research Grant Analysis

This module contains utility functions for data loading, saving, cleaning, and converting
used across the modeling pipeline for research grant analysis.
"""

import json
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from inspect_ai.log import read_eval_log
from pydantic import BaseModel


@dataclass
class HarmonisedKeywordGroup:
    """Represents a group of keywords that have been harmonised to a single term."""
    harmonised_keyword: str
    original_keywords: List[str]  # Keywords that map to this harmonised form
    
    
@dataclass  
class KeywordHarmonisationResult:
    """Result of the keywords harmonisation process."""
    harmonised_keywords: List[str]
    keyword_mappings: Dict[str, str]  # original -> harmonised
    merged_groups: Dict[str, List[str]]  # harmonised -> list of original keywords
    unchanged_keywords: List[str]
    total_original_keywords: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'harmonised_keywords': self.harmonised_keywords,
            'keyword_mappings': self.keyword_mappings,
            'merged_groups': self.merged_groups,
            'unchanged_keywords': self.unchanged_keywords,
            'total_original_keywords': self.total_original_keywords
        }


# Import the harmonisation output model from keywords_harmonisation.py
# This allows other modules to access it through ioutils
try:
    from modeling.keywords_harmonisation import KeywordsHarmonisationOutput
except ImportError:
    # Fallback definition if import fails
    class KeywordsHarmonisationOutput(BaseModel):
        """Pydantic model for structured keywords harmonisation output."""
        harmonised_keywords: List[str]
        keyword_mappings: Dict[str, str]
        merged_groups: Dict[str, List[str]]
        unchanged_keywords: List[str]


def collect_keywords_from_evaluation_results(eval_files: List[str] = None) -> Dict[str, List[str]]:
    """
    Collect keywords from Inspect AI evaluation result files.
    
    Args:
        eval_files: List of evaluation file paths. If None, searches for keyword extraction results.
        
    Returns:
        Dictionary with categories as keys and lists of keywords as values
    """
    logger = logging.getLogger(__name__)
    
    if eval_files is None:
        # Look for keyword extraction evaluation files
        logs_dir = Path("/home/lcheng/oz318/research-link-technology-landscaping/logs")
        eval_files = list(logs_dir.glob("*extract*keywords*.eval"))
    
    all_keywords = defaultdict(list)
    
    for eval_file in eval_files:
        try:
            logger.info(f"Loading keywords from {eval_file}")
            log = read_eval_log(str(eval_file))
            
            for sample in log.samples:
                if sample.output and sample.output.completion:
                    try:
                        # Parse the structured output
                        from modeling.keywords_extraction import KeywordsExtractionOutput
                        keywords_data = KeywordsExtractionOutput.model_validate_json(sample.output.completion)
                        
                        # Collect keywords by category
                        all_keywords['keywords'].extend(keywords_data.keywords)
                        all_keywords['methodology_keywords'].extend(keywords_data.methodology_keywords)
                        all_keywords['application_keywords'].extend(keywords_data.application_keywords)
                        all_keywords['technology_keywords'].extend(keywords_data.technology_keywords)
                    
                    except Exception as e:
                        logger.warning(f"Could not parse keywords from sample {sample.id}: {e}")
                        continue

        except Exception as e:
            logger.warning(f"Could not load evaluation file {eval_file}: {e}")
            continue
    
    # Clean and deduplicate keywords
    for category in all_keywords:
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in all_keywords[category]:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        # Filter out very short or generic keywords
        all_keywords[category] = [
            kw for kw in unique_keywords 
            if len(kw) > 2 and kw.lower() not in ['the', 'and', 'for', 'with', 'from', 'research', 'study', 'analysis']
        ]
    
    # Log summary
    total_keywords = sum(len(keywords) for keywords in all_keywords.values())
    logger.info(f"Collected {total_keywords} total keywords across {len(all_keywords)} categories")
    for category, keywords in all_keywords.items():
        if keywords:
            logger.info(f"  {category}: {len(keywords)} keywords")
    
    return dict(all_keywords)


def process_harmonisation_result(harmonisation_output: KeywordsHarmonisationOutput) -> KeywordHarmonisationResult:
    """
    Convert KeywordsHarmonisationOutput from LLM to KeywordHarmonisationResult for use in applications.
    
    Args:
        harmonisation_output: Output from the LLM harmonisation task
        
    Returns:
        KeywordHarmonisationResult object with processed harmonisation data
    """
    
    return KeywordHarmonisationResult(
        harmonised_keywords=harmonisation_output.harmonised_keywords,
        keyword_mappings=harmonisation_output.keyword_mappings,
        merged_groups=harmonisation_output.merged_groups,
        unchanged_keywords=harmonisation_output.unchanged_keywords,
        total_original_keywords=len(harmonisation_output.keyword_mappings)
    )


def load_harmonisation_results_from_eval(eval_file: str) -> KeywordHarmonisationResult:
    """
    Load harmonisation results from an Inspect AI evaluation log file.
    
    Args:
        eval_file: Path to the evaluation log file
        
    Returns:
        KeywordHarmonisationResult object with the harmonisation output
    """
    logger = logging.getLogger(__name__)
    
    try:
        log = read_eval_log(eval_file)
        
        if not log.samples:
            raise ValueError("No samples found in evaluation log")
        
        sample = log.samples[0]  # Should be only one sample
        
        if not sample.output or not sample.output.completion:
            raise ValueError("No output found in evaluation sample")
        
        # Parse the structured output
        harmonisation_output = KeywordsHarmonisationOutput.model_validate_json(sample.output.completion)
        
        # Convert to KeywordHarmonisationResult
        result = process_harmonisation_result(harmonisation_output)
        
        logger.info(f"Loaded harmonisation results: {len(result.harmonised_keywords)} harmonised keywords from {result.total_original_keywords} original keywords")
        
        return result
        
    except Exception as e:
        logger.error(f"Error loading harmonisation results from {eval_file}: {e}")
        raise


def save_harmonisation_results(result: KeywordHarmonisationResult, output_file: str):
    """
    Save harmonisation results to JSON file.
    
    Args:
        result: KeywordHarmonisationResult to save
        output_file: Output file path
    """
    logger = logging.getLogger(__name__)
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    
    logger.info(f"Harmonisation results saved to: {output_path}")
