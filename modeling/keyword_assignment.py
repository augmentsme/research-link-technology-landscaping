"""
Grant Keyword Assignment Module

This module provides functions to assign harmonised keywords to research grants based on
keyword extraction and harmonisation results from Inspect AI tasks.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

from inspect_ai.log import read_eval_log


@dataclass
class Grant:
    """Represents a research grant with its metadata."""
    id: str
    title: str
    summary: str
    funding_amount: Optional[float] = None
    funder: Optional[str] = None
    funding_scheme: Optional[str] = None
    status: Optional[str] = None
    
    @property
    def has_summary(self) -> bool:
        """Check if grant has a non-empty summary."""
        return bool(self.summary and self.summary.strip())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Grant':
        """Create Grant from dictionary data."""
        return cls(
            id=data.get('id', ''),
            title=data.get('title', ''),
            summary=data.get('grant_summary', ''),
            funding_amount=data.get('funding_amount', None),
            funder=data.get('funder', None),
            funding_scheme=data.get('funding_scheme', None),
            status=data.get('status', None)
        )


@dataclass
class GrantKeywordAssignment:
    """Represents the assignment of harmonised keywords to a grant."""
    grant_id: str
    grant_title: str
    grant_summary: str
    funding_amount: Optional[float]
    extracted_keywords: List[str]  # Raw keywords extracted from this grant
    harmonised_keywords: List[str]  # Harmonised keywords assigned to this grant
    keyword_mappings: Dict[str, str]  # Original keyword -> harmonised keyword mapping for this grant
    confidence_scores: Dict[str, float] = None  # Harmonised keyword -> confidence score


@dataclass
class KeywordAssignmentResult:
    """Result of the grant keyword assignment process."""
    assignments: List[GrantKeywordAssignment]
    keyword_grant_counts: Dict[str, int]  # Harmonised keyword -> number of grants
    keyword_funding_totals: Dict[str, float]  # Harmonised keyword -> total funding
    unassigned_grants: List[str]  # Grant IDs that couldn't be assigned keywords
    harmonisation_coverage: Dict[str, int]  # Original keyword -> number of grants using it
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'assignments': [asdict(assignment) for assignment in self.assignments],
            'keyword_grant_counts': self.keyword_grant_counts,
            'keyword_funding_totals': self.keyword_funding_totals,
            'unassigned_grants': self.unassigned_grants,
            'harmonisation_coverage': self.harmonisation_coverage,
            'summary': {
                'total_grants': len(self.assignments) + len(self.unassigned_grants),
                'assigned_grants': len(self.assignments),
                'unassigned_grants': len(self.unassigned_grants),
                'total_harmonised_keywords': len(self.keyword_grant_counts),
                'total_funding': sum(self.keyword_funding_totals.values())
            }
        }


def load_grants_data(grants_file: str) -> List[Grant]:
    """
    Load grants data from JSON file.
    
    Args:
        grants_file: Path to the grants JSON file
        
    Returns:
        List of Grant objects with summaries
    """
    logger = logging.getLogger(__name__)
    
    with open(grants_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    grants = []
    for grant_data in data:
        grant = Grant.from_dict(grant_data)
        if grant.has_summary:  # Only include grants with summaries
            grants.append(grant)
    
    logger.info(f"Loaded {len(grants)} grants with summaries from {grants_file}")
    return grants


def load_harmonised_keywords_from_eval_log(eval_file: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Load harmonised keywords from Inspect AI evaluation log file.
    
    Args:
        eval_file: Path to the harmonisation evaluation file
        
    Returns:
        Tuple of (harmonised_keywords, keyword_mappings)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading harmonised keywords from {eval_file}")
    
    harmonised_keywords = []
    keyword_mappings = {}
    
    try:
        log = read_eval_log(eval_file)
        
        for sample in log.samples:
            if sample.output and sample.output.completion:
                try:
                    harmonisation_result = json.loads(sample.output.completion)
                    
                    if 'harmonised_keywords' in harmonisation_result:
                        harmonised_keywords.extend(harmonisation_result['harmonised_keywords'])
                    
                    if 'keyword_mappings' in harmonisation_result:
                        keyword_mappings.update(harmonisation_result['keyword_mappings'])
                    
                    logger.info(f"Loaded {len(harmonised_keywords)} harmonised keywords")
                    break  # Take the first valid result
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Could not parse harmonisation result: {e}")
                    continue
    
    except Exception as e:
        logger.error(f"Error loading harmonised keywords: {e}")
        raise
    
    return harmonised_keywords, keyword_mappings


def load_grant_keywords_from_eval_log(eval_file: str) -> Dict[str, List[str]]:
    """
    Load grant keywords from keyword extraction evaluation log file.
    
    Args:
        eval_file: Path to the keyword extraction evaluation file
        
    Returns:
        Dictionary mapping grant IDs to their extracted keywords
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading grant keywords from {eval_file}")
    
    grant_keywords = {}
    
    try:
        log = read_eval_log(eval_file)
        
        for sample in log.samples:
            if sample.output and sample.output.completion:
                try:
                    # Extract grant ID from sample metadata or ID
                    grant_id = sample.id or sample.metadata.get('grant_id', '')
                    
                    keywords_result = json.loads(sample.output.completion)
                    
                    # Collect all keywords from different categories
                    all_keywords = []
                    if 'keywords' in keywords_result:
                        all_keywords.extend(keywords_result['keywords'])
                    if 'methodology_keywords' in keywords_result:
                        all_keywords.extend(keywords_result['methodology_keywords'])
                    if 'application_keywords' in keywords_result:
                        all_keywords.extend(keywords_result['application_keywords'])
                    if 'technology_keywords' in keywords_result:
                        all_keywords.extend(keywords_result['technology_keywords'])
                    
                    grant_keywords[grant_id] = all_keywords
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Could not parse keywords for grant {sample.id}: {e}")
                    continue
    
    except Exception as e:
        logger.error(f"Error loading grant keywords: {e}")
        raise
    
    logger.info(f"Loaded keywords for {len(grant_keywords)} grants")
    return grant_keywords


def assign_harmonised_keywords_to_grant(
    grant: Grant,
    grant_keywords: List[str],
    harmonised_keywords: List[str],
    keyword_mappings: Dict[str, str]
) -> Optional[GrantKeywordAssignment]:
    """
    Assign harmonised keywords to a single grant based on its extracted keywords.
    
    Args:
        grant: Grant object
        grant_keywords: Keywords extracted from this grant
        harmonised_keywords: List of all harmonised keywords
        keyword_mappings: Mapping from original to harmonised keywords
        
    Returns:
        GrantKeywordAssignment or None if no keywords can be assigned
    """
    if not grant_keywords:
        return None
    
    # Map extracted keywords to harmonised keywords
    assigned_harmonised = []
    grant_keyword_mappings = {}
    
    for keyword in grant_keywords:
        harmonised_keyword = keyword_mappings.get(keyword.lower(), keyword)
        if harmonised_keyword in harmonised_keywords:
            assigned_harmonised.append(harmonised_keyword)
            grant_keyword_mappings[keyword] = harmonised_keyword
    
    # Remove duplicates while preserving order
    unique_harmonised = []
    seen = set()
    for kw in assigned_harmonised:
        if kw not in seen:
            unique_harmonised.append(kw)
            seen.add(kw)
    
    if not unique_harmonised:
        return None
    
    return GrantKeywordAssignment(
        grant_id=grant.id,
        grant_title=grant.title,
        grant_summary=grant.summary,
        funding_amount=grant.funding_amount,
        extracted_keywords=grant_keywords,
        harmonised_keywords=unique_harmonised,
        keyword_mappings=grant_keyword_mappings
    )


def assign_keywords_to_grants(
    grants: List[Grant],
    grant_keywords: Dict[str, List[str]],
    harmonised_keywords: List[str],
    keyword_mappings: Dict[str, str]
) -> KeywordAssignmentResult:
    """
    Assign harmonised keywords to all grants.
    
    Args:
        grants: List of grants
        grant_keywords: Dictionary mapping grant IDs to their keywords
        harmonised_keywords: List of harmonised keywords
        keyword_mappings: Mapping from original to harmonised keywords
        
    Returns:
        KeywordAssignmentResult with assignment information
    """
    logger = logging.getLogger(__name__)
    
    assignments = []
    unassigned_grants = []
    keyword_grant_counts = defaultdict(int)
    keyword_funding_totals = defaultdict(float)
    harmonisation_coverage = defaultdict(int)
    
    for grant in grants:
        grant_kw = grant_keywords.get(grant.id, [])
        
        assignment = assign_harmonised_keywords_to_grant(
            grant, grant_kw, harmonised_keywords, keyword_mappings
        )
        
        if assignment:
            assignments.append(assignment)
            
            # Update statistics
            for harmonised_kw in assignment.harmonised_keywords:
                keyword_grant_counts[harmonised_kw] += 1
                if grant.funding_amount:
                    keyword_funding_totals[harmonised_kw] += grant.funding_amount
            
            for original_kw in assignment.extracted_keywords:
                harmonisation_coverage[original_kw] += 1
        else:
            unassigned_grants.append(grant.id)
    
    logger.info(f"Assigned keywords to {len(assignments)} grants, {len(unassigned_grants)} unassigned")
    
    return KeywordAssignmentResult(
        assignments=assignments,
        keyword_grant_counts=dict(keyword_grant_counts),
        keyword_funding_totals=dict(keyword_funding_totals),
        unassigned_grants=unassigned_grants,
        harmonisation_coverage=dict(harmonisation_coverage)
    )


def run_keyword_assignment_pipeline(
    grants_file: str,
    keyword_extraction_logs_dir: str,
    harmonisation_logs_dir: str,
    output_file: str
) -> KeywordAssignmentResult:
    """
    Run the complete keyword assignment pipeline.
    
    Args:
        grants_file: Path to grants JSON file
        keyword_extraction_logs_dir: Directory with keyword extraction evaluation logs
        harmonisation_logs_dir: Directory with harmonisation evaluation logs
        output_file: Output file for results
        
    Returns:
        KeywordAssignmentResult
    """
    logger = logging.getLogger(__name__)
    
    # Load data
    logger.info("Loading grants data...")
    grants = load_grants_data(grants_file)
    
    # Find evaluation files
    def find_latest_eval_file(logs_dir: str, pattern: str) -> Optional[str]:
        logs_path = Path(logs_dir)
        eval_files = list(logs_path.glob(pattern))
        if not eval_files:
            return None
        return str(max(eval_files, key=lambda x: x.stat().st_mtime))
    
    # Load keyword extraction results
    keyword_eval_file = find_latest_eval_file(keyword_extraction_logs_dir, "*extract*keywords*.eval")
    if not keyword_eval_file:
        raise FileNotFoundError("No keyword extraction evaluation files found")
    
    logger.info("Loading grant keywords...")
    grant_keywords = load_grant_keywords_from_eval_log(keyword_eval_file)
    
    # Load harmonisation results
    harmonisation_eval_file = find_latest_eval_file(harmonisation_logs_dir, "*harmon*keywords*.eval")
    if not harmonisation_eval_file:
        raise FileNotFoundError("No harmonisation evaluation files found")
    
    logger.info("Loading harmonised keywords...")
    harmonised_keywords, keyword_mappings = load_harmonised_keywords_from_eval_log(harmonisation_eval_file)
    
    # Assign keywords to grants
    logger.info("Assigning keywords to grants...")
    result = assign_keywords_to_grants(
        grants, grant_keywords, harmonised_keywords, keyword_mappings
    )
    
    # Save results
    logger.info(f"Saving results to {output_file}...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    
    logger.info("Keyword assignment pipeline completed successfully")
    return result
