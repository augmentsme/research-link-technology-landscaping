"""
Grant Topic Classification Module

This module provides functions to classify research grants into topics based on
keyword extraction and clustering results from Inspect AI tasks.
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
class GrantTopicAssignment:
    """Represents the assignment of a grant to one or more topics."""
    grant_id: str
    grant_title: str
    grant_summary: str
    funding_amount: Optional[float]
    assigned_topics: List[str]  # Topic names
    topic_scores: Dict[str, float]  # Topic name -> relevance score
    extracted_keywords: List[str]  # Keywords extracted from this grant
    matched_keywords: Dict[str, List[str]]  # Topic name -> matched keywords


@dataclass
class TopicAssignmentResult:
    """Result of the grant topic assignment process."""
    assignments: List[GrantTopicAssignment]
    topic_grant_counts: Dict[str, int]  # Topic name -> number of grants
    topic_funding_totals: Dict[str, float]  # Topic name -> total funding
    unassigned_grants: List[str]  # Grant IDs that couldn't be assigned
    keyword_coverage: Dict[str, int]  # Keyword -> number of grants using it
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'assignments': [asdict(assignment) for assignment in self.assignments],
            'topic_grant_counts': self.topic_grant_counts,
            'topic_funding_totals': self.topic_funding_totals,
            'unassigned_grants': self.unassigned_grants,
            'keyword_coverage': self.keyword_coverage,
            'summary': {
                'total_grants': len(self.assignments) + len(self.unassigned_grants),
                'assigned_grants': len(self.assignments),
                'unassigned_grants': len(self.unassigned_grants),
                'total_topics': len(self.topic_grant_counts),
                'total_funding': sum(self.topic_funding_totals.values())
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
        grants_data = json.load(f)
    
    grants = []
    for grant_data in grants_data:
        grant = Grant.from_dict(grant_data)
        if grant.has_summary:
            grants.append(grant)
        else:
            logger.debug(f"Skipping grant {grant.id} - no summary")
    
    logger.info(f"Loaded {len(grants)} grants with summaries from {grants_file}")
    return grants


def load_topic_clusters_from_eval_log(eval_file: str) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    """
    Load topic clusters from Inspect AI evaluation log file.
    
    Args:
        eval_file: Path to the clustering evaluation file
        
    Returns:
        Tuple of (topic_clusters, keyword_to_topic_mapping)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading topic clusters from {eval_file}")
    
    topic_clusters = {}
    keyword_to_topic = {}
    
    try:
        log = read_eval_log(eval_file)
        
        for sample in log.samples:
            if sample.output and sample.output.completion:
                try:
                    clustering_result = json.loads(sample.output.completion)
                    
                    if 'topics' in clustering_result:
                        for topic_data in clustering_result['topics']:
                            topic_name = topic_data.get('topic_name', '')
                            keywords = topic_data.get('keywords', [])
                            
                            topic_clusters[topic_name] = {
                                'description': topic_data.get('description', ''),
                                'keywords': keywords,
                                'primary_keywords': topic_data.get('primary_keywords', []),
                                'domain': topic_data.get('domain', None)
                            }
                            
                            # Build keyword-to-topic mapping
                            for keyword in keywords:
                                keyword_to_topic[keyword.lower()] = topic_name
                    
                    # Also handle direct keyword_to_topic_mapping if available
                    if 'keyword_to_topic_mapping' in clustering_result:
                        for keyword, topic in clustering_result['keyword_to_topic_mapping'].items():
                            keyword_to_topic[keyword.lower()] = topic
                    
                    logger.info(f"Loaded {len(topic_clusters)} topic clusters")
                    break  # Take the first valid result
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Could not parse clustering result: {e}")
                    continue
    
    except Exception as e:
        logger.error(f"Error loading topic clusters: {e}")
        raise
    
    return topic_clusters, keyword_to_topic


def load_grant_keywords_from_eval_log(eval_file: str) -> Dict[str, List[str]]:
    """
    Load grant keywords from keyword extraction evaluation log file.
    
    Args:
        eval_file: Path to the keyword extraction evaluation file
        
    Returns:
        Dictionary mapping grant_id to list of keywords
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading grant keywords from {eval_file}")
    
    grant_keywords = {}
    
    try:
        log = read_eval_log(eval_file)
        
        for sample in log.samples:
            if sample.output and sample.output.completion:
                try:
                    keywords_result = json.loads(sample.output.completion)
                    
                    # Extract grant ID from metadata
                    grant_id = sample.metadata.get('grant_id', '') if sample.metadata else ''
                    if not grant_id:
                        continue
                    
                    # Collect all keywords from different categories
                    all_keywords = []
                    
                    if isinstance(keywords_result, dict):
                        for key in ['keywords', 'methodology_keywords', 'application_keywords', 'technology_keywords']:
                            if key in keywords_result and isinstance(keywords_result[key], list):
                                all_keywords.extend(keywords_result[key])
                    
                    # Clean and normalize keywords
                    clean_keywords = []
                    for kw in all_keywords:
                        if kw and isinstance(kw, str):
                            clean_kw = kw.strip().lower()
                            if clean_kw and len(clean_kw) > 2:
                                clean_keywords.append(clean_kw)
                    
                    grant_keywords[grant_id] = clean_keywords
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Could not parse keywords result for sample {sample.id}: {e}")
                    continue
        
        logger.info(f"Loaded keywords for {len(grant_keywords)} grants")
        
    except Exception as e:
        logger.error(f"Error loading grant keywords: {e}")
        raise
    
    return grant_keywords


def find_latest_eval_file(logs_dir: str, pattern: str) -> Optional[str]:
    """
    Find the most recent evaluation file matching the pattern.
    
    Args:
        logs_dir: Directory containing log files
        pattern: Glob pattern to match files
        
    Returns:
        Path to the most recent file or None if no files found
    """
    logs_path = Path(logs_dir)
    files = list(logs_path.glob(pattern))
    
    if not files:
        return None
    
    # Return the most recent file
    return str(max(files, key=lambda x: x.stat().st_mtime))


def calculate_topic_scores(
    grant_keywords: List[str], 
    topic_clusters: Dict[str, Dict], 
    keyword_to_topic: Dict[str, str]
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """
    Calculate topic scores for a grant based on keyword matches.
    
    Args:
        grant_keywords: List of keywords extracted from the grant
        topic_clusters: Dictionary of topic cluster information
        keyword_to_topic: Mapping from keywords to topic names
        
    Returns:
        Tuple of (topic_scores, matched_keywords_by_topic)
    """
    topic_scores = {}
    matched_keywords_by_topic = defaultdict(list)
    
    # Find keyword matches
    for keyword in grant_keywords:
        if keyword in keyword_to_topic:
            topic_name = keyword_to_topic[keyword]
            matched_keywords_by_topic[topic_name].append(keyword)
    
    # Calculate scores for each topic based on keyword matches
    for topic_name, matched_keywords in matched_keywords_by_topic.items():
        if topic_name in topic_clusters:
            # Score based on:
            # 1. Number of matched keywords
            # 2. Primary keywords get higher weight
            
            base_score = len(matched_keywords)
            
            # Primary keywords get bonus points
            primary_keywords = topic_clusters[topic_name].get('primary_keywords', [])
            primary_keyword_matches = sum(1 for kw in matched_keywords 
                                        if kw in [pk.lower() for pk in primary_keywords])
            primary_keyword_bonus = primary_keyword_matches * 0.5
            
            topic_scores[topic_name] = base_score + primary_keyword_bonus
    
    return topic_scores, dict(matched_keywords_by_topic)


def assign_grant_to_topics(
    grant: Grant,
    grant_keywords: List[str],
    topic_clusters: Dict[str, Dict],
    keyword_to_topic: Dict[str, str],
    min_keyword_matches: int = 2
) -> Optional[GrantTopicAssignment]:
    """
    Assign a single grant to topics based on keyword matches.
    
    Args:
        grant: Grant object
        grant_keywords: Keywords extracted from the grant
        topic_clusters: Dictionary of topic cluster information
        keyword_to_topic: Mapping from keywords to topic names
        min_keyword_matches: Minimum number of keyword matches required
        
    Returns:
        GrantTopicAssignment if successful, None if unassigned
    """
    if not grant_keywords:
        return None
    
    topic_scores, matched_keywords_by_topic = calculate_topic_scores(
        grant_keywords, topic_clusters, keyword_to_topic
    )
    
    # Filter topics by minimum keyword matches
    qualified_topics = {}
    qualified_matches = {}
    
    for topic_name, score in topic_scores.items():
        matched_keywords = matched_keywords_by_topic.get(topic_name, [])
        if len(matched_keywords) >= min_keyword_matches:
            qualified_topics[topic_name] = score
            qualified_matches[topic_name] = matched_keywords
    
    if not qualified_topics:
        return None
    
    return GrantTopicAssignment(
        grant_id=grant.id,
        grant_title=grant.title,
        grant_summary=grant.summary,
        funding_amount=grant.funding_amount,
        assigned_topics=list(qualified_topics.keys()),
        topic_scores=qualified_topics,
        extracted_keywords=grant_keywords,
        matched_keywords=qualified_matches
    )


def assign_grants_to_topics(
    grants: List[Grant],
    grant_keywords: Dict[str, List[str]],
    topic_clusters: Dict[str, Dict],
    keyword_to_topic: Dict[str, str],
    min_keyword_matches: int = 2
) -> TopicAssignmentResult:
    """
    Assign all grants to topics based on keyword matches.
    
    Args:
        grants: List of Grant objects
        grant_keywords: Dictionary mapping grant_id to keywords
        topic_clusters: Dictionary of topic cluster information
        keyword_to_topic: Mapping from keywords to topic names
        min_keyword_matches: Minimum number of keyword matches required
        
    Returns:
        TopicAssignmentResult with all assignments
    """
    logger = logging.getLogger(__name__)
    
    assignments = []
    topic_grant_counts = defaultdict(int)
    topic_funding_totals = defaultdict(float)
    unassigned_grants = []
    keyword_coverage = defaultdict(int)
    
    logger.info(f"Assigning {len(grants)} grants to {len(topic_clusters)} topics")
    
    for grant in grants:
        grant_kw = grant_keywords.get(grant.id, [])
        
        assignment = assign_grant_to_topics(
            grant, grant_kw, topic_clusters, keyword_to_topic,
            min_keyword_matches
        )
        
        if assignment:
            assignments.append(assignment)
            
            # Update statistics
            for topic in assignment.assigned_topics:
                topic_grant_counts[topic] += 1
                if grant.funding_amount:
                    topic_funding_totals[topic] += grant.funding_amount
            
            # Update keyword coverage
            for keyword in assignment.extracted_keywords:
                keyword_coverage[keyword] += 1
        else:
            unassigned_grants.append(grant.id)
    
    result = TopicAssignmentResult(
        assignments=assignments,
        topic_grant_counts=dict(topic_grant_counts),
        topic_funding_totals=dict(topic_funding_totals),
        unassigned_grants=unassigned_grants,
        keyword_coverage=dict(keyword_coverage)
    )
    
    logger.info(f"Assignment complete: {len(assignments)} grants assigned, {len(unassigned_grants)} unassigned")
    
    return result


def save_topic_assignment_results(result: TopicAssignmentResult, output_file: str) -> None:
    """
    Save topic assignment results to JSON file.
    
    Args:
        result: TopicAssignmentResult to save
        output_file: Path to output file
    """
    logger = logging.getLogger(__name__)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    
    logger.info(f"Topic assignment results saved to {output_file}")


def classify_grants_by_topics(
    grants_file: str,
    logs_dir: str = "/home/lcheng/oz318/research-link-technology-landscaping/logs",
    output_file: str = "/home/lcheng/oz318/research-link-technology-landscaping/data/grant_topic_assignments.json",
    min_keyword_matches: int = 2,
    clustering_eval_file: Optional[str] = None,
    keywords_eval_file: Optional[str] = None
) -> TopicAssignmentResult:
    """
    Main function to classify grants by topics using results from keyword extraction and clustering.
    
    Args:
        grants_file: Path to the grants JSON file
        logs_dir: Directory containing evaluation log files
        output_file: Path to save results
        min_keyword_matches: Minimum number of keyword matches required for assignment
        clustering_eval_file: Specific clustering evaluation file (auto-detect if None)
        keywords_eval_file: Specific keywords evaluation file (auto-detect if None)
        
    Returns:
        TopicAssignmentResult
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting grant topic classification...")
    
    # Load grants data
    logger.info("Loading grants data...")
    grants = load_grants_data(grants_file)
    
    # Find evaluation files if not specified
    if clustering_eval_file is None:
        clustering_eval_file = find_latest_eval_file(logs_dir, "*cluster*keywords*.eval")
        if not clustering_eval_file:
            raise FileNotFoundError("No keyword clustering evaluation files found")
    
    if keywords_eval_file is None:
        keywords_eval_file = find_latest_eval_file(logs_dir, "*extract*keywords*.eval")
        if not keywords_eval_file:
            raise FileNotFoundError("No keyword extraction evaluation files found")
    
    # Load topic clusters
    logger.info("Loading topic clusters...")
    topic_clusters, keyword_to_topic = load_topic_clusters_from_eval_log(clustering_eval_file)
    
    # Load grant keywords
    logger.info("Loading grant keywords...")
    grant_keywords = load_grant_keywords_from_eval_log(keywords_eval_file)
    
    # Assign grants to topics
    logger.info("Assigning grants to topics...")
    result = assign_grants_to_topics(
        grants, grant_keywords, topic_clusters, keyword_to_topic,
        min_keyword_matches
    )
    
    # Save results
    save_topic_assignment_results(result, output_file)
    
    # Print summary
    summary = result.to_dict()['summary']
    logger.info("\nClassification Summary:")
    logger.info(f"Total grants processed: {summary['total_grants']}")
    logger.info(f"Grants assigned: {summary['assigned_grants']}")
    logger.info(f"Grants unassigned: {summary['unassigned_grants']}")
    logger.info(f"Topics with grants: {summary['total_topics']}")
    
    # Top topics by grant count
    top_topics = sorted(result.topic_grant_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("\nTop 10 Topics by Grant Count:")
    for topic, count in top_topics:
        funding = result.topic_funding_totals.get(topic, 0)
        logger.info(f"  {topic}: {count} grants, ${funding:,.0f}")
    
    return result


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run classification with default parameters
    classify_grants_by_topics(
        grants_file="/home/lcheng/oz318/research-link-technology-landscaping/data/active_grants.json"
    )
