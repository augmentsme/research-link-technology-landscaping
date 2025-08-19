"""
Harmonization module for keyword extraction results using lexical analysis.

This module provides functionality to harmonize keywords extracted from the extract task
by identifying semantic similarities, clustering related terms, and creating a unified
vocabulary through lexical analysis techniques.
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union, Any, Optional
from dataclasses import dataclass

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import numpy as np
from difflib import SequenceMatcher

# Inspect AI imports for the review task
from inspect_ai import Task, task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.solver import system_message, generate
from inspect_ai.util import json_schema
from inspect_ai.hooks import Hooks, SampleEnd, hooks, TaskEnd, TaskStart
from inspect_ai.scorer import scorer, Score, Target
from inspect_ai.solver import TaskState
from pydantic import BaseModel, Field

from config import KEYWORDS_PATH, RESULTS_DIR, TERMS_PATH, PROMPTS_DIR, REVIEW_FILE


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


@dataclass
class KeywordCluster:
    """Represents a cluster of harmonized keywords."""
    canonical_term: str
    variants: List[str]
    frequency: int
    categories: Set[str]


class ClusterReview(BaseModel):
    """Pydantic model for cluster review decisions."""
    model_config = {"extra": "forbid"}
    
    cluster_id: int = Field(description="Index of the cluster being reviewed")
    canonical_term: str = Field(description="The canonical term of the cluster")
    variants: List[str] = Field(description="List of variant terms in the cluster")
    decision: str = Field(description="Decision: 'accept' or 'reject'")
    reasoning: str = Field(description="Explanation for the decision")


class HarmonizationReviewOutput(BaseModel):
    """Output model for harmonization review task."""
    model_config = {"extra": "forbid"}
    
    reviews: List[ClusterReview] = Field(description="List of cluster review decisions")


@scorer(metrics=[])
def harmonization_reviewer():
    """
    Scorer that validates harmonization review decisions.
    """
    
    async def score(state: TaskState, target: Target) -> Score:
        """Score based on review decision quality."""
        
        if not state.output or not state.output.completion:
            return Score(
                value="incorrect", 
                explanation="No harmonization review output to score"
            )
        
        try:
            result = json.loads(state.output.completion)
            reviews = result.get("reviews", [])
            
            total_reviews = len(reviews)
            decisions = Counter(review.get("decision", "unknown") for review in reviews)
            
            explanation = f"Reviewed {total_reviews} clusters: "
            explanation += ", ".join([f"{decision}: {count}" for decision, count in decisions.items()])
            
            return Score(
                value=total_reviews,
                explanation=explanation
            )
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return Score(
                value="incorrect", 
                explanation=f"Error parsing harmonization review result: {str(e)}"
            )
    
    return score


@hooks(name="HarmonizationReviewHook", description="Hook to save harmonization review results and apply refinements")
class HarmonizationReviewHook(Hooks):
    """Hook to save harmonization review results and automatically apply refinements."""

    async def on_sample_end(self, data: SampleEnd) -> None:
        """Save harmonization review results."""
        try:
            output_text = data.sample.output.completion
            result_json = json.loads(output_text)
            
            with open(REVIEW_FILE, 'w', encoding='utf-8') as f:
                json.dump(result_json, f, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # Silently handle errors
    
    async def on_task_start(self, data: TaskStart) -> None:
        """Run harmonization automatically before the review task starts."""
        try:
            # Run harmonization directly
            harmonize_extracted_keywords(save_clusters=True)
        except Exception:
            pass  # Silently handle errors
    
    async def on_task_end(self, data: TaskEnd) -> None:
        """Automatically apply harmonization review decisions."""
        try:
            apply_harmonization_review()
        except Exception:
            pass  # Silently handle errors


class LexicalHarmonizer:
    """
    Harmonizes extracted keywords using lexical analysis techniques.
    
    This class provides methods to:
    - Normalize keywords through stemming and lemmatization
    - Identify semantic similarities using various text similarity metrics
    - Cluster related keywords
    - Generate canonical terms for keyword groups
    """
    
    def __init__(self, similarity_threshold: float = 0.7, min_cluster_size: int = 2):
        """
        Initialize the lexical harmonizer.
        
        Args:
            similarity_threshold: Minimum similarity score for clustering keywords
            min_cluster_size: Minimum number of keywords required to form a cluster
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        
        # Initialize NLP tools
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Keyword storage
        self.raw_keywords = []
        self.normalized_keywords = {}
        self.clusters = []
        
    def load_keywords(self, keywords_file: Union[str, Path] = None) -> None:
        """
        Load keywords from the extract task output file.
        
        Args:
            keywords_file: Path to keywords JSON file. If None, uses default from config.
        """
        if keywords_file is None:
            keywords_file = KEYWORDS_PATH
            
        with open(keywords_file, 'r', encoding='utf-8') as f:
            self.raw_keywords = json.load(f)
    
    def normalize_keyword(self, keyword: str) -> str:
        """
        Normalize a keyword through preprocessing steps.
        
        Args:
            keyword: Raw keyword string
            
        Returns:
            Normalized keyword string
        """
        # Convert to lowercase
        normalized = keyword.lower().strip()
        
        # Remove special characters but keep hyphens and spaces
        normalized = re.sub(r'[^\w\s\-]', '', normalized)
        
        # Replace multiple spaces with single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove leading/trailing spaces
        normalized = normalized.strip()
        
        return normalized
    
    def get_keyword_tokens(self, keyword: str) -> List[str]:
        """
        Tokenize and lemmatize keyword for analysis.
        
        Args:
            keyword: Keyword to tokenize
            
        Returns:
            List of processed tokens
        """
        normalized = self.normalize_keyword(keyword)
        tokens = word_tokenize(normalized)
        
        # Filter out stop words and short tokens
        tokens = [token for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        # Lemmatize tokens
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return lemmatized
    
    def calculate_similarity(self, keyword1: str, keyword2: str) -> float:
        """
        Calculate similarity between two keywords using multiple metrics.
        
        Args:
            keyword1: First keyword
            keyword2: Second keyword
            
        Returns:
            Similarity score between 0 and 1
        """
        # Exact match
        if keyword1.lower() == keyword2.lower():
            return 1.0
        
        # Normalized string similarity
        norm1 = self.normalize_keyword(keyword1)
        norm2 = self.normalize_keyword(keyword2)
        
        if norm1 == norm2:
            return 1.0
        
        # Quick exit for very different strings
        if abs(len(norm1) - len(norm2)) > max(len(norm1), len(norm2)) * 0.5:
            return 0.0
        
        # Sequential similarity (fast string matching)
        seq_sim = SequenceMatcher(None, norm1, norm2).ratio()
        
        # If strings are very different, don't bother with token analysis
        if seq_sim < 0.3:
            return seq_sim
        
        # Token-based similarity (more expensive)
        try:
            tokens1 = set(self.get_keyword_tokens(keyword1))
            tokens2 = set(self.get_keyword_tokens(keyword2))
            
            if tokens1 and tokens2:
                token_sim = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
            else:
                token_sim = 0.0
            
            # Stem-based similarity
            stems1 = set(self.stemmer.stem(token) for token in tokens1)
            stems2 = set(self.stemmer.stem(token) for token in tokens2)
            
            if stems1 and stems2:
                stem_sim = len(stems1.intersection(stems2)) / len(stems1.union(stems2))
            else:
                stem_sim = 0.0
            
            # Combine similarities with weights
            combined_similarity = (
                0.3 * seq_sim +
                0.4 * token_sim +
                0.3 * stem_sim
            )
            
            return combined_similarity
            
        except Exception:
            # Fall back to string similarity if tokenization fails
            return seq_sim
    
    def extract_all_keywords(self) -> Dict[str, Dict[str, int]]:
        """
        Extract all keywords from loaded data with frequency counts.
        
        Returns:
            Dictionary mapping categories to keyword frequency counts
        """
        categories = ["keywords", "methodology_keywords", "application_keywords", "technology_keywords"]
        keyword_counts = {category: defaultdict(int) for category in categories}
        
        for entry in self.raw_keywords:
            for category in categories:
                if category in entry:
                    for keyword in entry[category]:
                        normalized = self.normalize_keyword(keyword)
                        keyword_counts[category][normalized] += 1
        
        return keyword_counts
    
    def cluster_keywords(self, category_keywords: Dict[str, int], max_keywords: int = 1000) -> List[KeywordCluster]:
        """
        Cluster keywords within a category based on similarity.
        
        Args:
            category_keywords: Dictionary of keywords and their frequencies
            max_keywords: Maximum number of keywords to process (for performance)
            
        Returns:
            List of keyword clusters
        """
        # Sort keywords by frequency and take top max_keywords for clustering
        sorted_keywords = sorted(
            category_keywords.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:max_keywords]
        
        keywords = [kw for kw, freq in sorted_keywords]
        
        if len(keywords) < 2:
            return []
        
        # Use a more efficient approach for large datasets
        if len(keywords) > 200:
            return self._cluster_keywords_fast(dict(sorted_keywords))
        
        # Calculate pairwise similarities for smaller datasets
        similarity_matrix = np.zeros((len(keywords), len(keywords)))
        
        for i, kw1 in enumerate(keywords):
            for j, kw2 in enumerate(keywords):
                if i != j:
                    similarity_matrix[i][j] = self.calculate_similarity(kw1, kw2)
        
        # Convert similarity to distance matrix for clustering
        distance_matrix = 1 - similarity_matrix
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(
            eps=1 - self.similarity_threshold,
            min_samples=self.min_cluster_size,
            metric='precomputed'
        ).fit(distance_matrix)
        
        # Group keywords by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            if label != -1:  # -1 indicates noise/outliers
                clusters[label].append(keywords[idx])
        
        # Create KeywordCluster objects
        cluster_objects = []
        for cluster_id, cluster_keywords in clusters.items():
            # Select canonical term (most frequent in cluster)
            canonical_term = max(cluster_keywords, 
                                key=lambda k: category_keywords[k])
            
            # Calculate cluster statistics
            total_frequency = sum(category_keywords[k] for k in cluster_keywords)
            
            cluster_obj = KeywordCluster(
                canonical_term=canonical_term,
                variants=cluster_keywords,
                frequency=total_frequency,
                categories={category for category in category_keywords.keys()}
            )
            cluster_objects.append(cluster_obj)
        
        return cluster_objects
    
    def _cluster_keywords_fast(self, category_keywords: Dict[str, int]) -> List[KeywordCluster]:
        """
        Fast clustering approach for large keyword sets using string similarity.
        
        Args:
            category_keywords: Dictionary of keywords and their frequencies
            
        Returns:
            List of keyword clusters
        """
        keywords = list(category_keywords.keys())
        clusters = []
        used_keywords = set()
        
        for i, keyword1 in enumerate(keywords):
            if keyword1 in used_keywords:
                continue
                
            # Create a new cluster starting with this keyword
            cluster_keywords = [keyword1]
            used_keywords.add(keyword1)
            
            # Find similar keywords
            for j, keyword2 in enumerate(keywords[i+1:], i+1):
                if keyword2 in used_keywords:
                    continue
                    
                similarity = self.calculate_similarity(keyword1, keyword2)
                if similarity >= self.similarity_threshold:
                    cluster_keywords.append(keyword2)
                    used_keywords.add(keyword2)
            
            # Only create cluster if it meets minimum size requirement
            if len(cluster_keywords) >= self.min_cluster_size:
                # Select canonical term (most frequent in cluster)
                canonical_term = max(cluster_keywords, 
                                    key=lambda k: category_keywords[k])
                
                # Calculate cluster statistics
                total_frequency = sum(category_keywords[k] for k in cluster_keywords)
                
                cluster_obj = KeywordCluster(
                    canonical_term=canonical_term,
                    variants=cluster_keywords,
                    frequency=total_frequency,
                    categories=set()  # Will be set by caller
                )
                clusters.append(cluster_obj)
        
        return clusters
    
    def harmonize_keywords(self, max_keywords_per_category: int = None) -> Dict[str, List[KeywordCluster]]:
        """
        Harmonize all keywords across categories.
        
        Args:
            max_keywords_per_category: Optional limit on keywords per category (for testing)
        
        Returns:
            Dictionary mapping categories to their keyword clusters
        """
        if not self.raw_keywords:
            raise ValueError("No keywords loaded. Call load_keywords() first.")
        
        keyword_counts = self.extract_all_keywords()
        harmonized_results = {}
        
        for category, keywords in keyword_counts.items():
            
            if len(keywords) < self.min_cluster_size:
                continue
            
            # Apply keyword limit if specified (for testing)
            if max_keywords_per_category and len(keywords) > max_keywords_per_category:
                # Take the top keywords by frequency
                limited_keywords = dict(list(keywords.items())[:max_keywords_per_category])
                keywords = limited_keywords
            
            clusters = self.cluster_keywords(keywords)
            harmonized_results[category] = clusters
            
        return harmonized_results
    
    def save_harmonized_results(self, results: Dict[str, List[KeywordCluster]], 
                               output_file: Union[str, Path]) -> None:
        """
        Save harmonized results to JSON file.
        
        Args:
            results: Harmonized keyword clusters
            output_file: Output file path. If None, uses default location.
        """
        # if output_file is None:
        #     output_file = RESULTS_DIR / "harmonized_keywords.json"
        
        # Convert KeywordCluster objects to serializable format
        serializable_results = {}
        
        for category, clusters in results.items():
            serializable_results[category] = []
            
            for cluster in clusters:
                cluster_data = {
                    "canonical_term": cluster.canonical_term,
                    "variants": cluster.variants,
                    "frequency": cluster.frequency,
                    "categories": list(cluster.categories)
                }
                serializable_results[category].append(cluster_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    def create_mapping_dict(self, results: Dict[str, List[KeywordCluster]]) -> Dict[str, str]:
        """
        Create a mapping dictionary from variant keywords to canonical terms.
        
        Args:
            results: Harmonized keyword clusters
            
        Returns:
            Dictionary mapping variant keywords to canonical terms
        """
        mapping = {}
        
        for category, clusters in results.items():
            for cluster in clusters:
                canonical_term = cluster.canonical_term
                for variant in cluster.variants:
                    mapping[variant] = canonical_term
        
        return mapping
    
    def create_harmonized_terms_file(self, results: Dict[str, List[KeywordCluster]], 
                                     output_file: Union[str, Path] = None) -> None:
        """
        Create a harmonized terms file compatible with categorise.py task.
        This replaces variant keywords with their canonical terms while maintaining
        the same structure as the original keywords.json file.
        
        Args:
            results: Harmonized keyword clusters
            output_file: Output file path for harmonized terms
        """
        if output_file is None:
            output_file = TERMS_PATH
        
        # Create mapping from variants to canonical terms
        mapping = self.create_mapping_dict(results)
        
        # Process original keywords and replace with canonical terms
        harmonized_records = []
        
        for record in self.raw_keywords:
            harmonized_record = {
                "id": record["id"]
            }
            
            # Process each keyword category
            categories = ["keywords", "methodology_keywords", "application_keywords", "technology_keywords"]
            
            for category in categories:
                if category in record and isinstance(record[category], list):
                    # Replace keywords with canonical terms and remove duplicates
                    harmonized_terms = []
                    seen = set()
                    
                    for keyword in record[category]:
                        # Use canonical term if available, otherwise keep original
                        canonical = mapping.get(keyword, keyword)
                        
                        # Only add if we haven't seen this canonical term already
                        if canonical not in seen:
                            harmonized_terms.append(canonical)
                            seen.add(canonical)
                    
                    harmonized_record[category] = harmonized_terms
                else:
                    harmonized_record[category] = []
            
            harmonized_records.append(harmonized_record)
        
        # Save harmonized terms file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(harmonized_records, f, indent=2, ensure_ascii=False)
    



def load_harmonization_clusters() -> List[Dict]:
    """
    Load harmonization clusters for review.
    
    Returns:
        List of cluster dictionaries for review
    """
    clusters_file = RESULTS_DIR / "terms_clusters.json"
    
    if not clusters_file.exists():
        raise FileNotFoundError(f"Clusters file not found at {clusters_file}. Run harmonization first.")
    
    with open(clusters_file, 'r', encoding='utf-8') as f:
        cluster_data = json.load(f)
    
    # Convert to flat list of clusters with metadata
    clusters_for_review = []
    cluster_id = 0
    
    for category, clusters in cluster_data.items():
        for cluster in clusters:
            review_cluster = {
                "cluster_id": cluster_id,
                "category": category,
                "canonical_term": cluster["canonical_term"],
                "variants": cluster["variants"],
                "frequency": cluster["frequency"]
            }
            clusters_for_review.append(review_cluster)
            cluster_id += 1
    
    return clusters_for_review


@task
def review_harmonization() -> Task:
    """
    Review harmonization clusters to identify overly aggressive groupings.
    
    This task presents harmonization clusters to an LLM for review, identifying
    cases where specific terms are inappropriately grouped with general terms
    that could undermine emerging trends identification.
    """
    
    # Load clusters for review
    clusters = load_harmonization_clusters()
    
    # Create input text with clusters
    clusters_text = "HARMONIZATION CLUSTERS TO REVIEW:\n\n"
    
    for i, cluster in enumerate(clusters[:50]):  # Review first 50 clusters as sample
        clusters_text += f"Cluster {cluster['cluster_id']} ({cluster['category']}):\n"
        clusters_text += f"  Canonical Term: {cluster['canonical_term']}\n"
        clusters_text += f"  Variants: {', '.join(cluster['variants'])}\n"
        clusters_text += f"  Frequency: {cluster['frequency']}\n\n"
    
    dataset = MemoryDataset([Sample(
        id="harmonization_review",
        input=clusters_text
    )])
    
    return Task(
        dataset=dataset,
        solver=[
            system_message(str(PROMPTS_DIR / "review_harmonization.txt")),
            generate()
        ],
        scorer=[
            harmonization_reviewer()
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="harmonization_review",
                json_schema=json_schema(HarmonizationReviewOutput),
                strict=True
            )
        ),
        hooks=["HarmonizationReviewHook"]
    )


def harmonize_extracted_keywords(
    keywords_file: Union[str, Path] = None,
    similarity_threshold: float = 0.7,
    min_cluster_size: int = 2,
    output_file: Union[str, Path] = None,
    save_clusters: bool = True,
    max_keywords_per_category: int = None
) -> Tuple[Dict[str, List[KeywordCluster]], Dict]:
    """
    Main function to harmonize extracted keywords into terms.
    
    Args:
        keywords_file: Path to keywords JSON file
        similarity_threshold: Minimum similarity for clustering
        min_cluster_size: Minimum cluster size
        output_file: Output file for categorise-compatible terms (saves to TERMS_PATH if None)
        save_clusters: Whether to also save the detailed cluster information
        max_keywords_per_category: Maximum keywords to process per category (for testing)
        
    Returns:
        Harmonized results
    """
    
    # Initialize harmonizer
    harmonizer = LexicalHarmonizer(
        similarity_threshold=similarity_threshold,
        min_cluster_size=min_cluster_size
    )
    
    # Load and process keywords
    harmonizer.load_keywords(keywords_file)
    results = harmonizer.harmonize_keywords(max_keywords_per_category)

    # Create categorise.py compatible output (primary TERMS_PATH)
    if output_file:
        harmonizer.create_harmonized_terms_file(results, output_file)
    else:
        # Save to default terms file (categorise-compatible format)
        harmonizer.create_harmonized_terms_file(results, TERMS_PATH)
    
    # Optionally save detailed cluster information
    if save_clusters:
        clusters_file = RESULTS_DIR / "terms_clusters.json"
        harmonizer.save_harmonized_results(results, clusters_file)
    
    return results


def apply_harmonization_review() -> None:
    """
    Apply harmonization review decisions to create refined harmonization results.
    
    This function reads the review decisions and creates new harmonization files
    with problematic clusters removed (rejected clusters become individual terms).
    """
    
    # Load review decisions
    if not REVIEW_FILE.exists():
        raise FileNotFoundError(f"Review file not found at {REVIEW_FILE}. Run review_harmonization task first.")

    with open(REVIEW_FILE, 'r', encoding='utf-8') as f:
        review_data = json.load(f)
    
    reviews = review_data.get("reviews", [])
    
    # Load original cluster data
    clusters_file = RESULTS_DIR / "terms_clusters.json"
    with open(clusters_file, 'r', encoding='utf-8') as f:
        original_clusters = json.load(f)
    
    # Create refined clusters based on review decisions
    refined_clusters = {}
    
    for category, clusters in original_clusters.items():
        refined_clusters[category] = []
        
        for cluster in clusters:
            # Find corresponding review decision
            cluster_review = None
            for review in reviews:
                if (review.get("canonical_term") == cluster["canonical_term"] and 
                    set(review.get("variants", [])) == set(cluster["variants"])):
                    cluster_review = review
                    break
            
            if cluster_review is None:
                # No review found, accept original cluster
                refined_clusters[category].append(cluster)
                continue
            
            decision = cluster_review.get("decision", "accept")
            
            if decision == "accept":
                # Accept cluster as is
                refined_clusters[category].append(cluster)
            
            elif decision == "reject":
                # Reject cluster - treat each variant as its own term
                # Don't include the cluster (variants will remain as individual terms)
                continue
            
            else:
                # Unknown decision, default to accept
                refined_clusters[category].append(cluster)
    
    # Save refined clusters
    refined_clusters_file = RESULTS_DIR / "terms_clusters_refined.json"
    with open(refined_clusters_file, 'w', encoding='utf-8') as f:
        json.dump(refined_clusters, f, indent=2, ensure_ascii=False)
    
    # Create refined terms file for categorisation
    refined_terms_file = RESULTS_DIR / "terms_refined.json"
    create_refined_terms_file(refined_clusters, refined_terms_file)


def create_refined_terms_file(refined_clusters: Dict, output_file: Union[str, Path]) -> None:
    """
    Create a refined terms file based on review decisions.
    
    Args:
        refined_clusters: Refined cluster data
        output_file: Output file path
    """
    
    # Load original keywords to rebuild mapping
    with open(KEYWORDS_PATH, 'r', encoding='utf-8') as f:
        raw_keywords = json.load(f)
    
    # Create mapping from refined clusters
    mapping = {}
    for category, clusters in refined_clusters.items():
        for cluster in clusters:
            canonical_term = cluster["canonical_term"]
            for variant in cluster["variants"]:
                mapping[variant] = canonical_term
    
    # Process original keywords and replace with refined canonical terms
    refined_records = []
    
    for record in raw_keywords:
        refined_record = {"id": record["id"]}
        
        categories = ["keywords", "methodology_keywords", "application_keywords", "technology_keywords"]
        
        for category in categories:
            if category in record and isinstance(record[category], list):
                refined_terms = []
                seen = set()
                
                for keyword in record[category]:
                    # Use refined canonical term if available, otherwise keep original
                    canonical = mapping.get(keyword, keyword)
                    
                    if canonical not in seen:
                        refined_terms.append(canonical)
                        seen.add(canonical)
                
                refined_record[category] = refined_terms
            else:
                refined_record[category] = []
        
        refined_records.append(refined_record)
    
    # Save refined terms file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(refined_records, f, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Harmonize keywords from extract task output")
    parser.add_argument("--review", action="store_true", help="Run harmonization review task")
    parser.add_argument("--full", action="store_true", help="Run full harmonization pipeline")
    args = parser.parse_args()
    
    if args.review:
        eval(review_harmonization())
    elif args.full:
        harmonize_extracted_keywords(save_clusters=True)
        eval(review_harmonization())
    else:
        # For testing, limit the number of keywords and use a lower threshold
        results = harmonize_extracted_keywords(
            keywords_file=KEYWORDS_PATH,
            save_clusters=True,
            max_keywords_per_category=100,  # Limit for testing
            similarity_threshold=0.5  # Lower threshold for testing
        )
