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

from sklearn.cluster import DBSCAN
import numpy as np
from difflib import SequenceMatcher
from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import simple_preprocess
import warnings
import gensim.downloader as api
# Inspect AI imports for the review task
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.solver import system_message, generate
from inspect_ai.util import json_schema
from inspect_ai.hooks import Hooks, SampleEnd, hooks, TaskEnd
from inspect_ai.scorer import scorer, Score, Target, INCORRECT, CORRECT, accuracy, NOANSWER
from inspect_ai.solver import TaskState

from pydantic import BaseModel, Field

from config import KEYWORDS_PATH, RESULTS_DIR, PROMPTS_DIR, REVIEW_FILE, CLUSTERS_PROPOSAL_PATH, CLUSTERS_FINAL_PATH, KEYWORDS_FINAL_PATH, SIMILARITY_THRESHOLD

@dataclass
class KeywordCluster:
    """Represents a cluster of harmonized keywords."""
    canonical_term: str
    variants: List[str]
    frequency: int


class ClusterReview(BaseModel):
    """Output model for reviewing a single cluster."""
    model_config = {"extra": "forbid"}
    
    cluster_id: int = Field(description="Index of the cluster being reviewed")
    canonical_term: str = Field(description="The canonical term of the cluster")
    variants: List[str] = Field(description="List of variant terms in the cluster")
    decision: str = Field(description="Decision: 'accept' or 'reject'")
    reasoning: str = Field(description="Explanation for the decision")


@scorer(metrics=[accuracy()])
def harmonization_reviewer():
    """
    Scorer that validates harmonization review decisions for single clusters.
    """
    
    async def score(state: TaskState, target: Target) -> Score:
        """Score based on review decision quality for a single cluster."""
        
        if not state.output or not state.output.completion:
            return Score(
                value=NOANSWER, 
                explanation="No harmonization review output to score"
            )
        
        try:
            result = json.loads(state.output.completion)
            
            # Expect single cluster review format
            decision = result.get("decision", "unknown")
            cluster_id = result.get("cluster_id", "unknown")
            reasoning = result.get("reasoning", "")
            
            # Validate decision
            if decision not in ["accept", "reject"]:
                return Score(
                    value=NOANSWER,
                    explanation=f"Invalid decision '{decision}' for cluster {cluster_id}. Expected 'accept' or 'reject'"
                )
            return Score(
                value=CORRECT if decision == "accept" else INCORRECT,
            )
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return Score(
                value=NOANSWER, 
                explanation=f"Error parsing harmonization review result: {str(e)}"
            )
    
    return score


@hooks(name="HarmonizationReviewHook", description="Hook to save harmonization review results and apply refinements")
class HarmonizationReviewHook(Hooks):
    """Hook to save harmonization review results and automatically apply refinements."""

    def __init__(self):
        super().__init__()
        self.reviews = []

    async def on_sample_end(self, data: SampleEnd) -> None:
        """Save individual cluster review result."""
        try:
            output_text = data.sample.output.completion
            result_json = json.loads(output_text)
            self.reviews.append(result_json)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # Silently handle errors

    async def on_task_end(self, data: TaskEnd) -> None:
        """Aggregate all reviews and apply harmonization refinements."""
        try:
            # Save aggregated reviews in the expected format
            aggregated_result = {"reviews": self.reviews}
            with open(REVIEW_FILE, 'w', encoding='utf-8') as f:
                json.dump(aggregated_result, f, indent=2, ensure_ascii=False)
            
            # Apply refinements
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
        
        # Initialize Word2Vec model (will be loaded later)
        self.word2vec_model = None
        
        # Keyword storage
        self.raw_keywords = []
        self.normalized_keywords = {}
        self.clusters = []
        
    def load_keywords(self, keywords_file: Union[str, Path]) -> None:
        """
        Load keywords from the extract task output file.
        
        Args:
            keywords_file: Path to keywords JSON file.
        """
        keywords_path = Path(keywords_file)
        with keywords_path.open('r', encoding='utf-8') as f:
            self.raw_keywords = json.load(f)
        
        # Load pretrained Word2Vec model
        self._load_pretrained_word2vec_model()
    
    def _load_pretrained_word2vec_model(self) -> None:
        """
        Load a pretrained Word2Vec model for semantic similarity.
        Falls back to training a simple model if pretrained model is not available.
        """

        
        
        # Try to load from gensim-data first
        self.word2vec_model = api.load("word2vec-google-news-300")

    
    def _train_simple_word2vec_model(self) -> None:
        """
        Fallback method to train a simple Word2Vec model on the loaded keywords.
        """
        if not self.raw_keywords:
            self.word2vec_model = None
            return
        
        # Collect all keywords for training
        all_keywords = []
        categories = ["keywords", "methodology_keywords", "application_keywords", "technology_keywords"]
        
        for entry in self.raw_keywords:
            for category in categories:
                if category in entry:
                    all_keywords.extend(entry[category])
        
        # Tokenize keywords for Word2Vec training
        tokenized_keywords = []
        for keyword in all_keywords:
            tokens = self.get_keyword_tokens(keyword)
            if tokens:  # Only add non-empty token lists
                tokenized_keywords.append(tokens)
        
        # Train Word2Vec model if we have enough data
        if tokenized_keywords and len(tokenized_keywords) > 1:
            try:
                # Suppress gensim warnings during training
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = Word2Vec(
                        sentences=tokenized_keywords,
                        vector_size=100,
                        window=5,
                        min_count=1,
                        workers=1,
                        epochs=10
                    )
                    self.word2vec_model = model.wv  # Use KeyedVectors for consistency
            except Exception:
                # If Word2Vec training fails, model remains None
                self.word2vec_model = None
        else:
            self.word2vec_model = None
    
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
        Tokenize keyword for analysis using gensim tokenizer.
        
        Args:
            keyword: Keyword to tokenize
            
        Returns:
            List of processed tokens
        """
        normalized = self.normalize_keyword(keyword)
        
        # Use gensim's simple_preprocess which handles tokenization, lowercasing, 
        # and filtering of short tokens and punctuation
        tokens = simple_preprocess(normalized, min_len=3)
        
        return tokens
    

    
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
        
        if not self.word2vec_model:
            return []
        
        # Get vectors for keywords that exist in the model
        vectors = []
        valid_keywords = []
        
        for keyword in keywords:
            tokens = self.get_keyword_tokens(keyword)
            keyword_vectors = []
            
            for token in tokens:
                if token in self.word2vec_model:
                    keyword_vectors.append(self.word2vec_model[token])
            
            if keyword_vectors:
                # Calculate average vector for multi-token keywords
                avg_vector = np.mean(keyword_vectors, axis=0)
                vectors.append(avg_vector)
                valid_keywords.append(keyword)
        

        # Convert to numpy array for sklearn
        vectors_array = np.array(vectors)
        
        # Apply DBSCAN clustering directly on vectors
        clustering = DBSCAN(
            eps=1 - self.similarity_threshold,
            min_samples=self.min_cluster_size,
            metric='cosine'
        ).fit(vectors_array)
        
        # Group keywords by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            if label != -1:  # -1 indicates noise/outliers
                clusters[label].append(valid_keywords[idx])
        
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
                frequency=total_frequency
            )
            cluster_objects.append(cluster_obj)
        
        return cluster_objects
    
    
    def harmonize_keywords(self) -> Dict[str, List[KeywordCluster]]:
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
            
            limited_keywords = dict(list(keywords.items()))
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
                    "frequency": cluster.frequency
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
                                     output_file: Union[str, Path]) -> None:
        """
        Create a harmonized terms file compatible with categorise.py task.
        This replaces variant keywords with their canonical terms while maintaining
        the same structure as the original keywords.json file.
        
        Args:
            results: Harmonized keyword clusters
            output_file: Output file path for harmonized terms
        """
        # Create mapping from variants to canonical terms
        mapping = self.create_mapping_dict(results)
        
        # Use unified function
        create_harmonized_keywords_file(self.raw_keywords, mapping, output_file)
    



def load_harmonization_clusters() -> List[Dict]:

    clusters_file = CLUSTERS_PROPOSAL_PATH
    
    # if not clusters_file.exists():
        # harmonize_extracted_keywords(clusters_path=CLUSTERS_PROPOSAL_PATH)    
    
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
def harmonise() -> Task:
    """
    Review harmonization clusters to identify overly aggressive groupings.
    
    This task presents harmonization clusters to an LLM for review, identifying
    cases where specific terms are inappropriately grouped with general terms
    that could undermine emerging trends identification.
    """

    harmonize_extracted_keywords(clusters_path=CLUSTERS_PROPOSAL_PATH, keywords_file=KEYWORDS_PATH, similarity_threshold=SIMILARITY_THRESHOLD)
    # Load clusters for review
    clusters = load_harmonization_clusters()
    
    # Create one sample per cluster for individual review
    samples = []
    for cluster in clusters:
        cluster_text = f"HARMONIZATION CLUSTER TO REVIEW:\n\n"
        cluster_text += f"Cluster {cluster['cluster_id']} ({cluster['category']}):\n"
        cluster_text += f"  Canonical Term: {cluster['canonical_term']}\n"
        cluster_text += f"  Variants: {', '.join(cluster['variants'])}\n"
        cluster_text += f"  Frequency: {cluster['frequency']}\n"
        
        samples.append(Sample(
            id=f"cluster_{cluster['cluster_id']}",
            input=cluster_text,
            metadata={"cluster_id": cluster['cluster_id'], "category": cluster['category']}
        ))
    
    dataset = MemoryDataset(samples)
    
    return Task(
        dataset=dataset,
        solver=[
            system_message(str(PROMPTS_DIR / "harmonise.txt")),
            generate()
        ],
        scorer=[
            harmonization_reviewer()
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="single_cluster_review",
                json_schema=json_schema(ClusterReview),
                strict=True
            )
        ),
        hooks=["HarmonizationReviewHook"]
    )


def harmonize_extracted_keywords(
    clusters_path: Path,
    similarity_threshold: float,
    keywords_file: Union[str, Path],
    min_cluster_size: int = 2,
) -> Tuple[Dict[str, List[KeywordCluster]], Dict]:

    harmonizer = LexicalHarmonizer(
        similarity_threshold=similarity_threshold,
        min_cluster_size=min_cluster_size
    )
    
    harmonizer.load_keywords(keywords_file)
    results = harmonizer.harmonize_keywords()

    harmonizer.save_harmonized_results(results, clusters_path)
    
    return results


def apply_harmonization_review() -> None:
    """
    Apply harmonization review decisions to create refined harmonization results.
    
    This function reads the review decisions and creates new harmonization files
    with problematic clusters removed (rejected clusters become individual terms).
    """
    
    # Load review decisions
    if not REVIEW_FILE.exists():
        raise FileNotFoundError(f"Review file not found at {REVIEW_FILE}. Run harmonise task first.")

    with open(REVIEW_FILE, 'r', encoding='utf-8') as f:
        review_data = json.load(f)
    
    reviews = review_data.get("reviews", [])
    
    # Load original cluster data
    clusters_file = CLUSTERS_PROPOSAL_PATH
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
    refined_clusters_file = CLUSTERS_FINAL_PATH
    with open(refined_clusters_file, 'w', encoding='utf-8') as f:
        json.dump(refined_clusters, f, indent=2, ensure_ascii=False)
    
    # Create refined terms file for categorisation
    refined_terms_file = KEYWORDS_FINAL_PATH
    # Load original keywords to rebuild mapping
    with open(KEYWORDS_PATH, 'r', encoding='utf-8') as f:
        initial_keywords = json.load(f)
    
    # Create mapping from refined clusters and use unified function
    mapping = create_mapping_from_clusters(refined_clusters)
    create_harmonized_keywords_file(initial_keywords, mapping, refined_terms_file)


def create_harmonized_keywords_file(initial_keywords: List[Dict], mapping: Dict[str, str], output_file: Union[str, Path]) -> None:
    """
    Create a harmonized/refined keywords file by replacing variants with canonical terms.
    
    This unified function handles both harmonization (from KeywordCluster results) and 
    refinement (from review decisions) by taking a mapping dictionary.
    
    Args:
        initial_keywords: Original keywords data (list of records)
        mapping: Dictionary mapping variant terms to canonical terms
        output_file: Output file path
    """
    # Create case-insensitive mapping for lookup
    case_insensitive_mapping = {}
    for variant, canonical in mapping.items():
        case_insensitive_mapping[variant.lower()] = canonical
    
    # Process original keywords and replace with canonical terms
    harmonized_records = []
    
    for record in initial_keywords:
        harmonized_record = {"id": record["id"]}
        
        categories = ["keywords", "methodology_keywords", "application_keywords", "technology_keywords"]
        
        for category in categories:
            if category in record and isinstance(record[category], list):
                harmonized_terms = []
                seen = set()
                
                for keyword in record[category]:
                    # Use case-insensitive lookup for canonical term
                    canonical = case_insensitive_mapping.get(keyword.lower(), keyword)
                    
                    if canonical not in seen:
                        harmonized_terms.append(canonical)
                        seen.add(canonical)
                
                harmonized_record[category] = harmonized_terms
            else:
                harmonized_record[category] = []
        
        harmonized_records.append(harmonized_record)

    # Save harmonized keywords file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(harmonized_records, f, indent=2, ensure_ascii=False)


def create_mapping_from_clusters(refined_clusters: Dict) -> Dict[str, str]:
    """
    Create a mapping dictionary from refined clusters data.
    
    Args:
        refined_clusters: Dictionary of refined cluster data
        
    Returns:
        Dictionary mapping variant terms to canonical terms
    """
    mapping = {}
    for category, clusters in refined_clusters.items():
        for cluster in clusters:
            canonical_term = cluster["canonical_term"]
            for variant in cluster["variants"]:
                mapping[variant] = canonical_term
    return mapping


def main():

    import shutil

    harmonize_extracted_keywords(
        clusters_path=CLUSTERS_PROPOSAL_PATH,
        keywords_file=KEYWORDS_PATH,
        similarity_threshold=SIMILARITY_THRESHOLD,
        min_cluster_size=2
    )

    shutil.copy2(CLUSTERS_PROPOSAL_PATH, CLUSTERS_FINAL_PATH)

    

    # Load original keywords
    with open(KEYWORDS_PATH, 'r', encoding='utf-8') as f:
        initial_keywords = json.load(f)
    
    # Load final clusters to create mapping
    with open(CLUSTERS_FINAL_PATH, 'r', encoding='utf-8') as f:
        final_clusters = json.load(f)
    
    # Create mapping and harmonized keywords file
    mapping = create_mapping_from_clusters(final_clusters)
    create_harmonized_keywords_file(initial_keywords, mapping, KEYWORDS_FINAL_PATH)




if __name__ == "__main__":
    main()

