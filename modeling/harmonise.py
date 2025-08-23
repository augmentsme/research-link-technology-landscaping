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

from config import RESULTS_DIR, PROMPTS_DIR, REVIEW_FILE, CLUSTERS_PROPOSAL_PATH, CLUSTERS_FINAL_PATH, KEYWORDS_PATH, SIMILARITY_THRESHOLD, EXTRACTED_KEYWORDS_PATH
from extract import Keyword

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



class Harmoniser:


    def __init__(self, similarity_threshold: float = 0.7, min_cluster_size: int = 2):

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
        Expects new flattened structure: array of keyword objects at top level.
        
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
        
        # Collect all keywords for training from flat structure
        all_keywords = []
        
        for keyword in self.raw_keywords:
            # Extract term from keyword object
            term = keyword.get('term', '')
            description = keyword.get('description', '')
            # Combine term and description for richer training data
            combined_text = term
            if description:
                combined_text += " " + description
            all_keywords.append(combined_text)
        
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
    

    
    def extract_all_keywords(self) -> Dict[str, Dict]:
        """
        Extract all keywords from loaded flat data with enhanced information.
        
        Returns:
            Dictionary mapping keyword terms to their information including frequency, keyword objects, etc.
        """
        keyword_info = defaultdict(lambda: {'frequency': 0, 'keyword_objects': []})
        
        for keyword in self.raw_keywords:
            term = keyword.get('term', '')
            
            normalized_term = self.normalize_keyword(term)
            keyword_info[normalized_term]['frequency'] += 1
            keyword_info[normalized_term]['keyword_objects'].append(keyword)
        
        return keyword_info
    
    def cluster_keywords(self, keyword_info: Dict[str, Dict], max_keywords: int = 1000) -> List[KeywordCluster]:
        """
        Cluster keywords based on similarity using both terms and descriptions.
        
        Args:
            keyword_info: Dictionary of keywords with enhanced information (frequency, descriptions, ids, types)
            max_keywords: Maximum number of keywords to process (for performance)
            
        Returns:
            List of keyword clusters
        """
        # Sort keywords by frequency and take top max_keywords for clustering
        sorted_keywords = sorted(
            keyword_info.items(), 
            key=lambda x: x[1]['frequency'], 
            reverse=True
        )[:max_keywords]
        
        keywords = [kw for kw, info in sorted_keywords]
        
        if len(keywords) < 2:
            return []
        
        if not self.word2vec_model:
            return []
        
        # Get vectors for keywords that exist in the model
        vectors = []
        valid_keywords = []
        
        for keyword in keywords:
            kw_info = keyword_info[keyword]
            
            # Combine term and descriptions for richer semantic representation
            combined_text = keyword
            descriptions = kw_info.get('descriptions', [])
            if descriptions:
                # Use the most common description (first one) or combine multiple descriptions
                combined_text += " " + descriptions[0]
            
            # Tokenize combined text
            tokens = self.get_keyword_tokens(combined_text)
            keyword_vectors = []
            
            for token in tokens:
                if token in self.word2vec_model:
                    keyword_vectors.append(self.word2vec_model[token])
            
            if keyword_vectors:
                # Calculate average vector for multi-token keywords
                avg_vector = np.mean(keyword_vectors, axis=0)
                vectors.append(avg_vector)
                valid_keywords.append(keyword)
        
        if len(vectors) < 2:
            return []

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
        
        # Create KeywordCluster objects with enhanced information
        cluster_objects = []
        for cluster_id, cluster_keywords in clusters.items():
            # Select canonical term (most frequent in cluster)
            canonical_term = max(cluster_keywords, 
                                key=lambda k: keyword_info[k]['frequency'])
            
            # Calculate cluster statistics
            total_frequency = sum(keyword_info[k]['frequency'] for k in cluster_keywords)
            
            # Collect all keyword objects for the cluster
            all_keyword_objects = []
            for kw in cluster_keywords:
                all_keyword_objects.extend(keyword_info[kw].get('keyword_objects', []))
            
            cluster_obj = KeywordCluster(
                canonical_term=canonical_term,
                variants=cluster_keywords,
                frequency=total_frequency
            )
            # Add enhanced information as attributes
            cluster_obj.keyword_objects = all_keyword_objects
            
            cluster_objects.append(cluster_obj)
        
        return cluster_objects
    
    
    def harmonize_keywords(self) -> List[KeywordCluster]:
        """
        Harmonize all keywords without artificial categorization.
        
        Returns:
            List of keyword clusters
        """
        if not self.raw_keywords:
            raise ValueError("No keywords loaded. Call load_keywords() first.")
        
        keyword_info = self.extract_all_keywords()
        
        if len(keyword_info) < self.min_cluster_size:
            return []
            
        clusters = self.cluster_keywords(keyword_info)
        
        return clusters
    
    def save_harmonized_results(self, results: List[KeywordCluster], 
                               output_file: Union[str, Path]) -> None:
        """
        Save harmonized results to JSON file with keyword objects maintained.
        
        Args:
            results: List of harmonized keyword clusters
            output_file: Output file path.
        """
        # Convert KeywordCluster objects to serializable format
        serializable_results = []
        
        for i, cluster in enumerate(results):
            cluster_data = {
                "cluster_id": i,
                "canonical_term": cluster.canonical_term,
                "variants": getattr(cluster, 'keyword_objects', []),
                "frequency": cluster.frequency,
            }
            serializable_results.append(cluster_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    def create_mapping_dict(self, results: List[KeywordCluster]) -> Dict[str, str]:
        """
        Create a mapping dictionary from variant keywords to canonical terms.
        
        Args:
            results: List of harmonized keyword clusters
            
        Returns:
            Dictionary mapping variant keywords to canonical terms
        """
        mapping = {}
        
        for cluster in results:
            canonical_term = cluster.canonical_term
            keyword_objects = getattr(cluster, 'keyword_objects', [])
            
            for keyword_obj in keyword_objects:
                if isinstance(keyword_obj, dict):
                    variant_term = keyword_obj.get('term', '')
                    if variant_term:
                        mapping[variant_term] = canonical_term
                else:
                    # Fallback for old string format
                    mapping[str(keyword_obj)] = canonical_term
        
        return mapping
    
    def create_harmonized_terms_file(self, results: List[KeywordCluster], 
                                     output_file: Union[str, Path]) -> None:
        """
        Create a harmonized terms file compatible with categorise.py task.
        This replaces variant keywords with their canonical terms while maintaining
        the same structure as the original keywords.json file.
        
        Args:
            results: List of harmonized keyword clusters
            output_file: Output file path for harmonized terms
        """
        # Create mapping from variants to canonical terms
        mapping = self.create_mapping_dict(results)
        
        # Use unified function
        create_harmonized_keywords_file(self.raw_keywords, mapping, output_file)
    



def load_harmonization_clusters() -> List[Dict]:
    """
    Load harmonization clusters from the proposal file.
    
    Returns:
        List of clusters with enhanced information for review.
    """
    clusters_file = CLUSTERS_PROPOSAL_PATH
    
    with open(clusters_file, 'r', encoding='utf-8') as f:
        cluster_data = json.load(f)
    
    # Convert to list of clusters with metadata for review
    clusters_for_review = []
    
    for cluster in cluster_data:
        # Extract information from variant keyword objects
        variants = cluster.get("variants", [])
        variant_terms = []
        descriptions_info = {}
        all_descriptions = []
        types = []
        
        for variant_obj in variants:
            if isinstance(variant_obj, dict):
                term = variant_obj.get("term", "")
                description = variant_obj.get("description", "No description available")
                variant_type = variant_obj.get("type", "general")
                
                variant_terms.append(term)
                descriptions_info[term] = description
                all_descriptions.append(description)
                types.append(variant_type)
            else:
                # Fallback for old string format
                variant_terms.append(str(variant_obj))
                descriptions_info[str(variant_obj)] = "No description available"
        
        review_cluster = {
            "cluster_id": cluster.get("cluster_id", cluster.get("id", 0)),
            "canonical_term": cluster["canonical_term"],
            "variants": variant_terms,
            "frequency": cluster["frequency"],
            "descriptions": descriptions_info,
            "all_descriptions": all_descriptions[:5],  # Limit to first 5 descriptions
            "types": types
        }
        clusters_for_review.append(review_cluster)
    
    return clusters_for_review


@task
def harmonise() -> Task:
    """
    Review harmonization clusters to identify overly aggressive groupings.
    
    This task presents harmonization clusters to an LLM for review, identifying
    cases where specific terms are inappropriately grouped with general terms
    that could undermine emerging trends identification.
    """

    harmonize_extracted_keywords(clusters_path=CLUSTERS_PROPOSAL_PATH, keywords_file=EXTRACTED_KEYWORDS_PATH, similarity_threshold=SIMILARITY_THRESHOLD)
    # Load clusters for review
    clusters = load_harmonization_clusters()
    
    # Create one sample per cluster for individual review
    samples = []
    for cluster in clusters:
        cluster_text = f"HARMONIZATION CLUSTER TO REVIEW:\n\n"
        cluster_text += f"Cluster {cluster['cluster_id']}:\n"
        cluster_text += f"  Canonical Term: {cluster['canonical_term']}\n"
        cluster_text += f"  Variants: {', '.join(cluster['variants'])}\n"
        cluster_text += f"  Frequency: {cluster['frequency']}\n"
        
        # Add type information if available
        types = cluster.get('types', [])
        if types:
            cluster_text += f"  Types: {', '.join(set(types))}\n"
        
        cluster_text += "\n"
        
        # Add descriptions for context
        cluster_text += "KEYWORD DESCRIPTIONS:\n"
        descriptions_info = cluster.get('descriptions', {})
        for variant, description in descriptions_info.items():
            cluster_text += f"  • {variant}: {description}\n"
        
        # Add sample of all descriptions if available
        all_descriptions = cluster.get('all_descriptions', [])
        if all_descriptions:
            cluster_text += f"\nSAMPLE DESCRIPTIONS FROM CLUSTER:\n"
            for i, desc in enumerate(all_descriptions[:3], 1):
                cluster_text += f"  {i}. {desc}\n"
        
        samples.append(Sample(
            id=f"cluster_{cluster['cluster_id']}",
            input=cluster_text,
            metadata={"cluster_id": cluster['cluster_id']}
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
) -> List[KeywordCluster]:

    harmonizer = Harmoniser(
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
    refined_clusters = []
    
    for cluster in original_clusters:
        # Find corresponding review decision
        cluster_review = None
        
        # Extract variant terms from the cluster for comparison
        cluster_variant_terms = []
        variants = cluster.get("variants", [])
        for variant in variants:
            if isinstance(variant, dict):
                cluster_variant_terms.append(variant.get("term", ""))
            else:
                cluster_variant_terms.append(str(variant))
        
        for review in reviews:
            if (review.get("canonical_term") == cluster["canonical_term"] and 
                set(review.get("variants", [])) == set(cluster_variant_terms)):
                cluster_review = review
                break
        
        if cluster_review is None:
            # No review found, accept original cluster
            refined_clusters.append(cluster)
            continue
        
        decision = cluster_review.get("decision", "accept")
        
        if decision == "accept":
            # Accept cluster as is
            refined_clusters.append(cluster)
        
        elif decision == "reject":
            # Reject cluster - treat each variant as its own term
            # Don't include the cluster (variants will remain as individual terms)
            continue
        
        else:
            # Unknown decision, default to accept
            refined_clusters.append(cluster)
    
    # Save refined clusters
    refined_clusters_file = CLUSTERS_FINAL_PATH
    with open(refined_clusters_file, 'w', encoding='utf-8') as f:
        json.dump(refined_clusters, f, indent=2, ensure_ascii=False)
    
    # Create refined terms file for categorisation
    refined_terms_file = KEYWORDS_PATH
    # Load original keywords to rebuild mapping
    with open(EXTRACTED_KEYWORDS_PATH, 'r', encoding='utf-8') as f:
        initial_keywords = json.load(f)
    
    # Create mapping from refined clusters and use unified function
    mapping = create_mapping_from_clusters(refined_clusters)
    create_harmonized_keywords_file(initial_keywords, mapping, refined_terms_file)


def create_harmonized_keywords_file(initial_keywords: List[Dict], mapping: Dict[str, str], output_file: Union[str, Path]) -> None:
    """
    Create a harmonized/refined keywords file by replacing variants with canonical terms.
    Handles the new flattened keyword structure.
    
    Args:
        initial_keywords: Original flattened keywords data (list of keyword objects)
        mapping: Dictionary mapping variant terms to canonical terms
        output_file: Output file path
    """
    # Create case-insensitive mapping for lookup
    case_insensitive_mapping = {}
    for variant, canonical in mapping.items():
        case_insensitive_mapping[variant.lower()] = canonical
    
    # Process original keywords and replace with canonical terms
    harmonized_keywords = []
    seen_terms = set()
    
    for keyword in initial_keywords:
        if isinstance(keyword, dict) and 'term' in keyword:
            original_term = keyword['term']
            # Use case-insensitive lookup for canonical term
            canonical_term = case_insensitive_mapping.get(original_term.lower(), original_term)
            
            # Avoid duplicates - only keep first occurrence of each canonical term
            if canonical_term.lower() not in seen_terms:
                harmonized_keyword = {
                    'id': keyword.get('id', ''),
                    'term': canonical_term,
                    'type': keyword.get('type', 'general'),
                    'description': keyword.get('description', '')
                }
                harmonized_keywords.append(harmonized_keyword)
                seen_terms.add(canonical_term.lower())

    # Save harmonized keywords file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(harmonized_keywords, f, indent=2, ensure_ascii=False)


def create_mapping_from_clusters(refined_clusters: List) -> Dict[str, str]:
    """
    Create a mapping dictionary from refined clusters data.
    
    Args:
        refined_clusters: List of refined cluster data
        
    Returns:
        Dictionary mapping variant terms to canonical terms
    """
    mapping = {}
    for cluster in refined_clusters:
        canonical_term = cluster["canonical_term"]
        variants = cluster["variants"]
        
        for variant in variants:
            if isinstance(variant, dict):
                variant_term = variant.get("term", "")
                if variant_term:
                    mapping[variant_term] = canonical_term
            else:
                # Fallback for old string format
                mapping[str(variant)] = canonical_term
    return mapping


