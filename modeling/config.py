from pathlib import Path
from typing import Any, Dict, List
from dotenv import dotenv_values
import json
import math
import jsonlines
from dataclasses import dataclass
import utils
import pandas as pd
CONFIG = dotenv_values()
ROOT_DIR = Path(CONFIG["ROOT_DIR"])
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"

PROMPTS_DIR = ROOT_DIR / "PromptTemplates"

CLASSIFICATION_PATH = RESULTS_DIR / "classification.json"

@dataclass
class Keywords:
    keywords_dir = RESULTS_DIR / "keywords"
    keywords_dir.mkdir(parents=True, exist_ok=True)
    extracted_keywords_path: Path = keywords_dir / "extracted_keywords.jsonl"
    keywords_path: Path = keywords_dir / "keywords.jsonl"
    
    # Semantic clustering paths
    keywords_batches_dir: Path = RESULTS_DIR / "keywords_batches"
    keyword_embeddings_path: Path = keywords_batches_dir / "keyword_embeddings.npy"
    semantic_clusters_path: Path = keywords_batches_dir / "semantic_clusters.json"
    batch_dir: Path = keywords_batches_dir / "batches"
    
    template = lambda record: f"<keyword><name>{record['name']}</name><type>{record['type']}</type><description>{record['description']}</description></keyword>"
    
    def load(as_dataframe=True):
        return utils.load_jsonl_file(Keywords.keywords_path, as_dataframe=as_dataframe)
    
    def load_extracted(as_dataframe=True):
        return utils.load_jsonl_file(Keywords.extracted_keywords_path, as_dataframe=as_dataframe)


@dataclass
class Categories:
    category_dir: Path = RESULTS_DIR / "category"
    category_dir.mkdir(parents=True, exist_ok=True)
    category_extracted_path: Path = category_dir / "category_extracted.jsonl"
    category_proposal_path: Path = category_dir / "category_proposal.jsonl"
    
    # Semantic clustering paths
    categories_batches_dir: Path = RESULTS_DIR / "categories_batches"
    categories_batches_dir.mkdir(parents=True, exist_ok=True)
    category_embeddings_path: Path = categories_batches_dir / "category_embeddings.npy"
    semantic_clusters_path: Path = categories_batches_dir / "semantic_clusters.json"
    batch_dir: Path = categories_batches_dir / "batches"

    unknown_keywords_path: Path = category_dir / "unknown_keywords.jsonl"
    missing_keywords_path: Path = category_dir / "missing_keywords.jsonl"
    
    merge_dir: Path = RESULTS_DIR / "merge"
    merge_dir.mkdir(parents=True, exist_ok=True)
    merge_missing_path: Path = merge_dir / "merge_missing_keywords.jsonl"
    merge_unknown_path: Path = merge_dir / "merge_unknown_keywords.jsonl"
    merged_categories_path: Path = merge_dir / "merged_categories.jsonl"
    
    final_categories_path: Path = category_dir / "final_categories.jsonl"

    template = lambda record: f"<category><name>{record['name']}</name><description>{record['description']}</description><keywords>{','.join(record.get('keywords', []))}</keywords></category>"

    def load(as_dataframe=True):
        df = utils.load_jsonl_file(Categories.category_proposal_path, as_dataframe=as_dataframe)
        return df
    def load_merged(as_dataframe=True):
        df = utils.load_jsonl_file(Categories.merged_categories_path, as_dataframe=as_dataframe)
        return df
    def load_proposal(as_dataframe=True):
        df = utils.load_jsonl_file(Categories.category_proposal_path, as_dataframe=as_dataframe)
        return df

    def load_unknown_keywords(as_dataframe=True):
        return utils.load_jsonl_file(Categories.unknown_keywords_path, as_dataframe=as_dataframe)

    def load_missing_keywords(as_dataframe=True):
        return utils.load_jsonl_file(Categories.missing_keywords_path, as_dataframe=as_dataframe)


@dataclass
class Grants:
    grants_dir = RESULTS_DIR / "grants"
    grants_dir.mkdir(parents=True, exist_ok=True)
    raw_path = grants_dir / "grants_raw.jsonl"
    enriched_path = grants_dir / "grants_enriched.jsonl"
    grants_path = grants_dir / "grants.jsonl"
    
    neo4j_uri = "bolt://localhost:7687"
    neo4j_username = "neo4j"
    neo4j_password = CONFIG["NEO4J_PASSWORD"]
    
    
    link_path = grants_dir / "org_researcher_grant_links.jsonl"
    

    template = lambda record: f"<grant><title>{record['title']}</title><description>{record['grant_summary']}</description></grant>"

    def load(as_dataframe=True):
        return utils.load_jsonl_file(Grants.grants_path, as_dataframe=as_dataframe)

    def load_enriched(as_dataframe=True):
        return utils.load_jsonl_file(Grants.enriched_path, as_dataframe=as_dataframe)

    def load_raw(as_dataframe=True):
        return utils.load_jsonl_file(Grants.raw_path, as_dataframe=as_dataframe)
    
    def load_links(as_dataframe=True):
        return utils.load_jsonl_file(Grants.link_path, as_dataframe=as_dataframe)
    