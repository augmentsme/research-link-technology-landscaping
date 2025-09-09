from pathlib import Path
from typing import Any, Dict, List
from dotenv import dotenv_values
import json
import math
import jsonlines
from dataclasses import dataclass
import utils

CONFIG = dotenv_values()
ROOT_DIR = Path(CONFIG["ROOT_DIR"])
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"



# GRANTS_FILE = DATA_DIR / "grants_cleaned.jsonl"
PROMPTS_DIR = ROOT_DIR / "PromptTemplates"

FOR_CODES_CLEANED_PATH = DATA_DIR / "for_codes_cleaned.json"

CLASSIFICATION_PATH = RESULTS_DIR / "classification.json"

@dataclass
class Keywords:
    keywords_dir = RESULTS_DIR / "keywords"
    keywords_dir.mkdir(parents=True, exist_ok=True)
    extracted_keywords_path: Path = keywords_dir / "extracted_keywords.jsonl"
    keywords_path: Path = keywords_dir / "keywords.jsonl"
    template = lambda record: f"<keyword><name>{record['name']}</name><type>{record['type']}</type><description>{record['description']}</description></keyword>"
    def load(as_dataframe=True):
        return utils.load_jsonl_file(Keywords.keywords_path, as_dataframe=as_dataframe)
    def load_extracted():
        return utils.load_jsonl_file(Keywords.extracted_keywords_path)


@dataclass
class Categories:
    category_dir: Path = RESULTS_DIR / "category"
    category_dir.mkdir(parents=True, exist_ok=True)
    category_proposal_path: Path = category_dir / "category_proposal.jsonl"
    clusters_cache_path: Path = category_dir / "semantic_clusters_cache.json"
    embeddings_cache_path: Path = category_dir / "embeddings_cache.json"
    
    unknown_keywords_path: Path = category_dir / "unknown_keywords.jsonl"
    missing_keywords_path: Path = category_dir / "missing_keywords.jsonl"
    
    
    merge_missing_path: Path = category_dir / "merge_missing_keywords.jsonl"
    merge_unknown_path: Path = category_dir / "merge_unknown_keywords.jsonl"
    categories_path: Path = category_dir / "categories.jsonl"
    template = lambda record: f"<category><name>{record['name']}</name><description>{record['description']}</description><keywords>{''.join(f'<keyword>{k}</keyword>' for k in record.get('keywords', []))}</keywords></category>"

    def load():
        return utils.load_jsonl_file(Categories.categories_path, as_dataframe=True)
    def load_proposal():
        return utils.load_jsonl_file(Categories.category_proposal_path, as_dataframe=True)
    def load_unknown_keywords():
        return utils.load_jsonl_file(Categories.unknown_keywords_path, as_dataframe=True)
    def load_missing_keywords():
        return utils.load_jsonl_file(Categories.missing_keywords_path, as_dataframe=True)
    

@dataclass
class Grants:
    neo4j_uri = "bolt://localhost:7687"
    neo4j_username = "neo4j"
    cipher_query = "MATCH (g:grant) RETURN g"
    raw_path = DATA_DIR / "grants_raw.json"
    enriched_path = DATA_DIR / "grants_enriched.json"
    grants_path = DATA_DIR / "grants.jsonl"
    neo4j_password = CONFIG["NEO4J_PASSWORD"]  # Ensure this key exists in your .env file
    template = lambda record: f"<grant><title>{record['title']}</title><description>{record['grant_summary']}</description></grant>"


    def load(as_dataframe=True):
        return utils.load_jsonl_file(Grants.grants_path, as_dataframe=as_dataframe)

    def load_enriched(as_dataframe=True):
        return utils.load_json_file(Grants.enriched_path, as_dataframe=as_dataframe)

    def load_raw(as_dataframe=True):
        return utils.load_json_file(Grants.raw_path, as_dataframe=as_dataframe)

def finished_grants():
    if Keywords.keywords_path.exists():
        keywords = utils.load_jsonl_file(Keywords.keywords_path, as_dataframe=True)
        if keywords.empty:
            return []
        return keywords.grants.explode().unique()
    else:
        return []
