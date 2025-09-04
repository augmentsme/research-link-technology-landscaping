from pathlib import Path
import tiktoken
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






# KEYWORDS_EMBEDDING_DBPATH = RESULTS_DIR / "keywords_embeddings"
# SIMILARITY_THRESHOLD = 0.75
# MIN_CLUSTER_SIZE = 5



# CLUSTERS_PROPOSAL_PATH = RESULTS_DIR / "clusters_proposal.json"
# REVIEW_FILE = RESULTS_DIR / "review.json"
# CLUSTERS_FINAL_PATH = RESULTS_DIR / "clusters_final.json"

# KEYWORDS_PATH = RESULTS_DIR / "keywords.json"


# BATCH_SIZE = 100
# KEYWORDS_TYPE = None

# CATEGORY_DIR = RESULTS_DIR / "category"
# CATEGORY_PROPOSAL_PATH = CATEGORY_DIR / "category_proposal.json"
# CATEGORY_PATH = CATEGORY_DIR / "categories.json"

# COARSENED_CATEGORY_PATH = RESULTS_DIR / "coarsened_categories.json"
# REFINED_CATEGORY_PATH = RESULTS_DIR / "refined_categories.json"
# COMPREHENSIVE_TAXONOMY_PATH = RESULTS_DIR / "comprehensive_taxonomy.json"
FOR_CODES_CLEANED_PATH = DATA_DIR / "for_codes_cleaned.json"

CLASSIFICATION_PATH = RESULTS_DIR / "classification.json"

@dataclass
class db:
    # port = 8001
    # host = "localhost"
    path = RESULTS_DIR / "db"
    api_base = "http://localhost:8000/v1"
    model = "Qwen/Qwen3-Embedding-0.6B"


@dataclass
class Keywords:
    extracted_keywords_path: Path = RESULTS_DIR / "extracted_keywords.jsonl"
    keywords_path: Path = RESULTS_DIR / "keywords.jsonl"
    template = lambda record: f"<keyword><term>{record['term']}</term><description>{record['description']}</description></keyword>"
    def load():
        return utils.load_jsonl_file(Keywords.keywords_path, as_dataframe=True)
    def load_extracted():
        return utils.load_jsonl_file(Keywords.extracted_keywords_path)


@dataclass
class Categories:
    category_dir: Path = RESULTS_DIR / "category"
    category_proposal_path: Path = category_dir / "category_proposal.jsonl"
    batch_size: int = 100
    keywords_type: str = None
    categories_path: Path = category_dir / "categories.jsonl"
    unknown_keywords_path: Path = category_dir / "unknown_keywords.jsonl"
    missing_keywords_path: Path = category_dir / "missing_keywords.jsonl"
    template = lambda record: f"<category><name>{record['name']}</name><description>{record['description']}</description><keywords>{', '.join(record['keywords'])}</keywords></category>"
    def __post_init__(self):
        self.category_dir.mkdir(parents=True, exist_ok=True)
    def load():
        return utils.load_jsonl_file(Categories.categories_path, as_dataframe=True)
    def load_proposal():
        return utils.load_jsonl_file(Categories.category_proposal_path, as_dataframe=True)
    def load_unknown_keywords():
        return utils.load_jsonl_file(Categories.unknown_keywords_path, as_dataframe=True)
    def load_missing_keywords():
        return utils.load_jsonl_file(Categories.missing_keywords_path, as_dataframe=True)
    # def count_proposal_token():
    #     from transformers import AutoTokenizer
    #     proposals = Categories.load_proposal()
    #     return proposals.apply(Categories.template, axis=1).map(AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507").encode).map(len)

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


    def load():
        return utils.load_jsonl_file(Grants.grants_path, as_dataframe=True)

    def load_enriched():
        return utils.load_json_file(Grants.enriched_path, as_dataframe=True)

    def load_raw():
        return utils.load_json_file(Grants.raw_path, as_dataframe=True)

def finished_grants():
    keywords = utils.load_jsonl_file(Keywords.keywords_path, as_dataframe=True)
    return keywords.grants.explode().unique()
    
    