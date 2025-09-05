from pathlib import Path
from matplotlib.pylab import record
import tiktoken
from typing import Any, Dict, List, Callable, Optional
from dotenv import dotenv_values
import json
import math
import jsonlines
from dataclasses import dataclass
import utils
import re

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
    api_base = "http://localhost:1234/v1"
    model = "Qwen/Qwen3-Embedding-0.6B"


@dataclass
class Keywords:
    keywords_dir = RESULTS_DIR / "keywords"
    keywords_dir.mkdir(parents=True, exist_ok=True)
    extracted_keywords_path: Path = keywords_dir / "extracted_keywords.jsonl"
    keywords_path: Path = keywords_dir / "keywords.jsonl"

        
    def load() -> Any:
        return utils.load_jsonl_file(Keywords.keywords_path, as_dataframe=True)
    def load_extracted() -> Any:
        return utils.load_jsonl_file(Keywords.extracted_keywords_path)

DATA_TEMPLATE: Callable[[Dict[str, Any]], str] = lambda record: f"<item><name>{record['name']}</name><description>{record['description']}</description></item>"


def reverse_item(s: str) -> Dict[str, str]:
    """Reverse DATA_TEMPLATE: parse <item><name>...</name><description>...</description></item> into dict."""
    name = re.search(r"<name>(.*?)</name>", s).group(1)
    description = re.search(r"<description>(.*?)</description>", s).group(1)
    return {"name": name, "description": description}

# New function: parse a string containing multiple <item>...</item> blocks
def reverse_items(s: str) -> List[Dict[str, str]]:
    """Parse multiple <item>...</item> blocks and return a list of dicts."""
    items = re.findall(r"<item>(.*?)</item>", s, flags=re.DOTALL)
    return [reverse_item(f"<item>{item}</item>") for item in items]

class Template:
    def __init__(self, output_format: str = "html", keys: List[str] = ["name", "description"], type: str = "item") -> None:
        self.format: str = output_format
        self.keys: List[str] = keys
        self.type: str = type

    def __call__(self, record: Dict[str, Any]) -> str:
        if self.format == "html":
            return f"<{self.type}>{''.join([f'<{key}>{record[key]}</{key}>' for key in self.keys])}</{self.type}>"
        elif self.format == "md":
            return f"{''.join([f'**{key}**: {record[key]}\n\n' for key in self.keys])}"
        raise ValueError("Unsupported format")

    def inverse(self, s: str) -> Dict[str, str]:
        results = {}
        for k in self.keys:
            if self.format == "md":
                pat = re.search(rf"\*\*{k}\*\*: (.*?)(?=\n\n|\Z)", s, re.DOTALL)
            elif self.format == "html":
                pat = re.search(rf"<{k}>(.*?)</{k}>", s, re.DOTALL)
            else:
                raise ValueError("Unsupported format")
            if not pat:
                raise ValueError(f"Key {k} not found in the string.")
            val = pat.group(1).strip()
            results[k] = val
        return results
    
@dataclass
class Categories:
    category_dir: Path = RESULTS_DIR / "category"
    category_dir.mkdir(parents=True, exist_ok=True)
    batch_size: int = 2000
    keywords_type: str = None
    categories_path: Path = category_dir / "categories.jsonl"
    # latest_level: int = 0
    # unknown_keywords_path: Path = category_dir / "unknown_keywords.jsonl"
    # missing_keywords_path: Path = category_dir / "missing_keywords.jsonl"


    def load() -> Any:
        return utils.load_jsonl_file(Categories.categories_path, as_dataframe=True)
    def load_proposal() -> Any:
        return utils.load_jsonl_file(Categories.category_proposal_path, as_dataframe=True)
    # def load_unknown_keywords():
    #     return utils.load_jsonl_file(Categories.unknown_keywords_path, as_dataframe=True)
    # def load_missing_keywords():
    #     return utils.load_jsonl_file(Categories.missing_keywords_path, as_dataframe=True)
    # def count_proposal_token():
    #     from tiktoken import SimpleBytePairEncoding
    #     cats = Categories.load_proposal()
    #     enc = SimpleBytePairEncoding.from_tiktoken("cl100k_base")
    #     enc.encode(cats)
        
        
        # from transformers import AutoTokenizer
        # proposals = Categories.load_proposal()
        # return proposals.apply(Categories.template, axis=1).map(AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507").encode).map(len)

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


    def load() -> Any:
        return utils.load_jsonl_file(Grants.grants_path, as_dataframe=True)

    def load_enriched() -> Any:
        return utils.load_json_file(Grants.enriched_path, as_dataframe=True)

    def load_raw() -> Any:
        return utils.load_json_file(Grants.raw_path, as_dataframe=True)

def finished_grants() -> List[Any]:
    keywords = utils.load_jsonl_file(Keywords.keywords_path, as_dataframe=False)
    return [grant for kw in keywords for grant in kw["grants"]]
    keywords = utils.load_jsonl_file(Keywords.keywords_path, as_dataframe=False)
    return [grant for kw in keywords for grant in kw["grants"]]
