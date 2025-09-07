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


class Template:
    def __init__(self, output_format: str = "html", keys: List[str] = ["name", "description"], type: str = "item") -> None:
        self.format: str = output_format
        self.keys: List[str] = keys
        self.type: str = type

    def __call__(self, record: Dict[str, Any]) -> str:
        if self.format == "html":
            return f"<{self.type}>{''.join([f'<{key}>{record[key]}</{key}>' for key in self.keys])}</{self.type}>"
        elif self.format == "md":
            return f"{''.join([f'**{key}**: {record[key]}\n' for key in self.keys])}"
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
class Keywords:
    keywords_dir = RESULTS_DIR / "keywords"
    keywords_dir.mkdir(parents=True, exist_ok=True)
    extracted_keywords_path: Path = keywords_dir / "extracted_keywords.jsonl"
    keywords_path: Path = keywords_dir / "keywords.jsonl"
    def load() -> Any:
        return utils.load_jsonl_file(Keywords.keywords_path, as_dataframe=True)
    def load_extracted() -> Any:
        return utils.load_jsonl_file(Keywords.extracted_keywords_path)
    def finished_grants() -> List[Any]:
        keywords = utils.load_jsonl_file(Keywords.keywords_path, as_dataframe=False)
        return [grant for kw in keywords for grant in kw["grants"]]

    
@dataclass
class Categories:
    category_dir: Path = RESULTS_DIR / "category"
    category_dir.mkdir(parents=True, exist_ok=True)
    # batch_size: int = 100
    def load(level=None) -> Any:
        if level is None:
            level = Categories.get_max_level()
        return utils.load_jsonl_file(Categories.category_dir / f"{level}.jsonl", as_dataframe=True)
    def get_max_level():
        levels = []
        for file in (Categories.category_dir).glob("*.jsonl"):
            if file.stem.isnumeric():
                levels.append(int(file.stem))
        return max(levels)
    
    def get_available_mappings() -> List[tuple]:
        """Get list of available mapping files.
        
        Returns:
            List of tuples (source_level, target_level) for available mappings
        """
        mappings = []
        for file in Categories.category_dir.glob("*.mapping.jsonl"):
            # File format: "source-target.mapping.jsonl"
            stem = file.stem.replace(".mapping", "")
            if "-" in stem:
                try:
                    source_level, target_level = stem.split("-")
                    mappings.append((int(source_level), int(target_level)))
                except ValueError:
                    continue
        return sorted(mappings)
    
    def has_mapping(level=None, source_level=None) -> bool:
        """Check if a mapping file exists for the given levels.
        
        Args:
            level: Target level
            source_level: Source level (defaults to level-1)
        
        Returns:
            True if mapping file exists, False otherwise
        """
        if level is None:
            level = Categories.get_max_level()
        if source_level is None:
            source_level = level - 1 if level > 0 else 0
        
        mapping_path = Categories.category_dir / f"{source_level}-{level}.mapping.jsonl"
        return mapping_path.exists()
    
    def load_mapping(level=None, source_level=None) -> Any:
        """Load level mapping data with standardized column names.
        
        Args:
            level: Target level to load mappings for
            source_level: Source level (defaults to level-1)
        
        Returns:
            DataFrame with columns ['source_item', 'target_item']
        """
        if level is None:
            level = Categories.get_max_level()
        if source_level is None:
            source_level = level - 1 if level > 0 else 0
        
        mapping_path = Categories.category_dir / f"{source_level}-{level}.mapping.jsonl"
        
        df = utils.load_jsonl_file(mapping_path, as_dataframe=True)
        df = df.rename(columns={str(source_level): "source_item", str(level): "target_item"})
        return df



@dataclass
class Grants:
    neo4j_uri = "bolt://localhost:7687"
    neo4j_username = "neo4j"
    cipher_query = "MATCH (g:grant) RETURN g"
    raw_path = DATA_DIR / "grants_raw.json"
    enriched_path = DATA_DIR / "grants_enriched.json"
    grants_path = DATA_DIR / "grants.jsonl"
    neo4j_password = CONFIG.get("NEO4J_PASSWORD")
    template = lambda record: f"<grant><title>{record['title']}</title><description>{record['grant_summary']}</description></grant>"


    def load() -> Any:
        return utils.load_jsonl_file(Grants.grants_path, as_dataframe=True)

    def load_enriched() -> Any:
        return utils.load_json_file(Grants.enriched_path, as_dataframe=True)

    def load_raw() -> Any:
        return utils.load_json_file(Grants.raw_path, as_dataframe=True)


