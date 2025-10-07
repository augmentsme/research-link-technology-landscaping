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
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"


@dataclass
class Keywords:
    keywords_dir = RESULTS_DIR / "keywords"
    keywords_dir.mkdir(parents=True, exist_ok=True)
    extracted_keywords_path: Path = keywords_dir / "extracted_keywords.jsonl"
    keywords_path: Path = keywords_dir / "keywords.jsonl"
    
    template = lambda record: f"<keyword><name>{record['name']}</name><type>{record['type']}</type><description>{record['description']}</description></keyword>"
    
    def load(as_dataframe=True):
        return utils.load_jsonl_file(Keywords.keywords_path, as_dataframe=as_dataframe)
    
    def load_extracted(as_dataframe=True):
        return utils.load_jsonl_file(Keywords.extracted_keywords_path, as_dataframe=as_dataframe)
    


@dataclass
class Categories:

    CATEGORIY_PATH = RESULTS_DIR / "categories.jsonl"
    template = lambda record: f"<category><name>{record['name']}</name><description>{record['description']}</description><keywords>{','.join(record.get('keywords', []))}</keywords></category>"
    
    def last_merged():
        return max([int(i.stem) for i in RESULTS_DIR.iterdir() if i.stem.isdigit()])
    def last_merged_path():
        return RESULTS_DIR / str(Categories.last_merged()) / "output.jsonl"
    def load_last_merged():
        return utils.load_jsonl_file(Categories.last_merged_path(), as_dataframe=True)

    def load(as_dataframe=True):
        return utils.load_jsonl_file(Categories.CATEGORIY_PATH, as_dataframe=as_dataframe)

@dataclass
class Grants:
    grants_path = ROOT_DIR / "grants.json"

    template = staticmethod(
        lambda record: (
            f"<grant><title>{record.get('title') or record.get('grant_title', '')}</title>"
            f"<description>{record.get('grant_summary') or record.get('summary', '')}</description></grant>"
        )
    )

    @staticmethod
    def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        if "funding_amount" in df.columns:
            df["funding_amount"] = pd.to_numeric(df["funding_amount"], errors="coerce")

        if "start_date" in df.columns and "start_year" not in df.columns:
            df["start_year"] = pd.to_datetime(df["start_date"], errors="coerce").dt.year.astype("Int64")
        elif "start_year" in df.columns:
            df["start_year"] = pd.to_numeric(df["start_year"], errors="coerce").astype("Int64")

        if "end_date" in df.columns and "end_year" not in df.columns:
            df["end_year"] = pd.to_datetime(df["end_date"], errors="coerce").dt.year.astype("Int64")
        elif "end_year" in df.columns:
            df["end_year"] = pd.to_numeric(df["end_year"], errors="coerce").astype("Int64")

        return df

    @staticmethod
    def _load_source_records() -> List[Dict[str, Any]]:
        return utils.load_json_file(Grants.grants_path, as_dataframe=False)

    @staticmethod
    def load(as_dataframe: bool = True):
        records = Grants._load_source_records()
        if as_dataframe:
            df = pd.DataFrame(records)
            return Grants._normalize_dataframe(df)
        return records

    