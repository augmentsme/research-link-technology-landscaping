from pathlib import Path
from typing import Any, Dict, List, Optional
import json


import jsonlines
import pandas as pd

from inspect_ai.model import ModelOutput
import re



from nltk.stem import PorterStemmer
stemmer = PorterStemmer()  # Initialize stemmer once

def normalize(term: str) -> str:
    """
    Normalize keyword terms for deduplication using NLTK stemming.
    
    Args:
        term: The original keyword term
        
    Returns:
        Normalized term for comparison
    """
    # Convert to lowercase and strip whitespace
    normalized = term.lower().strip()

    # Normalize whitespace and remove common separators
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = normalized.replace('-', ' ').replace('_', ' ').replace('/', ' ')
    
    # Remove common punctuation that doesn't affect meaning
    normalized = re.sub(r'[.,;:()[\]{}"]', '', normalized)
    
    # Use NLTK stemming for consistent normalization
    words = normalized.split()
    stemmed_words = []
    
    for word in words:
        stemmed_word = stemmer.stem(word)
        stemmed_words.append(stemmed_word)
    
    stemmed_normalized = ' '.join(stemmed_words)
    
    # Remove extra spaces and return
    return re.sub(r'\s+', ' ', stemmed_normalized).strip()

def load_jsonl_file(path: Path, as_dataframe: bool = False) -> List[Dict[str, Any]]:
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found at {path}")
    
    if as_dataframe:
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            # If pandas fails, fall back to manual parsing
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return pd.DataFrame(data)
    
    # For non-dataframe case, use manual parsing to skip corrupted lines
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def save_jsonl_file(data: List[Dict[str, Any]], path: Path) -> None:
    if isinstance(path, str):
        path = Path(path)
    with jsonlines.open(path, 'w') as f:
        f.write_all(data)

def append_jsonl_file(data: List[Dict[str, Any]], path: Path) -> None:
    if isinstance(path, str):
        path = Path(path)
    with jsonlines.open(path, 'a') as f:
        f.write_all(data)

def load_json_file(path: Path, as_dataframe: bool = False) -> Any:
    """Internal helper to load a JSON file and raise a clear FileNotFoundError."""
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found at {path}")
    with open(path, 'r', encoding='utf-8') as f:
        if as_dataframe:
            return pd.read_json(f)
        return json.load(f)

def save_json_file(data: Any, path: Path) -> None:
    """Save data to a JSON file with consistent formatting.
    
    Args:
        data: The data to save (must be JSON serializable)
        path: Path to the file to save to
    """
    if isinstance(path, str):
        path = Path(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_results(input_string: str, tag="think"):
    pat = f"<{tag}>(.*?)</{tag}>"
    think_re = re.compile(pat, re.DOTALL | re.IGNORECASE)
    m = think_re.search(input_string)
    if m:
        think_text = m.group(1).strip()
        rest = think_re.sub('', input_string).strip()
    else:
        think_text = ""
        rest = input_string.strip()
    return think_text, json.loads(rest)

def sanitise_grant_id(grant_id: str) -> str:
    """Sanitise grant ID for use in filenames."""
    return grant_id.replace("/", "_")

def desanitise_grant_id(grant_id: str) -> str:
    """Reverse the sanitisation of a grant ID."""
    return grant_id.replace("_", "/")


import textwrap

def to_clipboard_with_max_width(series, max_width=80):
    """
    Copy a pandas Series to clipboard with controlled maximum width for each field.
    Long content will be wrapped with newlines instead of truncated.
    
    Args:
        series: pandas Series to copy
        max_width: maximum width for each line (default: 80)
    """
    # Create a copy to avoid modifying original
    series_copy = series.copy()
    
    # Wrap long strings with newlines
    for key, value in series_copy.items():
        if isinstance(value, str) and len(value) > max_width:
            # Use textwrap to break long strings into multiple lines
            wrapped = textwrap.fill(value, width=max_width, break_long_words=False, break_on_hyphens=False)
            series_copy[key] = wrapped
        elif isinstance(value, list):
            # Handle lists by converting to string and wrapping if needed
            str_value = str(value)
            if len(str_value) > max_width:
                wrapped = textwrap.fill(str_value, width=max_width, break_long_words=False, break_on_hyphens=False)
                series_copy[key] = wrapped
    
    # Copy to clipboard
    series_copy.to_clipboard()
    return series_copy

def get_keywords_freq_table_with_selector(keywords, grants, selector):
    return keywords.explode("grants")[keywords.explode("grants").grants.isin(grants[selector].id)].groupby("name").agg({"grants": "count"}).sort_values("grants", ascending=False).reset_index().merge(keywords, how="left", left_on="name", right_on="name").drop("grants_y", axis=1).rename(columns={"grants_x": "grants"})