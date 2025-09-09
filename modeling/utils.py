from pathlib import Path
from typing import Any, Dict, List, Optional
import json


import jsonlines
import pandas as pd

from inspect_ai.model import ModelOutput
import re


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
