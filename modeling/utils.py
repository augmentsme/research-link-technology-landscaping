from config import CONFIG
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


                    CONFIG.keywords_path)


def _load_json_file(path: Path) -> Any:
    """Internal helper to load a JSON file and raise a clear FileNotFoundError."""
    if not path.exists():
        raise FileNotFoundError(f"File not found at {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_categories(category_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load categories data (returns list of category dicts).

    Args:
        category_path: Path to categories file; defaults to config.CONFIG.category_path

    Returns:
        List of category dictionaries. If the file is a dict with a 'categories'
        key, that value is returned; otherwise the raw JSON object is returned
        (expected to be a list).
    """
    if category_path is None:
        category_path = CONFIG.category_path

    data = _load_json_file(category_path)
    if isinstance(data, dict):
        return data.get('categories', [])
    return data


def load_classification_results(classification_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load classification results JSON.

    Args:
        classification_path: Path to classification results file; defaults to config.CONFIG.classification_path

    Returns:
        Parsed JSON (expected to be a list of classification result dicts).
    """
    if classification_path is None:
        classification_path = CONFIG.classification_path

    return _load_json_file(classification_path)


def load_keywords(keywords_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load keywords JSON.

    Args:
        keywords_path: Path to keywords file; defaults to config.CONFIG.keywords_path

    Returns:
        Parsed JSON (expected to be a list of keyword dicts).
    """
    if keywords_path is None:
        keywords_path = CONFIG.keywords_path

    return _load_json_file(keywords_path)


def load_grants(grants_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load grants JSON.

    Args:
        grants_path: Path to grants file; defaults to config.CONFIG.grants_file

    Returns:
        Parsed JSON (expected to be a list of grant dicts).
    """
    if grants_path is None:
        grants_path = CONFIG.grants_file

    return _load_json_file(grants_path)
