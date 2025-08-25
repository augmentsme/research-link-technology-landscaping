from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from config import CATEGORY_PATH, CLASSIFICATION_PATH, KEYWORDS_PATH, GRANTS_FILE


def _load_json_file(path: Path) -> Any:
    """Internal helper to load a JSON file and raise a clear FileNotFoundError."""
    if not path.exists():
        raise FileNotFoundError(f"File not found at {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_categories(category_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load categories data (returns list of category dicts).

    Args:
        category_path: Path to categories file; defaults to config.CATEGORY_PATH

    Returns:
        List of category dictionaries. If the file is a dict with a 'categories'
        key, that value is returned; otherwise the raw JSON object is returned
        (expected to be a list).
    """
    if category_path is None:
        category_path = CATEGORY_PATH

    data = _load_json_file(category_path)
    if isinstance(data, dict):
        return data.get('categories', [])
    return data


def load_classification_results(classification_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load classification results JSON.

    Args:
        classification_path: Path to classification results file; defaults to config.CLASSIFICATION_PATH

    Returns:
        Parsed JSON (expected to be a list of classification result dicts).
    """
    if classification_path is None:
        classification_path = CLASSIFICATION_PATH

    return _load_json_file(classification_path)


def load_keywords(keywords_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load keywords JSON.

    Args:
        keywords_path: Path to keywords file; defaults to config.KEYWORDS_PATH

    Returns:
        Parsed JSON (expected to be a list of keyword dicts).
    """
    if keywords_path is None:
        keywords_path = KEYWORDS_PATH

    return _load_json_file(keywords_path)


def load_grants(grants_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load grants JSON.

    Args:
        grants_path: Path to grants file; defaults to config.GRANTS_FILE

    Returns:
        Parsed JSON (expected to be a list of grant dicts).
    """
    if grants_path is None:
        grants_path = GRANTS_FILE

    return _load_json_file(grants_path)
