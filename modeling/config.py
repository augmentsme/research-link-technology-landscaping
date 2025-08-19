from pathlib import Path
# Configuration constants - hardcoded to remove config.yaml dependency
ROOT_DIR = Path("/Users/luhancheng/Desktop/research-link-technology-landscaping/modeling")
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
PROMPTS_DIR = ROOT_DIR / "PromptTemplates"
GRANTS_FILE = DATA_DIR / "active_grants.json"
RESULTS_DIR = LOGS_DIR / "results"

# output files
KEYWORDS_PATH = RESULTS_DIR / "keywords.json"
CATEGORY_PATH = RESULTS_DIR / "categories.json"
REFINED_CATEGORY_PATH = RESULTS_DIR / "refined_categories.json"
