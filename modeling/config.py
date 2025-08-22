from pathlib import Path
from dotenv import dotenv_values

CONFIG = dotenv_values()
ROOT_DIR = Path(CONFIG["ROOT_DIR"])
DATA_DIR = Path(CONFIG["DATA_DIR"])
RESULTS_DIR = Path(CONFIG["RESULTS_DIR"])


EXTRACTED_KEYWORDS_DIR = RESULTS_DIR / "extract"
EXTRACTED_KEYWORDS_PATH = RESULTS_DIR / "keywords.json"



PROMPTS_DIR = ROOT_DIR / "PromptTemplates"
GRANTS_FILE = DATA_DIR / "grants_cleaned.json"


SIMILARITY_THRESHOLD = 0.9
CLUSTERS_PROPOSAL_PATH = RESULTS_DIR / "clusters_proposal.json"
REVIEW_FILE = RESULTS_DIR / "review.json"
CLUSTERS_FINAL_PATH = RESULTS_DIR / "clusters_final.json"
KEYWORDS_FINAL_PATH = RESULTS_DIR / "keywords_final.json"



NUM_KEYWORDS_PER_CATEGORY = 100
SAMPLE_SIZE = 1000
KEYWORDS_TYPE = "keywords"
NUM_SAMPLES = 10




CATEGORY_PATH = RESULTS_DIR / "categories.json"
CATEGORY_DIR = RESULTS_DIR / "categories"

COARSENED_CATEGORY_PATH = RESULTS_DIR / "coarsened_categories.json"
REFINED_CATEGORY_PATH = RESULTS_DIR / "refined_categories.json"
COMPREHENSIVE_TAXONOMY_PATH = RESULTS_DIR / "comprehensive_taxonomy.json"
FOR_CODES_CLEANED_PATH = DATA_DIR / "for_codes_cleaned.json"

CLASSIFICATION_PATH = RESULTS_DIR / "classification.json"