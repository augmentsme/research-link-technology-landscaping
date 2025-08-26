from pathlib import Path
from dotenv import dotenv_values
import json
import math

CONFIG = dotenv_values()
ROOT_DIR = Path(CONFIG["ROOT_DIR"])
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"

# DATA_DIR = Path(CONFIG["DATA_DIR"]).resolve()
# RESULTS_DIR = Path(CONFIG["RESULTS_DIR"]).resolve()
# FIGURES_DIR = Path(CONFIG["FIGURES_DIR"]).resolve()



GRANTS_FILE = DATA_DIR / "grants_cleaned.json"
PROMPTS_DIR = ROOT_DIR / "PromptTemplates"


EXTRACTED_KEYWORDS_DIR = RESULTS_DIR / "extract"
EXTRACTED_KEYWORDS_PATH = RESULTS_DIR / "extracted_keywords.json"



KEYWORDS_EMBEDDING_DBPATH = RESULTS_DIR / "keywords_embeddings"
SIMILARITY_THRESHOLD = 0.75
MIN_CLUSTER_SIZE = 5



CLUSTERS_PROPOSAL_PATH = RESULTS_DIR / "clusters_proposal.json"
REVIEW_FILE = RESULTS_DIR / "review.json"
CLUSTERS_FINAL_PATH = RESULTS_DIR / "clusters_final.json"

KEYWORDS_PATH = RESULTS_DIR / "keywords.json"


BATCH_SIZE = 100
KEYWORDS_TYPE = None

CATEGORY_PROPOSAL_PATH = RESULTS_DIR / "category_proposal.json"
CATEGORY_PATH = RESULTS_DIR / "categories.json"
# CATEGORY_DIR = RESULTS_DIR / "categories"

COARSENED_CATEGORY_PATH = RESULTS_DIR / "coarsened_categories.json"
REFINED_CATEGORY_PATH = RESULTS_DIR / "refined_categories.json"
COMPREHENSIVE_TAXONOMY_PATH = RESULTS_DIR / "comprehensive_taxonomy.json"
FOR_CODES_CLEANED_PATH = DATA_DIR / "for_codes_cleaned.json"

CLASSIFICATION_PATH = RESULTS_DIR / "classification.json"


