from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

ROOT_DIR = Path(os.getenv('ROOT_DIR', 
						  '/Users/chengluhan/research-link-technology-landscaping/modeling'))
DATA_DIR = Path(os.getenv('DATA_DIR', str(ROOT_DIR / 'data')))
RESULTS_DIR = Path(os.getenv('RESULTS_DIR', str(ROOT_DIR / 'results')))
LOGS_DIR = Path(os.getenv('LOGS_DIR', str(ROOT_DIR / 'logs')))

# Paths for prompts and data derived from the above
PROMPTS_DIR = ROOT_DIR / 'PromptTemplates'
GRANTS_FILE = DATA_DIR / 'grants_cleaned.json'

# output files
KEYWORDS_PATH = RESULTS_DIR / 'keywords.json'
TERMS_PATH = RESULTS_DIR / 'terms.json'  # Harmonized terms (clusters and categorise-compatible format)
CATEGORY_PATH = RESULTS_DIR / 'categories.json'
REFINED_CATEGORY_PATH = RESULTS_DIR / 'refined_categories.json'
FOR_CODES_CLEANED_PATH = DATA_DIR / 'for_codes_cleaned.json'

# review output path
REVIEW_FILE = RESULTS_DIR / 'harmonization_review.json'
