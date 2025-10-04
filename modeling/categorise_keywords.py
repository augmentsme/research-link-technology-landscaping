from inspect_ai import Task, task
from pathlib import Path
from typing import List
import jsonlines

import config
import utils
from pydantic import BaseModel, Field
from models import FieldOfResearch
from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks
from inspect_ai.dataset import Sample
from inspect_ai.solver import system_message, generate
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.util import json_schema
from scorer import keywords_confusion_matrix


# =============================================================================
# Data Models
# =============================================================================

class Category(BaseModel):
    """A flexible research category linked to FOR codes."""
    model_config = {"extra": "forbid"}
    name: str = Field(description="Name of the category")
    description: str = Field(description="A few sentences describing what this category is about, including its scope, focus areas, and the types of research or technologies it encompasses")
    keywords: List[str] = Field(description="List of keywords associated with this category")
    field_of_research: FieldOfResearch = Field(description="The field of research division this category falls under")


class CategoryList(BaseModel):
    """A list of research categories."""
    model_config = {"extra": "forbid"}
    categories: List[Category] = Field(description="List of research categories")


# =============================================================================
# System Prompt
# =============================================================================

CATEGORISE_SYSTEM_PROMPT = f"""
You are an expert Technology Analyst and Innovation Forecaster, specializing in synthesizing information to identify and map emerging technological domains for strategic planning.

Your task is to analyze a list of user-provided keywords and generate a comprehensive set of research and technology categories in a specific JSON format. Each category must be linked to an appropriate FOR (Fields of Research) division.

**IMPORTANT: You MUST complete this task immediately and provide the full categorization. Do NOT ask for clarification, approval, or propose alternative approaches. Proceed directly with categorizing all provided keywords.**

**Input Format:**
Each keyword will be provided in the format: `**keyword_term** (type): description`

**Core Objective:**
Your primary goal is to organize the provided keywords into meaningful, emergent categories that bridge the specificity of the keywords with the broadness of the 23 top-level FOR divisions. Your analysis should favor the identification of potential breakthroughs and new interdisciplinary fields.

---

### **CRITICAL RULES FOR CATEGORIZATION:**

**1. Category Creation & Granularity:**
*   **Natural Groupings:** Create categories based on the natural, thematic relationships between the keywords. The number of categories should be determined by these organic groupings, not a predefined target.
*   **Optimal Abstraction:** Categories should be more specific than the broad FOR divisions but more general than individual keywords. They should represent recognizable research areas.
*   **Focus on Emergence:** Actively look for intersections between keywords that suggest new or interdisciplinary domains. When in doubt, err on the side of creating a new, more specific category to avoid missing potential innovations.

**2. Keyword Handling & Coverage:**
*   **Complete Coverage:** EVERY keyword term provided by the user MUST be included in the `keywords` list of exactly one category. No keywords should be left uncategorized.
*   **No Placeholder Categories:** Do NOT create categories like "Clarification Required" or similar placeholder responses. Only create real, meaningful research categories.
*   **Contextual Analysis:** Use the keyword `(type)` and `description` to understand the context and make more informed grouping decisions.
*   **Exact Term Usage:** The `keywords` array within each category must contain the exact `keyword_term` strings from the input.

**3. Output JSON Structure & Content:**
*   **Field of Research Assignment:** Every category MUST be assigned to exactly ONE of the 23 field of research divisions using the descriptive enum name (e.g., "INFORMATION_COMPUTING_SCIENCES", "PHYSICAL_SCIENCES", "ENGINEERING"). The field_of_research field should contain the full enum name as a string.
*   **Available field of research Divisions:** AGRICULTURAL_VETERINARY_FOOD_SCIENCES, BIOLOGICAL_SCIENCES, BIOMEDICAL_CLINICAL_SCIENCES, BUILT_ENVIRONMENT_DESIGN, CHEMICAL_SCIENCES, COMMERCE_MANAGEMENT_TOURISM_SERVICES, CREATIVE_ARTS_WRITING, EARTH_SCIENCES, ECONOMICS, EDUCATION, ENGINEERING, ENVIRONMENTAL_SCIENCES, HEALTH_SCIENCES, HISTORY_HERITAGE_ARCHAEOLOGY, HUMAN_SOCIETY, INDIGENOUS_STUDIES, INFORMATION_COMPUTING_SCIENCES, LANGUAGE_COMMUNICATION_CULTURE, LAW_LEGAL_STUDIES, MATHEMATICAL_SCIENCES, PHILOSOPHY_RELIGIOUS_STUDIES, PHYSICAL_SCIENCES, PSYCHOLOGY
*   **Category Naming:** Each category requires a short, descriptive `name` (ideally under 4 words) that captures its core focus.
*   **High-Quality Descriptions:** The `description` for each category must be detailed and insightful. It should explain the category's scope and key focus areas, directly reflecting the keywords it contains.

**4. MANDATORY "Unknown" Category for Outliers:**
*   **Creation:** You MUST create a category with the exact name `Unknown`. This category serves as a container for any keywords that are true outliers.
*   **Condition for Use:** Place a keyword in the `Unknown` category ONLY if, after careful analysis, it cannot be logically grouped with any other keywords to form a coherent, thematic category. This is the designated place for single, disparate concepts that do not fit elsewhere.
*   **Description Requirement:** The `description` for the `Unknown` category must explicitly state: "This category contains disparate keywords and technologies that do not fit into the other defined domains and represent potential standalone areas of research."
*   **Field of Research Assignment:** For the `Unknown` category, analyze the keywords placed within it and assign the field of research division that represents the best possible fit for the majority of these keywords, or select "HUMAN_SOCIETY" if no clear fit emerges as it serves as the most general multidisciplinary division.
"""


# =============================================================================
# Dataset Loading
# =============================================================================

def _create_keyword_sample(batch_data, batch_id, batch_file):
    """Create sample for keyword categorization."""
    entries = [config.Keywords.template(kw) for kw in batch_data]
    
    return Sample(
        id=batch_id,
        input="\n".join(entries),
        metadata={"keywords": [kw['name'] for kw in batch_data]}
    )


def load_categorise_dataset(batch_dir: Path):
    """Load dataset from semantic clustering batches for keyword categorization."""
    return utils.load_batch_dataset(
        batch_dir=batch_dir,
        sample_creator_func=_create_keyword_sample,
        dataset_type="keyword"
    )


# =============================================================================
# Output Hook Factory
# =============================================================================

def create_keyword_categorise_hook(output_dir: Path):
    """Factory function to create a KeywordCategoriseHook with output_dir closure."""
    
    @hooks(name="KeywordCategoriseHook", description="Hook to save keyword categorization results")
    class KeywordCategoriseHook(Hooks):
        """Hook for handling keyword categorization output."""
        
        def __init__(self):
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            self.output_path = self.output_dir / "output.jsonl"
            self.unknown_path = self.output_dir / "unknown.jsonl"
            self.missing_path = self.output_dir / "missing.jsonl"
            
            self.categories_writer = jsonlines.open(self.output_path, 'a')
            self.unknown_writer = jsonlines.open(self.unknown_path, 'a')
            self.missing_writer = jsonlines.open(self.missing_path, 'a')

        async def on_sample_end(self, data: SampleEnd) -> None:
            """Process and write keyword categorization results."""
            _, result_json = utils.parse_results(data.sample.output.completion)
            
            categories = result_json.get('categories', [])
            
            for category in categories:
                category_keywords = set(category.get('keywords', []))
                
                if category.get('name') == "Unknown":
                    for keyword in category_keywords:
                        self.unknown_writer.write(keyword)
                else:
                    self.categories_writer.write(category)
            
            if hasattr(data.sample, 'scores') and 'keywords_confusion_matrix' in data.sample.scores:
                missing_keywords = data.sample.scores['keywords_confusion_matrix'].metadata.get('fn', [])
                for keyword in missing_keywords:
                    self.missing_writer.write(keyword)

        async def on_task_end(self, data: TaskEnd) -> None:

            self.categories_writer.close()
            self.unknown_writer.close()
            self.missing_writer.close()
        

    return KeywordCategoriseHook


# =============================================================================
# Task Definition
# =============================================================================

@task
def categorise(input_dir, output_dir):
    """
    Task for categorizing keywords into research categories.
    
    Args:
        input_dir: Directory containing keyword batch files
        output_dir: Directory to write categorization results
    """
    create_keyword_categorise_hook(output_dir)
    dataset = load_categorise_dataset(batch_dir=input_dir)
    
    response_schema = ResponseSchema(
        name="Categories",
        json_schema=json_schema(CategoryList),
        strict=True
    )
    
    return Task(
        dataset=dataset,
        solver=[
            system_message(CATEGORISE_SYSTEM_PROMPT),
            generate()
        ],
        scorer=[keywords_confusion_matrix()],
        config=GenerateConfig(response_schema=response_schema),
        hooks=["KeywordCategoriseHook"]
    )
