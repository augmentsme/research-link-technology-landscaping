from inspect_ai import Task, task
from itertools import batched
import shutil

import jsonlines
import config
from pathlib import Path
import utils
import pandas as pd
import numpy as np
from models import Category, CategoryList
from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks
from inspect_ai.dataset import json_dataset, FieldSpec, Sample, MemoryDataset
import json
from inspect_ai.solver import system_message, generate
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.util import json_schema

import json 
from scorer import keywords_confusion_matrix


SYSTEM_PROMPT = f"""
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





def keywords_to_sample(keywords, idx):
    entries = []
    for kw in keywords:
        entries.append(config.Keywords.template(kw))
    return Sample(id=idx, input="\n".join(entries), metadata={"keywords": [kw['name'] for kw in keywords]})

def load_dataset(keywords_path: Path, batch_size: int):
    keywords = utils.load_jsonl_file(keywords_path, as_dataframe=False)
    keywords = sorted(keywords, key=lambda x: x['type'])
    samples = list(map(keywords_to_sample, batched(keywords, n=batch_size), range(len(keywords)//batch_size + 1)))
    return MemoryDataset(samples)



def create_keywords_file_for_next_iteration(output_path: Path = config.Categories.category_dir / "missing_unknown_keywords.jsonl"):
    
    # Load original keywords to get descriptions
    original_keywords = utils.load_jsonl_file(config.Keywords.keywords_path, as_dataframe=False)
    keywords_dict = {kw['name']: kw for kw in original_keywords}
    
    # Load unknown and missing keywords (these are just strings in jsonl format)
    def load_string_jsonl(file_path):
        keywords = []
        if file_path.exists():
            with open(file_path, 'r') as f:
                for i, line in enumerate(f, 1):
                    try:
                        keyword = json.loads(line.strip())
                        keywords.append(keyword)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed line {i} in {file_path}: {e}")
        return keywords
    
    unknown_keywords = load_string_jsonl(config.Categories.unknown_keywords_path)
    missing_keywords = load_string_jsonl(config.Categories.missing_keywords_path)
    
    # Combine and deduplicate
    all_keyword_names = set(unknown_keywords + missing_keywords)
    
    # Create new keywords list with descriptions
    next_iteration_keywords = []
    not_found_keywords = []
    
    for keyword_name in all_keyword_names:
        if keyword_name in keywords_dict:
            next_iteration_keywords.append(keywords_dict[keyword_name])
        else:
            not_found_keywords.append(keyword_name)
    
    # Save to file
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    utils.save_jsonl_file(next_iteration_keywords, output_path)
    
    if not_found_keywords:
        print(f"Warning: {len(not_found_keywords)} keywords not found in original keywords file")
        if len(not_found_keywords) <= 10:
            print(f"Not found keywords: {not_found_keywords}")
        else:
            print(f"First 10 not found keywords: {not_found_keywords[:10]}")
    
    print(f"Successfully created keywords file with {len(next_iteration_keywords)} keywords at {output_path}")
    return len(next_iteration_keywords)



@hooks(name="CategoriseOutputHook", description="Hook to save categorisation results as JSON files")
class CategoriseOutputHook(Hooks):
    def __init__(self):
        self.output_write = jsonlines.open(config.Categories.category_proposal_path, 'a')
        self.unknown_write = jsonlines.open(config.Categories.unknown_keywords_path, 'a')
        self.missing_write = jsonlines.open(config.Categories.missing_keywords_path, 'a')
    async def on_sample_end(self, data: SampleEnd) -> None:
        """Save keywords extraction results."""

        _, result_json = utils.parse_results(data.sample.output.completion)
        self.output_write.write_all(result_json['categories'])
        self.missing_write.write_all(data.sample.scores['keywords_confusion_matrix'].metadata['fn'])
        for category in result_json['categories']:
            if category['name'] == "Unknown":
                self.unknown_write.write_all(category['keywords'])

    async def on_task_end(self, data: TaskEnd) -> None:
        """Aggregate and save all categorisation results at the end of the task."""
        self.output_write.close()
        self.unknown_write.close()
        self.missing_write.close()



@task
def categorise(keywords_path = config.Keywords.keywords_path, batch_size: int = 250):

    dataset = load_dataset(keywords_path=keywords_path, batch_size=batch_size)

    return Task(
        dataset=dataset,
        solver=[
            system_message(SYSTEM_PROMPT),
            generate(),
        ],
        scorer=[
            keywords_confusion_matrix()
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="Categories",
                json_schema=json_schema(CategoryList),
                strict=True
            ),
        ),
        hooks=["CategoriseOutputHook"]
    )


