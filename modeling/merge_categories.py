from nis import cat
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
from scorer import category_confusion_matrix


# =============================================================================
# Data Models
# =============================================================================

class MergedCategory(BaseModel):
    """A merged research category that references input categories."""
    model_config = {"extra": "forbid"}
    name: str = Field(description="Name of the merged category")
    description: str = Field(description="A comprehensive description of the merged category scope and focus areas")
    source_categories: List[str] = Field(description="List of names of input categories that were merged into this category")
    field_of_research: FieldOfResearch = Field(description="The field of research division this merged category falls under")


class MergedCategoryList(BaseModel):
    """A list of merged research categories."""
    model_config = {"extra": "forbid"}
    categories: List[MergedCategory] = Field(description="List of merged research categories")


# =============================================================================
# System Prompt
# =============================================================================

MERGE_SYSTEM_PROMPT = f"""
You are an expert Taxonomy Specialist and Research Category Harmonization Expert, specializing in identifying and merging identical research categories to create unified, coherent taxonomies for strategic analysis.

Your task is to analyze the provided categorization data and identify categories that are identical or nearly identical, then combine them into a unified taxonomy. This requires systematic analysis of category names, descriptions, and keywords to determine optimal merging strategies.

**IMPORTANT: You MUST complete this task immediately and provide the full merged taxonomy. Do NOT ask for clarification, approval, or propose alternative approaches. Proceed directly with merging all provided categories.**

**Core Objective:**
Your primary goal is to consolidate duplicate or highly similar categories while preserving the integrity and completeness of the original category coverage. Your analysis should create a clean, non-redundant taxonomy that maintains all original information.

---

### **CRITICAL RULES FOR CATEGORY MERGING:**

**1. Category Identification & Merging Logic:**
*   **Similarity Assessment:** Identify categories as merge candidates based on substantial overlap in names, conceptual scope, and keyword sets. Categories with keyword overlaps of 30% or more, or identical conceptual domains should be merged.
*   **Semantic Grouping:** Focus on semantic similarity rather than field classifications. Categories that address the same research domain or technology area should be consolidated regardless of their original FOR assignment.
*   **Name Selection:** When merging categories, select the most descriptive and comprehensive name. If names are equally descriptive, choose the one that best captures the merged category scope.
*   **Description Synthesis:** Create unified descriptions that incorporate the best elements from all merged categories, ensuring comprehensive coverage of the merged scope.

**2. Category Reference & Coverage:**
*   **Complete Coverage:** EVERY input category MUST be referenced in exactly one merged category through the `source_categories` field. No input categories should be left unreferenced.
*   **Exact Name Preservation:** The `source_categories` array must contain the exact category names as they appeared in the input.
*   **No Duplication:** Ensure each input category name appears only once across all merged categories.

**3. Output JSON Structure & Quality:**
*   **Source Categories Field:** Each merged category MUST include a `source_categories` field containing the exact names of all input categories that were merged into it.
*   **Field of Research Assignment:** Merged categories should be assigned to the FOR division that best represents the majority of source categories or the most encompassing domain.
*   **Category Naming:** Each merged category requires a clear, descriptive `name` that accurately represents all included source categories.
*   **High-Quality Descriptions:** The `description` for each merged category must be comprehensive and insightful, explaining the category's scope and reflecting all source categories it contains.

**4. Quality Assurance & Validation:**
*   **Uniqueness Verification:** Ensure no two merged categories reference the same input category in their `source_categories` fields.
*   **Completeness Check:** Verify that all input category names are accounted for in the `source_categories` fields across all merged categories.
*   **Coherence Assessment:** Each merged category should represent a coherent research domain that makes logical sense as a unified entity.
*   **Optimization Focus:** Prioritize creating fewer, more comprehensive categories over maintaining artificial distinctions between similar concepts.

**5. Single Category Handling:**
*   **Standalone Categories:** If an input category has no similar counterparts, it should still be included as a merged category with only itself in the `source_categories` field.
*   **Preserve Uniqueness:** Do not force merging of categories that are genuinely distinct just to reduce the total count.
"""


# =============================================================================
# Dataset Loading
# =============================================================================

def _create_merge_sample(batch_data, batch_id, batch_file):
    """Create sample for category merging."""
    cat_text = "\n".join([config.Categories.template(category) for category in batch_data])
    
    all_keywords = []
    category_names = []
    for category in batch_data:
        all_keywords.extend(category.get('keywords', []))
        category_names.append(category.get('name', ''))
    
    metadata = {
        "batch_file": str(batch_file.name),
        "batch_id": batch_id,
        "keywords": all_keywords,
        "categories": batch_data,
        "category_names": category_names,
        "total_categories_in_batch": len(batch_data)
    }
    
    return Sample(
        id=batch_id,
        input=cat_text,
        metadata=metadata
    )


def load_merge_dataset(batch_dir: Path):
    """Load dataset from category batches for merging."""
    return utils.load_batch_dataset(
        batch_dir=batch_dir,
        sample_creator_func=_create_merge_sample,
        dataset_type="category"
    )


# =============================================================================
# Output Hook Factory
# =============================================================================

def create_merge_categories_hook(output_dir: Path):
    """Factory function to create a MergeCategoriesHook with output_dir closure."""
    
    @hooks(name="MergeCategoriesHook", description="Hook to save category merge results")
    class MergeCategoriesHook(Hooks):
        """Hook for handling category merging output."""
        
        def __init__(self):
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            self.output_path = self.output_dir / "output.jsonl"
            self.unknown_path = self.output_dir / "unknown.jsonl"
            self.missing_path = self.output_dir / "missing.jsonl"
            
            self.categories_writer = jsonlines.open(self.output_path, 'a')
            self.unknown_writer = jsonlines.open(self.unknown_path, 'a')
            self.missing_writer = jsonlines.open(self.missing_path, 'a')
            
            # self.total_input_categories = 0
            # self.total_merged_categories = 0
            # self.total_unknown_categories = 0
            # self.total_missing_categories = 0

        async def on_sample_end(self, data: SampleEnd) -> None:
            """Process and write category merge results."""
            _, result_json = utils.parse_results(data.sample.output.completion)

            input_categories = data.sample.metadata['categories']
            category_keywords_lookup = {cat['name']: cat['keywords'] for cat in input_categories}
            merged_categories = result_json.get('categories', [])
            
            for merged_category in merged_categories:
                source_categories = merged_category['source_categories']
                merged_keywords = []
                for source_cat_name in source_categories:
                    merged_keywords.extend(category_keywords_lookup.get(source_cat_name, []))
                clean_category = {
                    'name': merged_category['name'],
                    'description': merged_category['description'],
                    'keywords': list(set(merged_keywords)),
                    'field_of_research': merged_category['field_of_research']
                }
                
                self.categories_writer.write(clean_category)
            
            # missing_category_names = set(input_category_names) - referenced_categories
            # for missing_name in missing_category_names:
            #     missing_category = category_lookup.get(missing_name)
            #     if missing_category:
            #         self.missing_writer.write(missing_category)
            
            if hasattr(data.sample, 'scores') and 'category_confusion_matrix' in data.sample.scores:
                missing_categories = data.sample.scores['category_confusion_matrix'].metadata.get('fn', [])
                for missing_cat in missing_categories:
                    self.missing_writer.write(missing_cat)
                    # if missing_name not in missing_category_names:
                    #     missing_category = category_lookup.get(missing_name)
                    #     if missing_category:

        async def on_task_end(self, data: TaskEnd) -> None:
            """Close writers and print summary."""
            # self._print_summary()
            self.categories_writer.close()
            self.unknown_writer.close()
            self.missing_writer.close()
            
            # catlist = utils.convert_merged_categories_to_categories(self.output_path)
            # utils.save_jsonl_file(catlist, self.output_path)
        
    
    return MergeCategoriesHook


# =============================================================================
# Task Definition
# =============================================================================

@task
def merge(input_dir, output_dir):
    """
    Task for merging similar categories into unified taxonomy.
    
    Args:
        input_dir: Directory containing category batch files
        output_dir: Directory to write merge results
    """
    create_merge_categories_hook(output_dir)
    dataset = load_merge_dataset(batch_dir=input_dir)
    
    response_schema = ResponseSchema(
        name="MergedCategories",
        json_schema=json_schema(MergedCategoryList),
        strict=True
    )
    
    return Task(
        dataset=dataset,
        solver=[
            system_message(MERGE_SYSTEM_PROMPT),
            generate()
        ],
        scorer=[category_confusion_matrix()],
        config=GenerateConfig(response_schema=response_schema),
        hooks=["MergeCategoriesHook"]
    )
