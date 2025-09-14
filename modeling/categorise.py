from inspect_ai import Task, task
from itertools import batched
import shutil
from typing import List, Dict, Any
import jsonlines
import config
from pathlib import Path
import utils
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from models import FieldOfResearch
from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks
from inspect_ai.dataset import json_dataset, FieldSpec, Sample, MemoryDataset
import json
from inspect_ai.solver import system_message, generate, self_critique
from inspect_ai.model import GenerateConfig, ResponseSchema, get_model
from inspect_ai.util import json_schema
from semantic_clustering import DatasetBuilder

import json 
from scorer import keywords_confusion_matrix, category_confusion_matrix

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





def _load_batch_dataset_generic(
    batch_dir: Path, 
    output_file_path: Path, 
    batch_id_prefix: str,
    sample_creator_func,
    dataset_type: str,
    mode: str = "keywords"
) -> MemoryDataset:
    """
    Generic function to load batch datasets with resume capability using dataset.filter().
    
    Args:
        batch_dir: Directory containing batch files
        output_file_path: Path to check for processed batches
        batch_id_prefix: Prefix for batch IDs ("semantic_batch_" or "merge_batch_")
        sample_creator_func: Function to create Sample from batch data and batch_id
        dataset_type: Type description for error messages ("keyword" or "category")
    """
    if not batch_dir.exists():
        raise FileNotFoundError(
            f"{dataset_type.capitalize()} batches directory not found at {batch_dir}. "
            f"Please run 'python semantic_clustering.py {dataset_type}s full-pipeline' first to generate batches."
        )
    
    try:
        from inspect_ai.dataset import Sample, MemoryDataset
        
        # Get all batch files
        batch_files = sorted(batch_dir.glob("*batch_*.jsonl"))
        if not batch_files:
            raise FileNotFoundError(f"No batch files found in {batch_dir}")
        
        # Create all samples first
        samples = []
        total_batches = 0
        
        for batch_file in batch_files:
            total_batches += 1
            if batch_id_prefix == "merge_batch_":
                batch_id = f"{batch_id_prefix}{batch_file.stem}"
            else:
                batch_id = f"{batch_id_prefix}{batch_file.stem.split('_')[-1]}"
            
            # Load data from this batch
            try:
                batch_data = utils.load_jsonl_file(batch_file, as_dataframe=False)
                sample = sample_creator_func(batch_data, batch_id, batch_file)
                samples.append(sample)
            except Exception as e:
                print(f"Warning: Could not load batch file {batch_file}: {e}")
                continue

        # Create initial dataset with all samples
        full_dataset = MemoryDataset(samples)
        
        # Get processed batch IDs and filter them out
        processed_batch_ids = get_processed_batch_ids(output_file_path, mode)
        
        # Filter out already processed batches
        filtered_dataset = full_dataset.filter(lambda sample: sample.id not in processed_batch_ids)
        
        print(f"Total {dataset_type} batches: {total_batches}")
        print(f"Already processed: {len(processed_batch_ids)}")
        print(f"Remaining to process: {len(filtered_dataset)}")
        
        return filtered_dataset
        
    except ImportError:
        raise ImportError("Could not import required modules from semantic_clustering")


def _create_keyword_sample(batch_data, batch_id, batch_file):
    """Create sample for keyword categorization."""
    entries = []
    for kw in batch_data:
        entries.append(config.Keywords.template(kw))
    
    return Sample(
        id=batch_id,
        input="\n".join(entries),
        metadata={"keywords": [kw['name'] for kw in batch_data]}
    )


def _create_merge_sample(batch_data, batch_id, batch_file):
    """Create sample for category merging."""
    # Format categories for merging using the Categories template
    cat_text = "\n".join([config.Categories.template(category) for category in batch_data])
    
    # Extract all keywords from this batch for metadata
    all_keywords = []
    category_names = []
    for category in batch_data:
        all_keywords.extend(category.get('keywords', []))
        category_names.append(category.get('name', ''))
    
    metadata = {
        "batch_file": str(batch_file.name),
        "batch_id": batch_id,
        "keywords": all_keywords,
        "categories": batch_data,  # Include full category data for scorer
        "category_names": category_names,  # Include category names for easier access
        "total_categories_in_batch": len(batch_data)
    }
    
    return Sample(
        id=batch_id,
        input=cat_text,
        metadata=metadata
    )


def get_processed_batch_ids(output_file_path: Path, mode: str = "keywords") -> set:
    """
    Dynamically determine which batches have already been processed by examining output files
    and checking which items from each batch have been processed.
    
    Args:
        output_file_path: Path to the output file containing processed items
        mode: Operating mode ("keywords" or "merge")
        
    Returns:
        Set of batch IDs that have already been processed
    """
    processed_batch_ids = set()
    
    # Skip if output file doesn't exist yet
    if not output_file_path.exists():
        return processed_batch_ids
    
    try:
        # Load processed items from output file
        processed_items = utils.load_jsonl_file(output_file_path, as_dataframe=False)
        if not processed_items:
            return processed_batch_ids
        
        # Extract processed item identifiers
        if mode == "merge":
            # For merge mode, get category names
            processed_names = {item.get('name') for item in processed_items if item.get('name')}
        else:
            # For keywords mode, get all keywords from all categories
            processed_keywords = set()
            for item in processed_items:
                if 'keywords' in item:
                    processed_keywords.update(item['keywords'])
        
        # Get the appropriate batch directory
        if mode == "merge":
            batch_dir = config.Categories.batch_dir
            batch_pattern = "*batch_*.jsonl"
        else:
            batch_dir = config.Keywords.batch_dir
            batch_pattern = "*batch_*.jsonl"
        
        # Check each batch file to see if its contents have been processed
        if batch_dir.exists():
            batch_files = sorted(batch_dir.glob(batch_pattern))
            
            for batch_file in batch_files:
                try:
                    # Load batch data
                    batch_data = utils.load_jsonl_file(batch_file, as_dataframe=False)
                    
                    # Generate batch ID the same way the loading function does
                    if mode == "merge":
                        batch_id = f"merge_batch_{batch_file.stem}"
                    else:
                        batch_id = f"semantic_batch_{batch_file.stem.split('_')[-1]}"
                    
                    # Check if this batch has been processed
                    batch_is_processed = False
                    
                    if mode == "merge":
                        # For merge mode, check if any category names from this batch were processed
                        for item in batch_data:
                            if item.get('name') in processed_names:
                                batch_is_processed = True
                                break
                    else:
                        # For keywords mode, check if any keywords from this batch were processed
                        for item in batch_data:
                            if item.get('name') in processed_keywords:
                                batch_is_processed = True
                                break
                    
                    if batch_is_processed:
                        processed_batch_ids.add(batch_id)
                        
                except Exception as e:
                    print(f"Warning: Could not load batch file {batch_file}: {e}")
                    continue
        
        print(f"Found {len(processed_batch_ids)} already processed batches in {mode} mode")
        return processed_batch_ids
        
    except Exception as e:
        print(f"Warning: Error determining processed batches from {output_file_path}: {e}")
        return processed_batch_ids


def load_categorise_dataset(batch_dir: Path = config.Keywords.batch_dir):
    """
    Load dataset from semantic clustering batches for keyword categorization with resume capability.
    
    This function loads keyword batches and filters out any that have already been processed,
    allowing for resumption of interrupted categorization runs.
    """
    return _load_batch_dataset_generic(
        batch_dir=batch_dir,
        output_file_path=config.Categories.category_extracted_path,
        batch_id_prefix="semantic_batch_",
        sample_creator_func=_create_keyword_sample,
        dataset_type="keyword",
        mode="keywords"
    )


def load_merge_dataset() -> MemoryDataset:

    return _load_batch_dataset_generic(
        batch_dir=config.Categories.batch_dir,
        output_file_path=config.Categories.merged_categories_path,
        batch_id_prefix="merge_batch_",
        sample_creator_func=_create_merge_sample,
        dataset_type="category",
        mode="merge"
    )




def create_keyword_lookup(keywords_path: Path = config.Keywords.keywords_path):
    """Create efficient keyword lookup dictionary."""
    try:
        original_keywords = utils.load_jsonl_file(keywords_path, as_dataframe=False)
        return {kw['name']: kw for kw in original_keywords}
    except Exception as e:
        print(f"Error loading keywords from {keywords_path}: {e}")
        return {}


def load_string_jsonl(file_path: Path):
    """Load list of strings from JSONL file with error handling."""
    keywords = []
    if not file_path.exists():
        return keywords
        
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f, 1):
                try:
                    keyword = json.loads(line.strip())
                    if isinstance(keyword, str):
                        keywords.append(keyword)
                    else:
                        print(f"Warning: Non-string value on line {i} in {file_path}: {keyword}")
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed line {i} in {file_path}: {e}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    
    return keywords


def deduplicate_categories(input_path=config.Categories.category_extracted_path):
    """
    Deduplicate categories by normalizing names and selecting best variants.
    
    Args:
        input_path: Path to category proposals file. If None, uses config default.
    
    Returns:
        list: Deduplicated categories with merged keywords
    """

    
    categories_dict = {}  # normalized_name -> {category_data, variants}
    all_categories = utils.load_jsonl_file(input_path, as_dataframe=False)
    
    for category in all_categories:
        normalized_name = utils.normalize(category["name"])
        
        if normalized_name not in categories_dict:
            categories_dict[normalized_name] = {
                "variants": [],
                "keywords": set()
            }
        
        categories_dict[normalized_name]["variants"].append(category)
        categories_dict[normalized_name]["keywords"].update(category.get("keywords", []))
    
    final_categories = []
    
    # For each normalized name, choose the variant with the longest description
    for normalized_name, data in categories_dict.items():
        # Find variant with longest description
        best_variant = max(data["variants"], key=lambda c: len(c.get("description", "")))
        
        # Create final category with all keywords
        merged_keywords = list(data["keywords"])
        final_category = {
            "name": best_variant["name"],
            "description": best_variant["description"],
            "keywords": merged_keywords,
            "field_of_research": best_variant["field_of_research"]
        }
        
        final_categories.append(final_category)
    
    return final_categories


def postprocess_categories(input_path=config.Categories.category_extracted_path, output_path=config.Categories.category_proposal_path):
    
    deduplicated = deduplicate_categories(input_path)
    utils.save_jsonl_file(deduplicated, output_path)
    


def hook_factory(mode):
    @hooks(name="OutputHook", description="Hook to save categorisation results as JSON files")
    class CategoriseOutputHook(Hooks):
        def __init__(self):
            self.mode = mode  # "keywords" or "merge"
            
            # Choose output paths based on mode
            if mode == "merge":
                self.merged_categories_path = config.Categories.merged_categories_path
                self.unknown_path = config.Categories.merge_unknown_path
                self.missing_path = config.Categories.merge_missing_path
            else:
                self.merged_categories_path = config.Categories.category_extracted_path
                self.unknown_path = config.Categories.unknown_keywords_path
                self.missing_path = config.Categories.missing_keywords_path
            
            # Open jsonlines writers for immediate writing
            self.categories_writer = jsonlines.open(self.merged_categories_path, 'a')
            
            if mode == "merge":
                self.unknown_writer = jsonlines.open(self.unknown_path, 'a')
                self.missing_writer = jsonlines.open(self.missing_path, 'a')
            else:
                self.unknown_keywords_writer = jsonlines.open(self.unknown_path, 'a')
                self.missing_keywords_writer = jsonlines.open(self.missing_path, 'a')
            
            # Initialize counters
            if mode == "merge":
                self.total_input_categories = 0
                self.total_merged_categories = 0
                self.total_unknown_categories = 0
                self.total_missing_categories = 0
            else:
                self.total_input_keywords = 0
                self.total_categories = 0
                self.total_categorized_keywords = 0
                self.total_unknown_keywords = 0
                self.total_missing_keywords = 0
            

        async def on_sample_end(self, data: SampleEnd) -> None:
            """Write categorisation results immediately to disk."""
            try:
                # Parse the LLM output
                _, result_json = utils.parse_results(data.sample.output.completion)
                
                if self.mode == "merge":
                    # For merge mode, track input categories and merged categories
                    input_categories = data.sample.metadata.get('category_names', [])
                    self.total_input_categories += len(input_categories)
                    
                    # Get merged categories
                    merged_categories = result_json.get('categories', [])
                    self.total_merged_categories += len(merged_categories)
                    
                    # Track which input categories are referenced
                    referenced_categories = set()
                    
                    for merged_category in merged_categories:
                        # Add source metadata
                        batch_id = data.sample.metadata.get('batch_id', data.sample.id)
                        merged_category['source_batch_id'] = batch_id
                        merged_category['source_batch_file'] = data.sample.metadata.get('batch_file')
                        
                        # Track referenced source categories
                        source_categories = merged_category.get('source_categories', [])
                        referenced_categories.update(source_categories)
                        
                        self.categories_writer.write(merged_category)
                    
                    # Find missing categories (input categories not referenced in any merged category)
                    missing_categories = set(input_categories) - referenced_categories
                    for missing_category in missing_categories:
                        self.missing_writer.write(missing_category)
                        self.total_missing_categories += 1
                    
                    # Handle missing categories from scorer if available
                    if hasattr(data.sample, 'scores') and 'category_confusion_matrix' in data.sample.scores:
                        scorer_missing = data.sample.scores['category_confusion_matrix'].metadata.get('fn', [])
                        for missing_category in scorer_missing:
                            if missing_category not in missing_categories:  # Avoid duplicates
                                self.missing_writer.write(missing_category)
                                self.total_missing_categories += 1
                
                else:
                    # For keywords mode, use existing logic
                    sample_keywords = set(data.sample.metadata.get('keywords', []))
                    self.total_input_keywords += len(sample_keywords)
                    
                    # Get categories
                    categories = result_json.get('categories', [])
                    self.total_categories += len(categories)
                    
                    # Write categories immediately, excluding "Unknown" categories from proposals
                    for category in categories:
                        # Track categorized keywords and write unknown keywords immediately
                        category_keywords = set(category.get('keywords', []))
                        self.total_categorized_keywords += len(category_keywords)
                        
                        # Handle "Unknown" category separately - don't include in proposals
                        if category.get('name') == "Unknown":
                            for keyword in category_keywords:
                                self.unknown_keywords_writer.write(keyword)
                                self.total_unknown_keywords += 1
                        else:
                            self.categories_writer.write(category)
                    
                    # Write missing keywords from scorer immediately
                    if hasattr(data.sample, 'scores') and 'keywords_confusion_matrix' in data.sample.scores:
                        missing_keywords = data.sample.scores['keywords_confusion_matrix'].metadata.get('fn', [])
                        for keyword in missing_keywords:
                            self.missing_keywords_writer.write(keyword)
                            self.total_missing_keywords += 1
                    
            except Exception as e:
                print(f"Error processing sample {data.sample.id}: {e}")

        async def on_task_end(self, data: TaskEnd) -> None:
            """Close file writers and print summary statistics."""
            # Print summary statistics
            print(f"\n=== Categorization Summary ===")
            print(f"Mode: {self.mode}")
            
            if self.mode == "merge":
                print(f"Total input categories: {self.total_input_categories}")
                print(f"Total merged categories: {self.total_merged_categories}")
                print(f"Missing categories: {self.total_missing_categories}")
                print(f"Merged categories written to: {self.merged_categories_path}")
                if self.total_missing_categories > 0:
                    print(f"Missing categories written to: {self.missing_path}")
            else:
                print(f"Total input keywords: {self.total_input_keywords}")
                print(f"Total categories created: {self.total_categories}")
                print(f"Total categorized keywords: {self.total_categorized_keywords}")
                print(f"Unknown keywords (excluded): {self.total_unknown_keywords}")
                print(f"Missing keywords: {self.total_missing_keywords}")
                print(f"Categories written to: {self.merged_categories_path}")
                if self.total_unknown_keywords > 0:
                    print(f"Unknown keywords written to: {self.unknown_path}")
                    
            print("===============================\n")
            
            if self.mode == "keywords":
                postprocess_categories()

            # Close writers
            self.categories_writer.close()
            if self.mode == "merge":
                self.unknown_writer.close()
                self.missing_writer.close()
            else:
                self.unknown_keywords_writer.close()
                self.missing_keywords_writer.close()


@task
def merge():

    dataset = load_merge_dataset()
    response_schema = ResponseSchema(
        name="MergedCategories",
        json_schema=json_schema(MergedCategoryList),
        strict=True
    )
    scorer_func = category_confusion_matrix()
    print("Running in category merging mode")

    # Create hook instance with mode
    hook_factory(mode="merge")

    return Task(
        dataset=dataset,
        solver=[
            system_message(MERGE_SYSTEM_PROMPT),
            generate()
        ],
        scorer=[
            scorer_func
        ],
        config=GenerateConfig(
            response_schema=response_schema,
        ),
        hooks=["OutputHook"]  # Use the dynamically created hook class name
    )



@task
def categorise():

    dataset = load_categorise_dataset()
    response_schema = ResponseSchema(
        name="Categories",
        json_schema=json_schema(CategoryList),
        strict=True
    )
    scorer_func = keywords_confusion_matrix()
    print("Running in keywords categorization mode")

    # Create hook instance with mode
    hook_factory(mode="keywords")

    return Task(
        dataset=dataset,
        solver=[
            system_message(CATEGORISE_SYSTEM_PROMPT),
            generate()
        ],
        scorer=[
            scorer_func
        ],
        config=GenerateConfig(
            response_schema=response_schema,
        ),
        hooks=["OutputHook"]  # Use the dynamically created hook class name
    )



