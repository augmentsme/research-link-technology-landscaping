- ✅ **COMPLETED**: For all 3 tasks, after each task run you should extract the results in json format as specified by each json schema, you may use inspect ai hooks as specified in https://inspect.aisi.org.uk/reference/inspect_ai.hooks.html#sampleend and https://inspect.aisi.org.uk/reference/inspect_ai.hooks.html#hooks

## Implementation Summary

The following hooks have been implemented for all 3 tasks:

### 1. Extract Task (`extract()`)
- **Hook**: `save_keywords_json` - Saves individual keyword extraction results as JSON
- **Hook**: `save_aggregated_results` - Creates aggregated results file for all keyword extractions
- **Output**: Individual files in `logs/results/extract/{sample_id}_keywords.json` and aggregated in `logs/results/all_keywords_extracted.json`
- **Schema**: `KeywordsExtractionOutput` with keywords, methodology_keywords, application_keywords, technology_keywords

### 2. Identify Task (`identify()`)
- **Hook**: `save_taxonomy_json` - Saves taxonomy creation result as JSON
- **Hook**: `save_aggregated_results` - Creates final taxonomy file
- **Output**: Results in `logs/results/identify/taxonomy.json`
- **Schema**: `TaxonomyOutput` with 2-level taxonomy structure (top-level categories and subcategories)

### 3. Classify Task (`classify()`)
- **Hook**: `save_classification_json` - Saves individual grant classification results as JSON
- **Hook**: `save_aggregated_results` - Creates aggregated results file for all classifications
- **Output**: Individual files in `logs/results/classify/{sample_id}_classification.json` and aggregated in `logs/results/all_grants_classified.json`
- **Schema**: `ClassificationOutput` with grant classifications including top_level_category, subcategory, and reasoning

### Features Implemented:
- **Automatic JSON extraction** from model responses according to defined schemas
- **Error handling** for JSON parsing failures with informative logging
- **Metadata preservation** including sample IDs, timestamps, and task metadata
- **Organized file structure** with separate directories for each task
- **Aggregated results** combining all individual results for analysis
- **Sample-end hooks** for individual result capture
- **Task-end hooks** for aggregated result generation

### Usage:
After running any task with `inspect eval`, the JSON results will be automatically saved to the `logs/results/` directory structure, making it easy to access and analyze the extracted data programmatically.