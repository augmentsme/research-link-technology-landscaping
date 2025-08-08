# Research Link Technology Landscaping

🔍 A keyword-based research analysis system for automatically extracting keywords from research grants, harmonising them, and assigning harmonised keywords to grants based on content analysis.

## Configuration

The system uses a centralized `config.yaml` file for all configuration settings including data paths, model parameters, and display options. Key configuration sections:

- **Data paths**: Configurable directories and file names
- **Modeling**: Default AI models, thresholds, and processing parameters  
- **Web dashboard**: Display settings, chart dimensions, and UI configuration
- **API settings**: Timeouts, retry limits, and service endpoints

## Features

- **Keyword Extraction**: Automatically extract key research terms from grant summaries
- **Keyword Harmonisation**: Consolidate variants and synonyms into standardized terms  
- **Grant Assignment**: Assign harmonised keywords to grants based on content matches
- **Interactive Dashboard**: Visualize results with funding analysis and keyword exploration
- **Centralized Configuration**: Single config file manages all system settings

## Quick Start

### Option 1: One Command Setup
```bash
./quickstart.sh
```

### Option 2: Step-by-Step
```bash
# Install dependencies
make install

# Run complete analysis workflow
make workflow

# View results in dashboard
make web-app
```

## Workflow Steps

The analysis pipeline generates a single combined JSON file containing all workflow results:

1. **Extract Keywords** (`make extract-keywords`)
   - Uses LLM to extract relevant research keywords from grant summaries
   - Generates evaluation logs with structured keyword data

2. **Harmonise Keywords** (`make harmonise-keywords`) 
   - Consolidates keyword variants and synonyms into standardized terms
   - Creates evaluation logs with harmonised keyword mappings

3. **Generate Combined Results** (`python scripts/generate_combined_results.py`)
   - Reads from both evaluation log files 
   - Assigns harmonised keywords to grants
   - Creates a single `data/combined_workflow_results.json` file containing:
     - Original keyword extractions per grant
     - Harmonised keywords and mappings  
     - Keyword assignments to grants
     - Statistical summaries and funding totals

4. **Launch Dashboard** (`make web-app`)
   - Reads from the single combined JSON file
   - Displays interactive visualizations of all results
   - Maps original keywords to their harmonised forms
   - Eliminates duplicates while preserving meaning

3. **Assign Keywords** (`make assign-keywords`)
   - Assigns harmonised keywords to grants based on content matching
   - Creates keyword-grant associations with coverage statistics
   - Generates comprehensive assignment data

## Generated Data

The workflow produces:

- `data/harmonised_keywords.json` - Harmonised keywords with mappings and groupings
- `data/grant_keyword_assignments.json` - Grant-to-keyword assignments with statistics

## Dashboard Features

The web dashboard (`make web-app`) provides:

- **Keyword Overview**: Grant distribution across harmonised keywords
- **Funding Analysis**: Financial breakdown by research area  
- **Topic Details**: Deep dive into specific topics and their grants
- **Data Tables**: Filterable views of all assignments

## Sample Results

From our analysis of 10 sample grants ($41.4M total funding):

- **9 Research Topics** identified including:
  - Artificial Intelligence and Machine Learning
  - Materials Science and Engineering
  - Renewable Energy and Sustainability
  - Health and Biomedical Research

- **26 Topic Assignments** created with confidence scores
- **177 Unique Keywords** extracted and clustered

## Requirements

- Python 3.8+
- OpenAI API key (for LLM-based analysis)
- Dependencies listed in `requirements.txt`

## Commands Reference

### Main Workflow
- `make workflow` - Run complete analysis pipeline
- `make web-app` - Start interactive dashboard

### Individual Steps  
- `make extract-keywords` - Extract keywords from grants
- `make harmonise-keywords` - Harmonise keyword variants into standard terms
- `make assign-keywords` - Assign harmonised keywords to grants

### Data Collection
- `make data` - Fetch core research data
- `make grants` - Fetch grants data only

### Utilities
- `make clean-workflow` - Remove generated analysis files
- `make help` - Show all available commands

## Project Structure

```
├── modeling/               # Analysis pipeline
│   ├── keyword_tasks.py         # Keyword extraction and harmonisation tasks
│   └── topic_classification.py  # Grant assignment logic
├── web/                   # Dashboard interface
│   └── app.py            # Streamlit dashboard
├── data/                 # Generated data files
├── Makefile             # Workflow orchestration
└── quickstart.sh        # One-command setup
```

## API Integration

The system integrates with:
- **OpenAI**: For LLM-based keyword extraction and clustering
- **Research APIs**: For grant and publication data collection

## License

Licensed under the terms specified in the LICENSE file.