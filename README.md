# Research Link Technology Landscaping

🔍 A keyword-based research analysis system for automatically discovering research topics and assigning grants to those topics based on content analysis.

## Configuration

The system uses a centralized `config.yaml` file for all configuration settings including data paths, model parameters, and display options. Key configuration sections:

- **Data paths**: Configurable directories and file names
- **Modeling**: Default AI models, thresholds, and processing parameters  
- **Web dashboard**: Display settings, chart dimensions, and UI configuration
- **API settings**: Timeouts, retry limits, and service endpoints

## Features

- **Keyword Extraction**: Automatically extract key research terms from grant summaries
- **Topic Clustering**: Group related keywords into coherent research topics  
- **Grant Assignment**: Assign grants to topics based on keyword matches
- **Interactive Dashboard**: Visualize results with funding analysis and topic exploration
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

The analysis pipeline consists of three main steps:

1. **Extract Keywords** (`make extract-keywords`)
   - Uses LLM to extract relevant research keywords from grant summaries
   - Generates structured keyword data for each grant

2. **Cluster Keywords** (`make cluster-keywords`) 
   - Groups related keywords into research topics
   - Creates topic descriptions and confidence scores
   - Outputs 9 distinct research topics

3. **Assign Topics** (`make assign-topics`)
   - Matches grants to topics based on keyword overlap
   - Calculates topic relevance scores
   - Generates comprehensive assignment data

## Generated Data

The workflow produces:

- `data/keyword_clusters.json` - Research topics with keywords and descriptions
- `data/grant_topic_assignments.json` - Grant-to-topic assignments with scores

## Dashboard Features

The web dashboard (`make web-app`) provides:

- **Topic Overview**: Grant distribution across research topics
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
- `make cluster-keywords` - Cluster keywords into topics
- `make assign-topics` - Assign grants to topics

### Data Collection
- `make data` - Fetch core research data
- `make grants` - Fetch grants data only

### Utilities
- `make clean-workflow` - Remove generated analysis files
- `make help` - Show all available commands

## Project Structure

```
├── modeling/               # Analysis pipeline
│   ├── keywords_extraction.py   # Keyword extraction logic
│   ├── keywords_clustering.py   # Topic clustering logic  
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