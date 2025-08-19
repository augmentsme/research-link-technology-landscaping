# Research Landscape Visualizations

This module (`visualisation.py`) provides interactive visualizations for the research link technology landscaping analysis.

## Features

- **Treemap Visualization**: Hierarchical view of research domains → categories → keywords
- **Interactive Interface**: Plotly-based interactive charts with hover details
- **Data Integration**: Seamlessly integrates with analysis pipeline results
- **Error Handling**: Gracefully handles missing data files

## Usage

### Standalone Usage
```bash
python visualisation.py
```

### As Module Import
```python
from visualisation import create_research_landscape_treemap

# Use with pre-loaded data
fig = create_research_landscape_treemap(
    refined_categories=refined_categories,
    detailed_categories=detailed_categories,
    classification_results=classification_results
)

# Or let the function load data automatically
fig = create_research_landscape_treemap()

if fig is not None:
    fig.show()
```

## Data Requirements

The visualization requires the following data files:

### Required Files
- `logs/results/categories.json` - Detailed categories with keywords (from categorise task)
- `logs/results/refined_categories.json` - Strategic domains (from refine task)

### Optional Files
- `logs/results/classification.json` - Grant classifications (from classify task)

## Visualization Types

### Research Landscape Treemap
- **Level 1**: Strategic Domains (Blue) - High-level research themes
- **Level 2**: Detailed Categories (Orange) - Specific research areas  
- **Level 3**: Keywords (Green) - Individual research terms

### Features
- **Size**: Proportional to number of keywords in each category
- **Color**: Different colors for each hierarchical level
- **Interactivity**: Hover for details, click to drill down
- **Statistics**: Summary of grants and classifications

## Output

- **Interactive HTML**: Opens in browser for exploration
- **Console Summary**: Detailed statistics about the research landscape
- **Classifications**: Grant distribution across categories (if available)

## Prerequisites

To generate the required data files, run the analysis pipeline:

```bash
# Generate categories and keywords
make categorise

# Generate strategic domains
make refine  

# Generate grant classifications (optional)
make classify
```

## Error Handling

The module gracefully handles:
- Missing data files
- Empty datasets
- Incomplete analysis results

When data is missing, it provides clear instructions on which pipeline steps need to be run.
