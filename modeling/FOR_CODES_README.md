# FOR Codes Cleaner

This module (`for_codes_cleaner.py`) processes the Australian and New Zealand Standard Research Classification (ANZSRC) Field of Research (FOR) codes from the original Excel format into a clean, structured JSON format.

## Usage

### Standalone Usage
```bash
python for_codes_cleaner.py
```

### As Part of Preprocessing Pipeline
The FOR codes cleaning is automatically included when running the main preprocessing script:
```bash
python preprocessing.py
```

## Input

- **Excel File**: `data/anzsrc2020_for.xlsx` (ANZSRC 2020 FOR codes, Table 4)
- **Format**: Excel spreadsheet with hierarchical structure

## Output

- **JSON File**: `data/for_codes_cleaned.json`
- **Format**: Hierarchical JSON structure

### Output Structure

```json
{
  "30": {
    "code": 30,
    "name": "AGRICULTURAL, VETERINARY AND FOOD SCIENCES",
    "definition": {
      "description": "This division covers the sciences and technologies supporting agriculture...",
      "includes": ["crop and pasture production", "agronomy", "horticultural production", ...]
    },
    "exclusions": ["Indigenous agricultural, veterinary and food sciences is included in Division 45 Indigenous studies."],
    "groups": {
      "3001": {
        "code": 3001,
        "name": "Agricultural biotechnology",
        "definition": {
          "description": "This group covers agricultural biotechnology.",
          "includes": []
        },
        "exclusions": ["Improvement of plants through selective breeding is included in Group 3004..."]
      }
    }
  }
}
```

## Features

- ✅ **Hierarchical Structure**: Maintains proper division → group relationships
- ✅ **Text Cleaning**: Normalizes formatting and extracts bullet points
- ✅ **Data Validation**: Ensures completeness and consistency
- ✅ **Error Handling**: Fixes known data issues (e.g., missing group 5204 name)
- ✅ **JSON Output**: Clean, structured format for programmatic use

## Data Processing

1. **Load Excel Data**: Reads ANZSRC Excel file (Table 4, skipping header rows)
2. **Identify Hierarchy**: Distinguishes between divisions (2-digit) and groups (4-digit codes)
3. **Clean Text**: Normalizes formatting, extracts bullet points from definitions
4. **Structure Data**: Creates hierarchical relationships between divisions and groups
5. **Validate**: Ensures data completeness and consistency
6. **Export**: Saves as structured JSON

## Statistics

- **23 Divisions**: High-level research domains (codes 30-52)
- **213 Groups**: Specific research areas (codes 3001-5299)
- **Complete Coverage**: All entries validated and structured

This tool is designed to be run once to process the FOR codes data into a usable format for the research landscaping analysis pipeline.
