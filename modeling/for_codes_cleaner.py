"""
FOR Codes Data Cleaner

This script cleans and structures the Australian and New Zealand Standard Research 
Classification (ANZSRC) Field of Research (FOR) codes from the Excel file into 
a structured JSON format.

This is designed to be run once to process the FOR codes data.

DEPRECATED: Use anzsrc_cleaner.py for new projects that need both FOR and SEO codes.
This file is maintained for backward compatibility.
"""

import json
import pandas as pd
import re
from pathlib import Path
from config import DATA_DIR
from anzsrc_cleaner import clean_anzsrc_codes_data, validate_anzsrc_data, export_anzsrc_codes_to_json


# Utility functions are now imported from anzsrc_cleaner.py


def clean_for_codes_data(excel_file_path):
    """
    Clean and restructure the FOR codes data from ANZSRC Excel file
    Reads both Table 3 (complete hierarchy) and Table 4 (definitions/exclusions)
    
    Args:
        excel_file_path: Path to the ANZSRC FOR codes Excel file
        
    Returns:
        dict: Hierarchical structure of FOR codes including fields
        
    DEPRECATED: Use clean_anzsrc_codes_data from anzsrc_cleaner.py instead
    """
    return clean_anzsrc_codes_data(excel_file_path, 'FOR')


def validate_for_data(hierarchy):
    """
    Validate the cleaned FOR codes data structure
    
    DEPRECATED: Use validate_anzsrc_data from anzsrc_cleaner.py instead
    """
    return validate_anzsrc_data(hierarchy, 'FOR')


def export_for_codes_to_json(hierarchy, output_path):
    """
    Export the cleaned FOR codes to JSON
    
    DEPRECATED: Use export_anzsrc_codes_to_json from anzsrc_cleaner.py instead
    """
    return export_anzsrc_codes_to_json(hierarchy, output_path, 'FOR')


def main(output_path=None):
    """Main function to clean and export FOR codes"""
    # Set up file paths
    excel_file_path = DATA_DIR / "anzsrc2020_for.xlsx"

    # Use default path if none provided
    if output_path is None:
        output_path = DATA_DIR / "for_codes_cleaned.json"

    # Process the FOR codes
    hierarchy = clean_for_codes_data(excel_file_path)

    # Validate the data
    validate_for_data(hierarchy)

    # Export to JSON
    export_for_codes_to_json(hierarchy, output_path)

    return hierarchy


if __name__ == "__main__":
    main()
