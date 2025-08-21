"""
FOR Codes Data Cleaner

This script cleans and structures the Australian and New Zealand Standard Research 
Classification (ANZSRC) Field of Research (FOR) codes from the Excel file into 
a structured JSON format.

This is designed to be run once to process the FOR codes data.
"""

import json
import pandas as pd
import re
from pathlib import Path
from config import DATA_DIR


def clean_text(text):
    """Clean text by removing extra whitespace and normalizing formatting"""
    if pd.isna(text):
        return None
    return ' '.join(str(text).split())


def clean_definition(definition):
    """Clean and format definitions"""
    if pd.isna(definition):
        return None
    
    # Clean up the text
    cleaned = clean_text(definition)
    
    # Split bullet points if they exist
    if '•' in cleaned:
        parts = cleaned.split('•')
        main_text = parts[0].strip()
        bullets = [bullet.strip() for bullet in parts[1:] if bullet.strip()]
        
        return {
            'description': main_text,
            'includes': bullets
        }
    else:
        return {
            'description': cleaned,
            'includes': []
        }


def clean_exclusions(exclusions):
    """Clean and structure exclusions"""
    if pd.isna(exclusions):
        return []
    
    # Split by lowercase letters followed by )
    exclusion_pattern = r'([a-z]\))'
    parts = re.split(exclusion_pattern, str(exclusions))
    
    exclusions_list = []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            exclusion_text = clean_text(parts[i + 1])
            if exclusion_text:
                exclusions_list.append(exclusion_text)
    
    return exclusions_list


def clean_for_codes_data(excel_file_path):
    """
    Clean and restructure the FOR codes data from ANZSRC Excel file
    
    Args:
        excel_file_path: Path to the ANZSRC FOR codes Excel file
        
    Returns:
        dict: Hierarchical structure of FOR codes
    """
    # Load the Excel data
    df = pd.read_excel(excel_file_path, sheet_name="Table 4", skiprows=8)
    
    # Clean and restructure the data
    cleaned_data = []
    current_division = None
    current_division_name = None

    for idx, row in df.iterrows():
        group_value = row['Group']
        definition = row['Definition']
        exclusions = row['Exclusions'] if pd.notna(row['Exclusions']) else None
        
        # Check if this is a division (all caps, no numbers)
        if isinstance(group_value, str) and group_value.isupper() and not any(char.isdigit() for char in group_value):
            # This is a division level
            division_code = int(row['Unnamed: 0']) if pd.notna(row['Unnamed: 0']) else None
            current_division = division_code
            current_division_name = group_value
            
            cleaned_data.append({
                'level': 'division',
                'code': division_code,
                'name': group_value,
                'definition': definition,
                'exclusions': exclusions,
                'parent_code': None,
                'parent_name': None
            })
        
        # Check if this is a group (4-digit number)
        elif isinstance(group_value, int) and 1000 <= group_value <= 9999:
            group_name = row['Unnamed: 2'] if pd.notna(row['Unnamed: 2']) else None
            
            cleaned_data.append({
                'level': 'group',
                'code': group_value,
                'name': group_name,
                'definition': definition,
                'exclusions': exclusions,
                'parent_code': current_division,
                'parent_name': current_division_name
            })

    # Convert to DataFrame
    cleaned_df = pd.DataFrame(cleaned_data)
    
    # Apply text cleaning functions
    cleaned_df['definition_structured'] = cleaned_df['definition'].apply(clean_definition)
    cleaned_df['exclusions_structured'] = cleaned_df['exclusions'].apply(clean_exclusions)
    
    # Fix missing name for group 5204 if it exists
    missing_name_mask = (cleaned_df['code'] == 5204) & cleaned_df['name'].isna()
    if missing_name_mask.any():
        fixed_name = "Cognitive and computational psychology"
        cleaned_df.loc[missing_name_mask, 'name'] = fixed_name
    
    # Create hierarchical structure
    hierarchy = {}

    # Group by divisions
    for _, row in cleaned_df[cleaned_df['level'] == 'division'].iterrows():
        division_code = row['code']
        
        hierarchy[division_code] = {
            'code': division_code,
            'name': row['name'],
            'definition': row['definition_structured'],
            'exclusions': row['exclusions_structured'],
            'groups': {}
        }

    # Add groups to divisions
    for _, row in cleaned_df[cleaned_df['level'] == 'group'].iterrows():
        parent_code = int(row['parent_code'])
        group_code = row['code']
        
        if parent_code in hierarchy:
            hierarchy[parent_code]['groups'][group_code] = {
                'code': group_code,
                'name': row['name'],
                'definition': row['definition_structured'],
                'exclusions': row['exclusions_structured']
            }
    
    print(f"FOR codes cleaned: {len(hierarchy)} divisions, {sum(len(div['groups']) for div in hierarchy.values())} groups")
    
    return hierarchy


def validate_for_data(hierarchy):
    """Validate the cleaned FOR codes data structure"""
    divisions = len(hierarchy)
    total_groups = sum(len(div['groups']) for div in hierarchy.values())

    divisions_with_names = sum(1 for div in hierarchy.values() if div.get('name'))
    # Count groups with names
    groups_with_names = 0
    total_group_count = 0
    for div in hierarchy.values():
        for group in div['groups'].values():
            total_group_count += 1
            if group.get('name'):
                groups_with_names += 1

    issues = []
    if divisions_with_names != divisions:
        issues.append(f"missing division names:{divisions - divisions_with_names}")
    if groups_with_names != total_group_count:
        issues.append(f"missing group names:{total_group_count - groups_with_names}")

    if issues:
        print(f"FOR codes validation issues: {', '.join(issues)}")
    else:
        print(f"FOR codes validation OK: divisions={divisions}, groups={total_groups}")

    return True


def export_for_codes_to_json(hierarchy, output_path):
    """Export the cleaned FOR codes to JSON"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hierarchy, f, indent=2, ensure_ascii=False)
    print(f"Exported FOR codes to: {output_path}")
    return output_path


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
