"""
ANZSRC Data Cleaner

This script cleans and structures the Australian and New Zealand Standard Research 
Classification (ANZSRC) codes from Excel files into a structured JSON format.

Supports both:
- Field of Research (FOR) codes
- Socio-Economic Objective (SEO) codes

This is designed to be run once to process the ANZSRC codes data.
"""

import json
import pandas as pd
import re
from pathlib import Path
from config import DATA_DIR
import utils


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


def get_code_ranges(classification_type):
    """Get the valid code ranges for each classification type"""
    if classification_type.lower() == 'for':
        return {
            'division_range': (30, 52),  # FOR divisions are 30-52
            'group_range': (1000, 9999),
            'field_range': (100000, 999999)
        }
    elif classification_type.lower() == 'seo':
        return {
            'division_range': (10, 99),  # SEO divisions are 10-99
            'group_range': (1000, 9999),
            'field_range': (100000, 999999)
        }
    else:
        raise ValueError(f"Unknown classification type: {classification_type}")


def clean_anzsrc_codes_data(excel_file_path, classification_type):
    """
    Clean and restructure the ANZSRC codes data from Excel file
    Reads both Table 3 (complete hierarchy) and Table 4 (definitions/exclusions)
    
    Args:
        excel_file_path: Path to the ANZSRC Excel file
        classification_type: Either 'FOR' or 'SEO'
        
    Returns:
        dict: Hierarchical structure of codes including fields
    """
    ranges = get_code_ranges(classification_type)
    
    # Load Table 3 for complete hierarchy (divisions, groups, fields)
    df_table3 = pd.read_excel(excel_file_path, sheet_name="Table 3", skiprows=8)
    
    # Load Table 4 for definitions and exclusions
    df_table4 = pd.read_excel(excel_file_path, sheet_name="Table 4", skiprows=8)
    
    # Clean and restructure the data from Table 3
    cleaned_data = []
    current_division = None
    current_division_name = None
    current_group = None
    current_group_name = None

    for idx, row in df_table3.iterrows():
        division_code = row['Unnamed: 0']
        group_value = row['Group']
        field_code = row['Unnamed: 2']
        field_name = row['Unnamed: 3']
        
        # Check if this is a division (2-digit number in first column for FOR, or wider range for SEO)
        if pd.notna(division_code) and isinstance(division_code, (int, float)):
            division_code = int(division_code)
            if ranges['division_range'][0] <= division_code <= ranges['division_range'][1]:
                current_division = division_code
                current_division_name = group_value
                current_group = None
                current_group_name = None
                
                cleaned_data.append({
                    'classification_type': classification_type.upper(),
                    'level': 'division',
                    'code': division_code,
                    'name': group_value,
                    'definition': None,  # Will be filled from Table 4
                    'exclusions': None,  # Will be filled from Table 4
                    'parent_code': None,
                    'parent_name': None
                })
        
        # Check if this is a group (4-digit number in Group column)
        elif pd.notna(group_value) and isinstance(group_value, (int, float)):
            group_code = int(group_value)
            if ranges['group_range'][0] <= group_code <= ranges['group_range'][1]:
                current_group = group_code
                current_group_name = field_code  # In Table 3, group names are in 'Unnamed: 2'
                
                cleaned_data.append({
                    'classification_type': classification_type.upper(),
                    'level': 'group',
                    'code': group_code,
                    'name': current_group_name,
                    'definition': None,  # Will be filled from Table 4
                    'exclusions': None,  # Will be filled from Table 4
                    'parent_code': current_division,
                    'parent_name': current_division_name
                })
        
        # Check if this is a field (6-digit number in field_code column)
        elif pd.notna(field_code) and isinstance(field_code, (int, float)):
            field_code_int = int(field_code)
            if ranges['field_range'][0] <= field_code_int <= ranges['field_range'][1]:
                cleaned_data.append({
                    'classification_type': classification_type.upper(),
                    'level': 'field',
                    'code': field_code_int,
                    'name': field_name if pd.notna(field_name) else None,
                    'definition': None,  # Fields don't have definitions in Table 4
                    'exclusions': None,
                    'parent_code': current_group,
                    'parent_name': current_group_name
                })

    # Now merge definitions and exclusions from Table 4
    definitions_exclusions = {}
    current_division_t4 = None

    for idx, row in df_table4.iterrows():
        group_value = row['Group']
        definition = row['Definition']
        exclusions = row['Exclusions'] if pd.notna(row['Exclusions']) else None
        
        # Check if this is a division (all caps, no numbers for FOR, or number for SEO)
        if isinstance(group_value, str) and group_value.isupper() and not any(char.isdigit() for char in group_value):
            division_code = int(row['Unnamed: 0']) if pd.notna(row['Unnamed: 0']) else None
            current_division_t4 = division_code
            
            if division_code:
                definitions_exclusions[division_code] = {
                    'definition': definition,
                    'exclusions': exclusions
                }
        
        # Check if this is a group (4-digit number)
        elif isinstance(group_value, int) and ranges['group_range'][0] <= group_value <= ranges['group_range'][1]:
            definitions_exclusions[group_value] = {
                'definition': definition,
                'exclusions': exclusions
            }

    # Convert to DataFrame and merge definitions/exclusions
    cleaned_df = pd.DataFrame(cleaned_data)
    
    # Merge definitions and exclusions from Table 4
    for idx, row in cleaned_df.iterrows():
        code = row['code']
        if code in definitions_exclusions:
            cleaned_df.loc[idx, 'definition'] = definitions_exclusions[code]['definition']
            cleaned_df.loc[idx, 'exclusions'] = definitions_exclusions[code]['exclusions']
    
    # Apply text cleaning functions
    cleaned_df['definition_structured'] = cleaned_df['definition'].apply(clean_definition)
    cleaned_df['exclusions_structured'] = cleaned_df['exclusions'].apply(clean_exclusions)
    
    # Fix missing name for group 5204 if it exists (FOR specific)
    if classification_type.upper() == 'FOR':
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
            'classification_type': classification_type.upper(),
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
                'exclusions': row['exclusions_structured'],
                'fields': {}
            }
    
    # Add fields to groups
    for _, row in cleaned_df[cleaned_df['level'] == 'field'].iterrows():
        if pd.notna(row['parent_code']):
            parent_group_code = int(row['parent_code'])
            field_code = row['code']
            
            # Find the division that contains this group
            division_code = None
            for div_code, div_data in hierarchy.items():
                if parent_group_code in div_data['groups']:
                    division_code = div_code
                    break
            
            if division_code and parent_group_code in hierarchy[division_code]['groups']:
                hierarchy[division_code]['groups'][parent_group_code]['fields'][field_code] = {
                    'code': field_code,
                    'name': row['name'],
                    'definition': None,  # Fields don't have definitions
                    'exclusions': None
                }
    
    total_groups = sum(len(div['groups']) for div in hierarchy.values())
    total_fields = sum(sum(len(group['fields']) for group in div['groups'].values()) for div in hierarchy.values())
    print(f"{classification_type.upper()} codes cleaned: {len(hierarchy)} divisions, {total_groups} groups, {total_fields} fields")
    
    return hierarchy


def validate_anzsrc_data(hierarchy, classification_type):
    """Validate the cleaned ANZSRC codes data structure"""
    divisions = len(hierarchy)
    total_groups = sum(len(div['groups']) for div in hierarchy.values())
    total_fields = sum(sum(len(group['fields']) for group in div['groups'].values()) for div in hierarchy.values())

    divisions_with_names = sum(1 for div in hierarchy.values() if div.get('name'))
    # Count groups with names
    groups_with_names = 0
    total_group_count = 0
    for div in hierarchy.values():
        for group in div['groups'].values():
            total_group_count += 1
            if group.get('name'):
                groups_with_names += 1
    
    # Count fields with names
    fields_with_names = 0
    total_field_count = 0
    for div in hierarchy.values():
        for group in div['groups'].values():
            for field in group['fields'].values():
                total_field_count += 1
                if field.get('name'):
                    fields_with_names += 1

    issues = []
    if divisions_with_names != divisions:
        issues.append(f"missing division names:{divisions - divisions_with_names}")
    if groups_with_names != total_group_count:
        issues.append(f"missing group names:{total_group_count - groups_with_names}")
    if fields_with_names != total_field_count:
        issues.append(f"missing field names:{total_field_count - fields_with_names}")

    if issues:
        print(f"{classification_type.upper()} codes validation issues: {', '.join(issues)}")
    else:
        print(f"{classification_type.upper()} codes validation OK: divisions={divisions}, groups={total_groups}, fields={total_fields}")

    return True


def export_anzsrc_codes_to_json(hierarchy, output_path, classification_type):
    """Export the cleaned ANZSRC codes to JSON"""
    utils.save_json_file(hierarchy, output_path)
    print(f"Exported {classification_type.upper()} codes to: {output_path}")
    return output_path


def merge_anzsrc_codes(for_hierarchy, seo_hierarchy):
    """Merge FOR and SEO hierarchies into a single structure"""
    merged = {
        'FOR': for_hierarchy,
        'SEO': seo_hierarchy,
        'metadata': {
            'for_divisions': len(for_hierarchy),
            'for_groups': sum(len(div['groups']) for div in for_hierarchy.values()),
            'for_fields': sum(sum(len(group['fields']) for group in div['groups'].values()) for div in for_hierarchy.values()),
            'seo_divisions': len(seo_hierarchy),
            'seo_groups': sum(len(div['groups']) for div in seo_hierarchy.values()),
            'seo_fields': sum(sum(len(group['fields']) for group in div['groups'].values()) for div in seo_hierarchy.values()),
        }
    }
    
    print(f"Merged ANZSRC codes: FOR({merged['metadata']['for_divisions']} div, {merged['metadata']['for_groups']} groups, {merged['metadata']['for_fields']} fields) + SEO({merged['metadata']['seo_divisions']} div, {merged['metadata']['seo_groups']} groups, {merged['metadata']['seo_fields']} fields)")
    return merged


def process_for_codes(output_path=None):
    """Process FOR codes specifically"""
    excel_file_path = DATA_DIR / "anzsrc2020_for.xlsx"
    if output_path is None:
        output_path = DATA_DIR / "for_codes_cleaned.json"
    
    hierarchy = clean_anzsrc_codes_data(excel_file_path, 'FOR')
    validate_anzsrc_data(hierarchy, 'FOR')
    export_anzsrc_codes_to_json(hierarchy, output_path, 'FOR')
    return hierarchy


def process_seo_codes(output_path=None):
    """Process SEO codes specifically"""
    excel_file_path = DATA_DIR / "anzsrc2020_seo.xlsx"
    if output_path is None:
        output_path = DATA_DIR / "seo_codes_cleaned.json"
    
    hierarchy = clean_anzsrc_codes_data(excel_file_path, 'SEO')
    validate_anzsrc_data(hierarchy, 'SEO')
    export_anzsrc_codes_to_json(hierarchy, output_path, 'SEO')
    return hierarchy


def main():
    """Main function to clean and export both FOR and SEO codes"""
    print("Processing ANZSRC codes...")
    
    # Process FOR codes
    print("\n=== Processing FOR codes ===")
    for_hierarchy = process_for_codes()
    
    # Process SEO codes
    print("\n=== Processing SEO codes ===")
    seo_hierarchy = process_seo_codes()
    
    # Merge both hierarchies
    print("\n=== Merging codes ===")
    merged_hierarchy = merge_anzsrc_codes(for_hierarchy, seo_hierarchy)
    
    # Export merged data
    merged_output_path = DATA_DIR / "anzsrc_codes_merged.json"
    utils.save_json_file(merged_hierarchy, merged_output_path)
    print(f"Exported merged ANZSRC codes to: {merged_output_path}")
    
    return merged_hierarchy


if __name__ == "__main__":
    main()
