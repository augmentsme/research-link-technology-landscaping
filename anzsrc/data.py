import pandas as pd
import os


def clean_anzsrc_data(file_path, classification_type):
    """
    Clean ANZSRC Excel data and extract structured information.
    
    Args:
        file_path (str): Path to the Excel file
        classification_type (str): Either 'FoR' (Fields of Research) or 'SEO' (Socio-Economic Objectives)
    
    Returns:
        dict: Cleaned data with divisions and groups
    """
    # Read the Excel file
    df = pd.read_excel(file_path, sheet_name="Table 4")
    
    # Find the actual data start (after headers)
    data_start_row = None
    for i, row in df.iterrows():
        if pd.notna(row.iloc[0]) and isinstance(row.iloc[0], (int, float)) and row.iloc[0] > 0:
            data_start_row = i
            break
    
    if data_start_row is None:
        raise ValueError("Could not find data start row")
    
    # Extract data starting from the first division
    data_df = df.iloc[data_start_row:].copy()
    
    # Set proper column names
    data_df.columns = ['Code', 'GroupCode', 'GroupName', 'Definition', 'Exclusions']
    
    # Clean and structure the data
    divisions = {}
    current_division = None
    
    for _, row in data_df.iterrows():
        code = row['Code']
        group_code = row['GroupCode']
        group_name = row['GroupName']
        definition = row['Definition']
        exclusions = row['Exclusions']
        
        # Skip completely empty rows
        if pd.isna(code) and pd.isna(group_code) and pd.isna(group_name):
            continue
        
        # Check if this is a division row (has code in first column)
        if pd.notna(code) and isinstance(code, (int, float)):
            division_code = str(int(code)).zfill(2)
            current_division = {
                'code': division_code,
                'name': str(group_code).strip() if pd.notna(group_code) else '',  # Division name is in GroupCode column
                'definition': str(definition).strip() if pd.notna(definition) else '',
                'exclusions': str(exclusions).strip() if pd.notna(exclusions) else '',
                'groups': {}
            }
            divisions[division_code] = current_division
        
        # Check if this is a group row (has code in GroupCode column)
        elif pd.notna(group_code) and current_division is not None:
            group_code_str = str(int(group_code)) if isinstance(group_code, (int, float)) else str(group_code)
            group_name_str = str(group_name).strip() if pd.notna(group_name) else ''
            group_definition = str(definition).strip() if pd.notna(definition) else ''
            group_exclusions = str(exclusions).strip() if pd.notna(exclusions) else ''
            
            current_division['groups'][group_code_str] = {
                'code': group_code_str,
                'name': group_name_str,
                'definition': group_definition,
                'exclusions': group_exclusions
            }
    
    return {
        'classification_type': classification_type,
        'divisions': divisions
    }


def create_combined_dataframe(for_data, seo_data):
    """
    Create a combined pandas DataFrame with multi-level index (classification, division, group).
    
    Args:
        for_data (dict): Fields of Research data
        seo_data (dict): Socio-Economic Objectives data
    
    Returns:
        pd.DataFrame: Combined DataFrame with multi-level index
    """
    records = []
    
    # Process FoR data
    for div_code, division in for_data['divisions'].items():
        # Add division level record
        records.append({
            'classification': 'FoR',
            'division_code': div_code,
            'division_name': division['name'],
            'group_code': None,
            'group_name': None,
            'level': 'division',
            'definition': division['definition'],
            'exclusions': division['exclusions']
        })
        
        # Add group level records
        for group_code, group in division['groups'].items():
            records.append({
                'classification': 'FoR',
                'division_code': div_code,
                'division_name': division['name'],
                'group_code': group_code,
                'group_name': group['name'],
                'level': 'group',
                'definition': group['definition'],
                'exclusions': group['exclusions']
            })
    
    # Process SEO data
    for div_code, division in seo_data['divisions'].items():
        # Add division level record
        records.append({
            'classification': 'SEO',
            'division_code': div_code,
            'division_name': division['name'],
            'group_code': None,
            'group_name': None,
            'level': 'division',
            'definition': division['definition'],
            'exclusions': division['exclusions']
        })
        
        # Add group level records
        for group_code, group in division['groups'].items():
            records.append({
                'classification': 'SEO',
                'division_code': div_code,
                'division_name': division['name'],
                'group_code': group_code,
                'group_name': group['name'],
                'level': 'group',
                'definition': group['definition'],
                'exclusions': group['exclusions']
            })
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Set multi-level index
    df = df.set_index(['classification', 'division_code', 'group_code'])
    
    # Sort the index
    df = df.sort_index()
    
    return df


def load_anzsrc():
    for_data = clean_anzsrc_data("/home/lcheng/oz318/research-link-technology-landscaping/anzsrc/anzsrc2020_for.xlsx", "FoR")
    seo_data = clean_anzsrc_data("/home/lcheng/oz318/research-link-technology-landscaping/anzsrc/anzsrc2020_seo.xlsx", "SEO")
    combined_df = create_combined_dataframe(for_data, seo_data)
    return combined_df



df = load_anzsrc()
df.head()