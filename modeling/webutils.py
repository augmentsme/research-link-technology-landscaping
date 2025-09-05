"""Shared utilities for the multipage Research Link Technology Landscaping app."""

import streamlit as st
import json
import pandas as pd
import config

@st.cache_data
def load_data():
    """Load and cache the data"""
    try:
        keywords = config.Keywords.load()
        grants = config.Grants.load()
        categories = config.Categories.load()
        return keywords, grants, categories
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_data
def load_for_codes_mapping():
    """Load and create FOR codes to name mapping"""
    try:
        with open('data/for_codes_cleaned.json', 'r') as f:
            for_codes_data = json.load(f)
        
        code_to_name = {}
        
        # Add divisions (2-digit codes)
        for div_code, div_data in for_codes_data.items():
            code_to_name[div_code] = div_data['name']
            
            # Add groups (4-digit codes)
            if 'groups' in div_data:
                for group_code, group_data in div_data['groups'].items():
                    code_to_name[group_code] = group_data['name']
                    
                    # Add fields (6-digit codes)
                    if 'fields' in group_data:
                        for field_code, field_data in group_data['fields'].items():
                            code_to_name[field_code] = field_data['name']
        
        return code_to_name
    except Exception as e:
        st.error(f"Error loading FOR codes mapping: {e}")
        return {}

@st.cache_data
def get_unique_for_codes():
    """Get unique FOR codes efficiently - cached computation"""
    _, grants, _ = load_data()
    unique_for_codes = set()
    
    # Add primary FOR codes (only from grants that exist)
    for val in grants['for_primary'].dropna():
        unique_for_codes.add(str(int(val)))
    
    # Add secondary FOR codes (from comma-separated strings, only from grants that exist)
    for for_codes_str in grants['for'].dropna():
        for code in str(for_codes_str).split(','):
            code = code.strip()
            if code and len(code) <= 6:  # Filter out very long codes (likely malformed)
                unique_for_codes.add(code)
    
    return unique_for_codes

@st.cache_data
def create_for_code_options(unique_for_codes, for_codes_mapping):
    """Create options with both code and name, organized by hierarchy - cached"""
    for_code_options = []
    for_code_values = []
    
    # Separate codes by length for better organization
    codes_by_length = {2: [], 4: [], 6: []}
    for code in unique_for_codes:
        length = len(code)
        if length in codes_by_length:
            codes_by_length[length].append(code)
    
    # Add options organized by hierarchy (divisions first, then groups, then fields)
    for length in [2, 4, 6]:
        for code in sorted(codes_by_length[length]):
            name = for_codes_mapping.get(code, "Unknown")
            # Truncate very long names for better UI
            display_name = name[:80] + "..." if len(name) > 80 else name
            
            # Add prefix to indicate hierarchy level
            if length == 2:
                prefix = "📁 "  # Division
            elif length == 4:
                prefix = "  📂 "  # Group (indented)
            else:
                prefix = "    📄 "  # Field (more indented)
            
            option_text = f"{prefix}{code} - {display_name}"
            for_code_options.append(option_text)
            for_code_values.append(code)
    
    return for_code_options, for_code_values

def has_for_code(for_codes_str, for_primary_val, for_code_filter):
    """Check if a grant has any of the specified FOR codes"""
    if pd.isna(for_codes_str) and pd.isna(for_primary_val):
        return False
    
    # Check primary FOR code
    if not pd.isna(for_primary_val):
        primary_str = str(int(for_primary_val))
        if any(code in primary_str for code in for_code_filter):
            return True
    
    # Check secondary FOR codes
    if not pd.isna(for_codes_str):
        for_codes_list = str(for_codes_str).split(',')
        for filter_code in for_code_filter:
            if any(filter_code in for_code.strip() for for_code in for_codes_list):
                return True
    return False
