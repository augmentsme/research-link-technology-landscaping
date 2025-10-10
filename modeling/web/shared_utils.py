"""
Shared utilities for the Research Landscape Analysis multipage app.
Contains common functions, data loading, and configurations used across multiple pages.
"""

import streamlit as st
import sys
import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pathlib

# Add the parent directory to the Python path to import from the modeling package
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Change working directory to parent to ensure relative paths work
os.chdir(parent_dir)

import config

# Utility functions for simplified field filtering
def format_research_field(field: Optional[str]) -> str:
    """Format a research field value for display."""
    if field is None or pd.isna(field) or str(field).strip() == "":
        return "Unspecified"
    return str(field)

# Load data with caching
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
def load_field_display_names():
    """Load and create FOR codes to name mapping"""
    try:
        with open('data/for_codes_cleaned.json', 'r') as f:
            for_codes_data = json.load(f)
        
        field_display_names = {}
        for code_str, details in for_codes_data.items():
            if isinstance(details, dict) and 'name' in details:
                # Map from FOR code to display name for the dropdown
                field_display_names[code_str] = details['name']
        
        return field_display_names
    except Exception as e:
        st.error(f"Error loading FOR codes data: {e}")
        return {}

@st.cache_data
def get_unique_funders(grants):
    """Get sorted list of unique funders from grants data"""
    return sorted(grants['funder'].dropna().unique())

@st.cache_data
def get_unique_sources(grants):
    """Get sorted list of unique sources from grants data"""
    return sorted(grants['source'].dropna().unique())

@st.cache_data
def get_unique_keyword_types(keywords):
    """Get sorted list of unique keyword types from keywords data"""
    return sorted(keywords['type'].dropna().unique())

@st.cache_data
def get_unique_research_fields_from_categories(categories):
    """Get sorted list of unique research fields from categories data"""
    if categories is not None and 'field_of_research' in categories.columns:
        return sorted(categories['field_of_research'].dropna().unique())
    return []

@st.cache_data
def get_unique_research_fields():
    """Get unique research subjects from grants data - cached computation"""
    try:
        _, grants, _ = load_data()
        if grants is None or 'primary_subject' not in grants.columns:
            return set()
        subjects = grants['primary_subject'].dropna().unique()
        return set(str(subject) for subject in subjects if str(subject).strip())
    except Exception as e:
        st.error(f"Error getting unique research subjects: {e}")
        return set()

def create_research_field_options(unique_fields):
    """Create research field options for dropdowns with display names"""
    sorted_fields = sorted(unique_fields, key=lambda x: format_research_field(x))
    field_options = [format_research_field(field) for field in sorted_fields]
    field_values = list(sorted_fields)
    return field_options, field_values

# Function to load CSS from the 'static' folder
def load_css():
    # Load Tailwind-enhanced CSS
    css_path = pathlib.Path("web/static/css/tailwind-enhanced.css")
    with open(css_path) as f:
        st.html(f"<style>{f.read()}</style>")
