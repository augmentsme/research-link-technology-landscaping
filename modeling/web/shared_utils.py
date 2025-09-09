"""
Shared utilities for the Research Landscape Analysis multipage app.
Contains common functions, data loading, and configurations used across multiple pages.
"""

import streamlit as st
import sys
import os
import json
import pandas as pd
import random
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Add the parent directory to the Python path to import from the modeling package
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Change working directory to parent to ensure relative paths work
os.chdir(parent_dir)

import config
from visualisation import create_keyword_trends_visualization, create_research_landscape_treemap

# Utility functions for simplified field filtering
def field_to_division_codes(field_names):
    """Convert field names to division codes for filtering"""
    field_to_division = {
        'AGRICULTURAL_VETERINARY_FOOD_SCIENCES': '30',
        'BIOLOGICAL_SCIENCES': '31',
        'BIOMEDICAL_CLINICAL_SCIENCES': '32',
        'BUILT_ENVIRONMENT_DESIGN': '33',
        'CHEMICAL_SCIENCES': '34',
        'COMMERCE_MANAGEMENT_TOURISM_SERVICES': '35',
        'CREATIVE_ARTS_WRITING': '36',
        'EARTH_SCIENCES': '37',
        'ECONOMICS': '38',
        'EDUCATION': '39',
        'ENGINEERING': '40',
        'ENVIRONMENTAL_SCIENCES': '41',
        'HEALTH_SCIENCES': '42',
        'HISTORY_HERITAGE_ARCHAEOLOGY': '43',
        'HUMAN_SOCIETY': '44',
        'INDIGENOUS_STUDIES': '45',
        'INFORMATION_COMPUTING_SCIENCES': '46',
        'LANGUAGE_COMMUNICATION_CULTURE': '47',
        'LAW_LEGAL_STUDIES': '48',
        'MATHEMATICAL_SCIENCES': '49',
        'PHILOSOPHY_RELIGIOUS_STUDIES': '50',
        'PHYSICAL_SCIENCES': '51',
        'PSYCHOLOGY': '52'
    }
    return [field_to_division[field] for field in field_names if field in field_to_division]

def has_research_field_simple(for_primary_val, division_codes):
    """Simplified function to check if primary FOR code matches any division codes"""
    if pd.isna(for_primary_val):
        return False
    
    # Extract 2-digit division code from primary FOR code
    primary_str = str(int(for_primary_val))
    if len(primary_str) >= 2:
        primary_division = primary_str[:2]
        return primary_division in division_codes
    return False

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

def get_unique_values_from_data(keywords, grants, categories):
    """Extract unique values for filters from the data"""
    try:
        # Get unique values for filters
        unique_funders = sorted(grants['funder'].dropna().unique())
        unique_sources = sorted(grants['source'].dropna().unique())
        unique_keyword_types = sorted(keywords['type'].dropna().unique())
        
        # Get unique research fields from categories
        unique_research_fields = []
        if categories is not None and 'field_of_research' in categories.columns:
            unique_research_fields = sorted(categories['field_of_research'].dropna().unique())
        
        return unique_funders, unique_sources, unique_keyword_types, unique_research_fields
    except Exception as e:
        st.error(f"Error extracting unique values: {e}")
        return [], [], [], []

def create_research_field_options(unique_fields):
    """Create research field options for dropdowns with display names"""
    field_display_names = {
        'AGRICULTURAL_VETERINARY_FOOD_SCIENCES': 'Agricultural, Veterinary & Food Sciences',
        'BIOLOGICAL_SCIENCES': 'Biological Sciences',
        'BIOMEDICAL_CLINICAL_SCIENCES': 'Biomedical & Clinical Sciences',
        'BUILT_ENVIRONMENT_DESIGN': 'Built Environment & Design',
        'CHEMICAL_SCIENCES': 'Chemical Sciences',
        'COMMERCE_MANAGEMENT_TOURISM_SERVICES': 'Commerce, Management, Tourism & Services',
        'CREATIVE_ARTS_WRITING': 'Creative Arts & Writing',
        'EARTH_SCIENCES': 'Earth Sciences',
        'ECONOMICS': 'Economics',
        'EDUCATION': 'Education',
        'ENGINEERING': 'Engineering',
        'ENVIRONMENTAL_SCIENCES': 'Environmental Sciences',
        'HEALTH_SCIENCES': 'Health Sciences',
        'HISTORY_HERITAGE_ARCHAEOLOGY': 'History, Heritage & Archaeology',
        'HUMAN_SOCIETY': 'Human Society',
        'INDIGENOUS_STUDIES': 'Indigenous Studies',
        'INFORMATION_COMPUTING_SCIENCES': 'Information & Computing Sciences',
        'LANGUAGE_COMMUNICATION_CULTURE': 'Language, Communication & Culture',
        'LAW_LEGAL_STUDIES': 'Law & Legal Studies',
        'MATHEMATICAL_SCIENCES': 'Mathematical Sciences',
        'PHILOSOPHY_RELIGIOUS_STUDIES': 'Philosophy & Religious Studies',
        'PHYSICAL_SCIENCES': 'Physical Sciences',
        'PSYCHOLOGY': 'Psychology'
    }
    
    field_options = []
    field_values = []
    
    # Sort fields alphabetically by display name
    sorted_fields = sorted(unique_fields, key=lambda x: field_display_names.get(x, x))
    
    for field in sorted_fields:
        display_name = field_display_names.get(field, field.replace('_', ' ').title())
        field_options.append(display_name)
        field_values.append(field)
    
    return field_options, field_values

def setup_page_config(page_title: str, page_icon: str = "🔬"):
    """Setup page configuration for each page"""
    st.set_page_config(
        page_title=f"Research Landscape - {page_title}",
        page_icon=page_icon,
        layout="wide"
    )

def clear_previous_page_state():
    """Clear any lingering state from previous pages to prevent widget conflicts"""
    # Store current page in session state to detect page changes
    current_page = st.get_option("client.toolbarMode")  # This helps detect page context
    
    # Clear any widget states that shouldn't persist across pages
    keys_to_clear = [key for key in st.session_state.keys() 
                     if key.startswith(('filter_', 'search_', 'temp_'))]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
