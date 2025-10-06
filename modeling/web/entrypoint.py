"""
Research Landscape Analysis - Multipage Streamlit Application
Main entry point for the research landscape analysis tool.
"""

import streamlit as st
import pandas as pd
from shared_utils import (load_data, get_unique_funders, 
                          get_unique_sources, get_unique_keyword_types, 
                          get_unique_research_fields_from_categories)

# Configure the main page
st.set_page_config(
    page_title="Research Landscape Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page content
st.title("Research Landscape Analysis")
st.markdown("---")

# Load data and display summary statistics
try:
    with st.spinner("Loading data..."):
        keywords, grants, categories = load_data()
    
    if keywords is not None and grants is not None:
        # Calculate summary statistics
        num_keywords = len(keywords)
        num_grants = len(grants)
        num_categories = len(categories) if categories is not None else 0
        
        # Get unique values for additional stats
        unique_funders = get_unique_funders(grants)
        unique_sources = get_unique_sources(grants)
        unique_keyword_types = get_unique_keyword_types(keywords)
        unique_research_fields = get_unique_research_fields_from_categories(categories)
        
        # Calculate date range if available
        date_range = "N/A"
        if 'year' in grants.columns:
            min_year = grants['year'].min()
            max_year = grants['year'].max()
            date_range = f"{min_year} - {max_year}"
        elif 'date_start' in grants.columns:
            try:
                grants['date_start'] = pd.to_datetime(grants['date_start'])
                min_date = grants['date_start'].min().year
                max_date = grants['date_start'].max().year
                date_range = f"{min_date} - {max_date}"
            except:
                pass
        
        # Display summary statistics
        st.subheader("Dataset Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Keywords", 
                value=f"{num_keywords:,}",
                help="Total number of research keywords in the dataset"
            )
        
        with col2:
            st.metric(
                label="Total Grants", 
                value=f"{num_grants:,}",
                help="Total number of research grants in the dataset"
            )
        
        with col3:
            st.metric(
                label="Unique Funders", 
                value=f"{len(unique_funders):,}",
                help="Number of different funding organizations"
            )
        
        with col4:
            st.metric(
                label="Time Range", 
                value=date_range,
                help="Time period covered by the dataset"
            )
        
        # Additional statistics in a second row
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric(
                label="Keyword Types", 
                value=f"{len(unique_keyword_types):,}",
                help="Different types of keywords (e.g., extracted, manual)"
            )
        
        with col6:
            st.metric(
                label="Data Sources", 
                value=f"{len(unique_sources):,}",
                help="Number of different data sources"
            )
        
        with col7:
            st.metric(
                label="Research Fields", 
                value=f"{len(unique_research_fields):,}",
                help="Number of different research areas"
            )
        
        with col8:
            # Calculate average keywords per grant
            if 'grants' in keywords.columns:
                avg_keywords_per_grant = keywords['grants'].apply(len).mean()
                st.metric(
                    label="Avg Keywords/Grant", 
                    value=f"{avg_keywords_per_grant:.1f}",
                    help="Average number of keywords per grant"
                )
            else:
                st.metric(
                    label="Categories", 
                    value=f"{num_categories:,}",
                    help="Number of research categories"
                )
        
        st.markdown("---")
        
    else:
        st.error("Unable to load data. Please check your data files.")
        st.markdown("---")

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.markdown("---")
