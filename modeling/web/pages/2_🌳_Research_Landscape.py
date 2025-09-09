import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from pathlib import Path

# Add the web directory to Python path to import shared_utils
web_dir = str(Path(__file__).parent.parent)
if web_dir not in sys.path:
    sys.path.insert(0, web_dir)

from visualisation import create_research_landscape_treemap
from shared_utils import (
    setup_page_config, 
    load_data,
    clear_previous_page_state
)

# Page configuration
setup_page_config("Research Landscape")

# Clear any previous page state
clear_previous_page_state()

# Load data
keywords, grants, categories = load_data()

# Main content
st.header("Research Landscape Treemap")
st.markdown("Explore the hierarchical structure of research categories and keywords.")

# Clear sidebar and add page-specific controls
st.sidebar.empty()

# Sidebar controls for treemap
with st.sidebar:
    st.subheader("🌳 Treemap Settings")
    
    # Filtering Options in Expander
    with st.expander("🔍 Filtering Options", expanded=True):
        max_research_fields = st.selectbox(
            "Maximum Research Fields",
            options=[None, 5, 10, 15, 20],
            index=2,
            format_func=lambda x: "All" if x is None else str(x),
            help="Maximum number of research fields to include"
        )
        
        max_categories_per_field = st.selectbox(
            "Maximum categories per research field",
            options=[None, 3, 5, 10, 15],
            index=2,
            format_func=lambda x: "All" if x is None else str(x),
            help="Maximum number of categories to show per research field"
        )
        
        max_keywords_per_category = st.selectbox(
            "Maximum keywords per category",
            options=[None, 5, 10, 20, 30],
            index=2,
            format_func=lambda x: "All" if x is None else str(x),
            help="Maximum number of keywords to show per category"
        )
    
    # Display Settings in Expander
    with st.expander("⚙️ Display Settings", expanded=False):
        treemap_height = st.slider(
            "Visualization height",
            min_value=400,
            max_value=1200,
            value=800,
            step=50
        )
        
        font_size = st.slider(
            "Font size",
            min_value=6,
            max_value=16,
            value=9
        )
    
    # Update settings button at bottom of sidebar
    st.markdown("---")
    update_settings = st.button("Update Settings", type="primary", use_container_width=True)

# Auto-generate treemap visualization (on page load or when update button is clicked)
# Use session state to track if this is the first load
if "landscape_initialized" not in st.session_state:
    st.session_state.landscape_initialized = True
    should_generate = True
elif update_settings:
    should_generate = True
else:
    should_generate = False

if should_generate:
    with st.spinner("Creating research landscape treemap..."):
        categories_list = categories.to_dict('records')
        
        fig_treemap = create_research_landscape_treemap(
            categories=categories_list,
            classification_results=[],  # Empty for now
            title="Research Landscape: Research Fields → Categories → Keywords",
            height=treemap_height,
            font_size=font_size,
            max_research_fields=max_research_fields,
            max_categories_per_field=max_categories_per_field,
            max_keywords_per_category=max_keywords_per_category
        )
        
        if fig_treemap is not None:
            st.plotly_chart(fig_treemap, use_container_width=True)
            
            # Debug expander showing underlying data  
            with st.expander("🔍 Debug: View Underlying Data", expanded=False):
                st.subheader("Categories DataFrame")
                st.write(f"**Shape:** {categories.shape}")
                st.dataframe(categories, use_container_width=True)
                
                st.subheader("Categories List (Dict Format)")
                st.write(f"**Length:** {len(categories_list)} categories")
                # Show first few categories as example
                st.json(categories_list[:3] if len(categories_list) >= 3 else categories_list)
            
            # Show some statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Categories", len(categories))
            with col2:
                total_keywords = sum(len(cat.get('keywords', [])) for cat in categories_list)
                st.metric("Total Keywords", total_keywords)
            with col3:
                unique_fields = len(set(cat.get('field_of_research', cat.get('for_code', 'Unknown')) for cat in categories_list))
                st.metric("Unique Research Fields", unique_fields)
        else:
            st.warning("No data available for the selected parameters.")
else:
    st.info("💡 Adjust settings in the sidebar and click 'Update Settings' to generate a new visualization.")
