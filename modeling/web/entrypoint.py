"""
Research Landscape Analysis - Multipage Streamlit Application
Main entry point for the research landscape analysis tool.
"""

import streamlit as st
import pandas as pd
from shared_utils import setup_page_config, load_data, get_unique_values_from_data

# Configure the main page
st.set_page_config(
    page_title="Research Landscape Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page content
st.title("🔬 Research Landscape Analysis")
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
        unique_funders, unique_sources, unique_keyword_types, unique_research_fields = get_unique_values_from_data(keywords, grants, categories)
        
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
        st.subheader("📊 Dataset Summary")
        
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

st.markdown("""
## Welcome to the Research Landscape Analysis Tool

This application provides comprehensive analysis and visualization of research data, including:

### 📊 Available Analysis Pages:

- **📈 Keyword Trends** - Analyze how research keywords evolve over time with cumulative occurrence tracking
- **🌳 Research Landscape** - Interactive treemap visualization of research areas and their relationships  
- **📊 Grant Distributions** - Explore grant distribution patterns across time and funding sources

### 🚀 Getting Started:

1. **Navigate** using the sidebar to select an analysis page
2. **Filter** your data using the filtering options in each page's sidebar
3. **Customize** visualizations using the display settings
4. **Interact** with the plots to explore your data in detail

### 💡 Features:

- **Auto-generation**: Plots generate automatically when you visit each page
- **Interactive Filtering**: Real-time filtering by funders, sources, research fields, and keyword types
- **Organized Controls**: All settings are grouped in collapsible sidebar sections
- **Export Ready**: High-quality visualizations suitable for reports and presentations

---

**👈 Select a page from the sidebar to begin your analysis!**
""")

# Add some useful information in the sidebar
with st.sidebar:
    st.markdown("### 📋 Quick Navigation")
    st.markdown("""
    - **Keyword Trends**: Time-series analysis
    - **Research Landscape**: Hierarchical overview  
    - **Grant Distributions**: Funding patterns
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 Tips")
    st.markdown("""
    - Use filters to focus on specific areas
    - Plots update automatically on page load
    - Click 'Update Settings' for manual refresh
    - Hover over visualizations for details
    """)
    
    # Show data status in sidebar
    if 'keywords' in locals() and 'grants' in locals():
        if keywords is not None and grants is not None:
            st.markdown("---")
            st.markdown("### ✅ Data Status")
            st.success("Data loaded successfully")
        else:
            st.markdown("---")
            st.markdown("### ❌ Data Status")
            st.error("Data loading failed")
