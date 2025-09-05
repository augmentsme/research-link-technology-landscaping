import streamlit as st
from webutils import load_data, load_for_codes_mapping

st.set_page_config(
    page_title="Research Link Technology Landscaping",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Research Link Technology Landscaping")
st.markdown("Interactive visualizations for research landscape analysis")

# Load and display data status
keywords, grants, categories = load_data()
for_codes_mapping = load_for_codes_mapping()

if keywords is None or grants is None or categories is None:
    st.error("Failed to load data. Please check your data files.")
    st.stop()

# Welcome message and navigation guide
st.markdown("""
Welcome to the Research Link Technology Landscaping platform! This interactive tool provides comprehensive visualizations for research landscape analysis.

### 📊 Available Analyses

**👈 Select a page from the sidebar** to explore different visualizations:

- **📈 Keyword Trends**: Analyze how research keywords evolve over time with cumulative occurrence tracking
- **🌳 Research Landscape**: Explore the hierarchical structure of research categories and keywords
- **📊 Grant Distributions**: Visualize the distribution of research grants by year with filtering capabilities  
- **🔍 Search**: Search through keywords and grants database to find specific information

### 📈 Data Overview

""")

# Display summary statistics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Keywords", len(keywords))
    
with col2:
    st.metric("Total Grants", len(grants))
    
with col3:
    st.metric("Total Categories", len(categories))

# Additional information
st.markdown("""
### 🔧 Features

- **Interactive Filtering**: Filter data by funder, source, Field of Research (FOR) codes, and keyword types
- **Multiple Visualization Methods**: Choose from top N keywords, random sampling, or custom keyword selection
- **Export Capabilities**: Download visualizations and data for further analysis
- **Real-time Updates**: All visualizations update dynamically based on your filter selections

### 🚀 Getting Started

1. **Choose a visualization type** from the sidebar navigation
2. **Apply filters** to focus on specific research areas or timeframes
3. **Generate visualizations** using the controls in each page
4. **Explore the data** using interactive charts and tables

---
*Built with Streamlit • Research Link Technology Landscaping Project*
""")

# Quick data preview
with st.expander("🔍 Quick Data Preview", expanded=False):
    st.subheader("Sample Keywords")
    if 'term' in keywords.columns:
        sample_keywords = keywords.head(10)[['term', 'type']].copy() if 'type' in keywords.columns else keywords.head(10)[['term']].copy()
    else:
        sample_keywords = keywords.head(10)
    st.dataframe(sample_keywords)
    
    st.subheader("Sample Grants")
    grant_columns = ['id', 'title', 'funder', 'start_year'] if all(col in grants.columns for col in ['id', 'title', 'funder', 'start_year']) else list(grants.columns)[:4]
    sample_grants = grants.head(10)[grant_columns].copy()
    st.dataframe(sample_grants)
