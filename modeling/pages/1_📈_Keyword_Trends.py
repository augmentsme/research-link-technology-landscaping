import streamlit as st
import pandas as pd
import random
import numpy as np
from webutils import load_data
from visualisation import create_keyword_trends_visualization

st.set_page_config(page_title="Keyword Trends", page_icon="📈", layout="wide")

st.markdown("# 📈 Keyword Trends")
st.sidebar.header("Keyword Trends")
st.markdown("Analyze how research keywords evolve over time with cumulative occurrence tracking.")

# Load data
keywords, grants, categories = load_data()

if keywords is None or grants is None or categories is None:
    st.error("Failed to load data. Please check your data files.")
    st.stop()

# Sidebar controls for keyword trends
with st.sidebar:
    st.subheader("Keyword Trends Settings")
    
    # Get unique values for filter options
    unique_funders = sorted(grants['funder'].dropna().unique().tolist())
    unique_sources = sorted(grants['source'].dropna().unique().tolist())
    unique_keyword_types = sorted(keywords['type'].dropna().unique().tolist())
    
    st.markdown("### Grant Filters")
    
    # Filter by funder
    funder_filter = st.multiselect(
        "Filter by Funder",
        options=unique_funders,
        default=[],
        help="Select specific funders to filter grants by (leave empty for all funders)"
    )
    
    # Filter by source
    source_filter = st.multiselect(
        "Filter by Source",
        options=unique_sources,
        default=[],
        help="Select specific sources to filter grants by (leave empty for all sources)"
    )
    
    # Filter by keyword type
    keyword_type_filter = st.multiselect(
        "Filter by Keyword Type",
        options=unique_keyword_types,
        default=[],
        help="Select specific keyword types (leave empty for all types)"
    )
    
    # Display active filters
    active_filters = []
    if funder_filter:
        active_filters.append(f"Funders: {', '.join(funder_filter[:3])}" + ("..." if len(funder_filter) > 3 else ""))
    if source_filter:
        active_filters.append(f"Sources: {', '.join(source_filter[:3])}" + ("..." if len(source_filter) > 3 else ""))
    if keyword_type_filter:
        active_filters.append(f"Types: {', '.join(keyword_type_filter)}")
        
    if active_filters:
        st.info(f"Active filters: {' | '.join(active_filters)}")
    
    st.markdown("### Display Settings")
    
    min_count = st.slider(
        "Minimum keyword count", 
        min_value=1, 
        max_value=50, 
        value=10,
        help="Minimum number of occurrences for a keyword to be included"
    )
    
    # Keyword selection method
    keyword_selection_method = st.radio(
        "Keyword selection method",
        options=["Top N keywords", "Random sample", "Specify custom keywords"],
        index=0,
        help="Choose how to select keywords to display"
    )
    
    if keyword_selection_method == "Top N keywords":
        top_n = st.slider(
            "Number of top keywords to show", 
            min_value=5, 
            max_value=50, 
            value=20,
            help="Show the N most frequently occurring keywords"
        )
        custom_keywords = None
    elif keyword_selection_method == "Random sample":
        # Get filtered keywords first to determine available options
        filtered_keywords = keywords.copy()
        if keyword_type_filter:
            filtered_keywords = filtered_keywords[filtered_keywords['type'].isin(keyword_type_filter)]
        
        available_keyword_names = filtered_keywords['name'].tolist()
        random.seed(42)  # For reproducible random sampling
        max_sample = min(20, len(available_keyword_names))
        top_n = st.slider(
            "Number of random keywords to show", 
            min_value=5, 
            max_value=max_sample, 
            value=min(15, max_sample),
            help="Show a random sample of keywords"
        )
        if st.button("🎲 Generate new random sample"):
            random.seed()  # Remove seed to get truly random sample
        
        if available_keyword_names:
            custom_keywords = random.sample(available_keyword_names, min(top_n, len(available_keyword_names)))
            top_n = 0  # Don't use top_n when using custom keywords
        else:
            custom_keywords = None
            top_n = 0
    else:  # Specify custom keywords
        custom_keywords_input = st.text_area(
            "Enter keywords (one per line)",
            height=100,
            help="Enter the specific keywords you want to track, one per line"
        )
        if custom_keywords_input.strip():
            custom_keywords = [kw.strip() for kw in custom_keywords_input.split('\n') if kw.strip()]
            top_n = 0  # Don't use top_n when using custom keywords
        else:
            custom_keywords = None
            top_n = 5  # Fallback to showing top 5
    
    st.markdown("### Visualization Options")
    
    show_average = st.checkbox(
        "Show average trend line", 
        value=True,
        help="Display the average trend across all selected keywords"
    )
    
    show_error_bars = st.checkbox(
        "Show error bars", 
        value=True if show_average else False,
        disabled=not show_average,
        help="Show standard error bars around the average trend line"
    )
    
    use_cumulative = st.checkbox(
        "Use cumulative counts", 
        value=False,
        help="Show cumulative occurrences over time (unchecked shows yearly counts)"
    )
    
    viz_height = st.slider(
        "Visualization height", 
        min_value=400, 
        max_value=1000, 
        value=600,
        help="Height of the trend visualization in pixels"
    )

# Main content area
st.subheader("Keyword Trends Analysis")

# Create the visualization
try:
    fig = create_keyword_trends_visualization(
        keywords_df=keywords,
        grants_df=grants,
        min_count=min_count,
        top_n=top_n,
        title="Keyword Trends Over Time",
        height=viz_height,
        show_average=show_average,
        show_error_bars=show_error_bars,
        custom_keywords=custom_keywords,
        funder_filter=funder_filter if funder_filter else None,
        source_filter=source_filter if source_filter else None,
        keyword_type_filter=keyword_type_filter if keyword_type_filter else None,
        use_cumulative=use_cumulative
    )
    
    if fig is not None:
        st.plotly_chart(fig, width='stretch')
        
        # Display selected keywords information
        if custom_keywords:
            st.subheader("Selected Keywords")
            if len(custom_keywords) <= 20:
                st.write(f"**Keywords ({len(custom_keywords)}):** {', '.join(custom_keywords)}")
            else:
                st.write(f"**{len(custom_keywords)} keywords selected:** {', '.join(custom_keywords[:10])}, ... and {len(custom_keywords) - 10} more")
        
        # Display methodology info
        with st.expander("📊 Methodology & Data Info"):
            st.markdown(f"""
            **Data Processing:**
            - Minimum keyword count threshold: {min_count}
            - Visualization type: {'Cumulative' if use_cumulative else 'Yearly'} occurrences
            - Keywords displayed: {keyword_selection_method.lower()}
            
            **Filtering Applied:**
            - Funder filter: {'Applied' if funder_filter else 'None'}
            - Source filter: {'Applied' if source_filter else 'None'}
            - Keyword type filter: {'Applied' if keyword_type_filter else 'None'}
            
            **Interpretation:**
            - Each data point represents the occurrence of keywords in grants starting in that year
            - {'Cumulative view shows the running total over time' if use_cumulative else 'Yearly view shows the count for each individual year'}
            - Average line shows the mean trend across all displayed keywords
            - Error bars represent standard error of the mean
            """)
    else:
        st.warning("No data available for the current filter settings. Try adjusting the filters or reducing the minimum count threshold.")
        
except Exception as e:
    st.error(f"Error creating visualization: {e}")
    st.write("Please check your data and filter settings.")

# Summary statistics
if keywords is not None and grants is not None:
    st.subheader("📈 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Keywords", len(keywords))
    
    with col2:
        st.metric("Total Grants", len(grants))
    
    with col3:
        if grants['start_year'].notna().any():
            year_range = f"{int(grants['start_year'].min())}-{int(grants['start_year'].max())}"
            st.metric("Year Range", year_range)
        else:
            st.metric("Year Range", "N/A")
    
    with col4:
        if funder_filter or source_filter or keyword_type_filter:
            # Calculate filtered counts
            filtered_grants = grants.copy()
            if funder_filter:
                filtered_grants = filtered_grants[filtered_grants['funder'].isin(funder_filter)]
            if source_filter:
                filtered_grants = filtered_grants[filtered_grants['source'].isin(source_filter)]
                
            filtered_keywords = keywords.copy()
            if keyword_type_filter:
                filtered_keywords = filtered_keywords[filtered_keywords['type'].isin(keyword_type_filter)]
                
            st.metric("Filtered Data", f"{len(filtered_keywords)} keywords, {len(filtered_grants)} grants")
        else:
            st.metric("Filters Applied", "None")
