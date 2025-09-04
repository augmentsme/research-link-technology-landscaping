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

# Add the parent directory to the Python path to import from the modeling package
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Change working directory to parent to ensure relative paths work
os.chdir(parent_dir)

import config
from visualisation import create_keyword_trends_visualization, create_research_landscape_treemap

st.set_page_config(
    page_title="Research Link Technology Landscaping",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Research Link Technology Landscaping")
st.markdown("Interactive visualizations for research landscape analysis")

# Load data
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

keywords, grants, categories = load_data()
for_codes_mapping = load_for_codes_mapping()

if keywords is None or grants is None or categories is None:
    st.error("Failed to load data. Please check your data files.")
    st.stop()

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["📈 Keyword Trends", "🌳 Research Landscape", "📊 Grant Distributions", "🔍 Search"])

# Tab 1: Keyword Trends Visualization
with tab1:
    st.header("Keyword Trends Over Time")
    st.markdown("Analyze how research keywords evolve over time with cumulative occurrence tracking.")
    
    # Sidebar controls for keyword trends
    with st.sidebar:
        st.subheader("Keyword Trends Settings")
        
        # Get unique values for filter options
        unique_funders = sorted(grants['funder'].dropna().unique().tolist())
        unique_sources = sorted(grants['source'].dropna().unique().tolist())
        unique_keyword_types = sorted(keywords['type'].dropna().unique().tolist())
        
        # Get unique FOR codes efficiently - cached computation
        @st.cache_data
        def get_unique_for_codes():
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
        
        unique_for_codes = get_unique_for_codes()
        
        # Create options with both code and name, organized by hierarchy - cached
        @st.cache_data
        def create_for_code_options(unique_for_codes, for_codes_mapping):
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
        
        for_code_options, for_code_values = create_for_code_options(unique_for_codes, for_codes_mapping)
        
        # Create a mapping from display text back to code
        option_to_code = dict(zip(for_code_options, for_code_values))
        
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
        
        # Filter by FOR codes
        st.markdown("**FOR (Field of Research) Codes:**")
        
        selected_for_code_options = st.multiselect(
            "Select FOR codes to filter by",
            options=for_code_options,
            default=[],
            help="FOR codes are organized hierarchically. Selecting a division includes all its groups and fields. Use the search box to quickly find specific research areas."
        )
        
        # Convert selected options back to codes
        for_code_filter = [option_to_code[option] for option in selected_for_code_options]
        
        st.markdown("### Keyword Filters")
        
        # Filter by keyword type
        keyword_type_filter = st.multiselect(
            "Filter by Keyword Type",
            options=unique_keyword_types,
            default=[],
            help="Select specific keyword types to filter by (leave empty for all types)"
        )
        
        # Show current filter status
        active_filters = []
        if funder_filter:
            active_filters.append(f"Funders: {', '.join(funder_filter)}")
        if source_filter:
            active_filters.append(f"Sources: {', '.join(source_filter)}")
        if for_code_filter:
            # Show FOR codes with names in the active filters
            for_display_parts = []
            for code in for_code_filter[:3]:  # Show first 3
                name = for_codes_mapping.get(code, "Unknown")
                short_name = name[:30] + "..." if len(name) > 30 else name
                for_display_parts.append(f"{code} ({short_name})")
            
            for_display = ', '.join(for_display_parts)
            if len(for_code_filter) > 3:
                for_display += f"... and {len(for_code_filter) - 3} more"
            active_filters.append(f"FOR Codes: {for_display}")
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
            index=1,
            help="Choose how to select keywords to display"
        )
        
        if keyword_selection_method == "Top N keywords":
            top_n = st.slider(
                "Number of top keywords to show", 
                min_value=0, 
                max_value=50, 
                value=20,
                help="Number of individual keyword lines to display (0 = show only averages)"
            )
            custom_keywords = []
            random_sample_size = 0
        elif keyword_selection_method == "Random sample":
            top_n = 0  # Don't use top_n when random sampling
            custom_keywords = []
            
            # Get available keywords for sampling based on current filters
            filtered_grants_temp = grants.copy()
            if funder_filter:
                filtered_grants_temp = filtered_grants_temp[filtered_grants_temp['funder'].isin(funder_filter)]
            if source_filter:
                filtered_grants_temp = filtered_grants_temp[filtered_grants_temp['source'].isin(source_filter)]
            if for_code_filter:
                # Apply FOR code filtering
                def has_for_code(for_codes_str, for_primary_val):
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
                
                mask = filtered_grants_temp.apply(
                    lambda row: has_for_code(row.get('for'), row.get('for_primary')), axis=1
                )
                filtered_grants_temp = filtered_grants_temp[mask]
            
            # Apply keyword type filter
            filtered_keywords_temp = keywords.copy()
            if keyword_type_filter:
                filtered_keywords_temp = filtered_keywords_temp[filtered_keywords_temp['type'].isin(keyword_type_filter)]
            
            # Filter keywords to only include those with grants in the filtered set
            if funder_filter or source_filter or for_code_filter:
                filtered_grants_ids = set(filtered_grants_temp['id'])
                filtered_keywords = filtered_keywords_temp[filtered_keywords_temp['grants'].apply(
                    lambda x: len([g for g in x if g in filtered_grants_ids]) > min_count
                )]
            else:
                filtered_keywords = filtered_keywords_temp[filtered_keywords_temp.grants.map(len) > min_count]
                
            max_available = len(filtered_keywords)
            
            if max_available > 0:
                random_sample_size = st.slider(
                    "Number of random keywords to sample",
                    min_value=1,
                    max_value=min(50, max_available),
                    value=min(20, max_available),
                    help=f"Randomly select keywords from {max_available} available keywords"
                )
            else:
                st.warning("No keywords available with current filters. Please adjust your filter settings.")
                random_sample_size = 0
            
            # Add a random seed option for reproducibility
            use_random_seed = st.checkbox(
                "Use random seed for reproducible results",
                value=True,
                help="Enable to get the same random sample each time"
            )
            
            if use_random_seed:
                random_seed = st.number_input(
                    "Random seed",
                    min_value=0,
                    max_value=9999,
                    value=42,
                    help="Seed for random number generator"
                )
            else:
                random_seed = None
                
            # Add a button to generate new random sample
            if st.button("🎲 Generate New Random Sample", help="Click to get a different random selection"):
                if 'random_sample_counter' not in st.session_state:
                    st.session_state.random_sample_counter = 0
                st.session_state.random_sample_counter += 1
        else:
            top_n = 0  # Don't use top_n when custom keywords are specified
            random_sample_size = 0
            
            # Get available keywords for selection based on current filters
            filtered_grants_temp = grants.copy()
            if funder_filter:
                filtered_grants_temp = filtered_grants_temp[filtered_grants_temp['funder'].isin(funder_filter)]
            if source_filter:
                filtered_grants_temp = filtered_grants_temp[filtered_grants_temp['source'].isin(source_filter)]
            if for_code_filter:
                # Apply FOR code filtering
                def has_for_code(for_codes_str, for_primary_val):
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
                
                mask = filtered_grants_temp.apply(
                    lambda row: has_for_code(row.get('for'), row.get('for_primary')), axis=1
                )
                filtered_grants_temp = filtered_grants_temp[mask]
            
            # Apply keyword type filter
            filtered_keywords_temp = keywords.copy()
            if keyword_type_filter:
                filtered_keywords_temp = filtered_keywords_temp[filtered_keywords_temp['type'].isin(keyword_type_filter)]
            
            # Filter keywords to only include those with grants in the filtered set
            if funder_filter or source_filter or for_code_filter:
                filtered_grants_ids = set(filtered_grants_temp['id'])
                keywords_filtered = filtered_keywords_temp[filtered_keywords_temp['grants'].apply(
                    lambda x: len([g for g in x if g in filtered_grants_ids]) > min_count
                )]
            else:
                keywords_filtered = filtered_keywords_temp[filtered_keywords_temp.grants.map(len) > min_count]
            
            # Extract actual keyword terms (handle both 'term' column and index-based keywords)
            if 'term' in keywords_filtered.columns:
                available_keywords = sorted([str(term) for term in keywords_filtered['term'].tolist()])
            else:
                # If no 'term' column, use the index as the keyword terms
                available_keywords = sorted([str(term) for term in keywords_filtered.index.tolist()])
            
            # Multi-select for custom keywords
            custom_keywords = st.multiselect(
                "Select specific keywords to track",
                options=available_keywords,
                default=[],
                help=f"Choose from {len(available_keywords)} keywords that meet the minimum count criteria after filtering"
            )
            
            if custom_keywords:
                st.info(f"Selected {len(custom_keywords)} custom keywords")
            else:
                st.warning("Please select at least one keyword to display trends")
        
        show_average = st.checkbox("Show average line", value=True)
        
        if show_average:
            average_from_population = st.radio(
                "Average calculation method",
                options=[False, True],
                format_func=lambda x: "Sample (Top N)" if not x else "Population (All keywords)",
                index=1,
                help="Choose whether average is calculated from top N keywords or entire population"
            )
            
            show_error_bars = st.checkbox("Show error bars", value=True)
        else:
            average_from_population = False
            show_error_bars = False
        
        show_baseline = st.checkbox(
            "Show baseline (avg keywords/grant)", 
            value=True,
            help="Show cumulative average keywords per grant per year as baseline"
        )
        
        # Toggle for cumulative vs raw counts
        use_cumulative = st.checkbox(
            "Use cumulative counts",
            value=True,
            help="Show cumulative keyword counts over time (checked) or raw yearly counts (unchecked)"
        )
    
    # Generate keyword trends visualization
    if st.button("Generate Keyword Trends", type="primary"):
        # Check if custom keywords are selected when needed
        if keyword_selection_method == "Specify custom keywords" and not custom_keywords:
            st.error("Please select at least one keyword when using custom keyword selection.")
        else:
            with st.spinner("Creating keyword trends visualization..."):
                # Handle random sampling
                if keyword_selection_method == "Random sample":
                    import random
                    import numpy as np
                    
                    # Get available keywords for random sampling (apply filters first)
                    filtered_grants_temp = grants.copy()
                    if funder_filter:
                        filtered_grants_temp = filtered_grants_temp[filtered_grants_temp['funder'].isin(funder_filter)]
                    if source_filter:
                        filtered_grants_temp = filtered_grants_temp[filtered_grants_temp['source'].isin(source_filter)]
                    if for_code_filter:
                        # Apply FOR code filtering
                        def has_for_code(for_codes_str, for_primary_val):
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
                        
                        mask = filtered_grants_temp.apply(
                            lambda row: has_for_code(row.get('for'), row.get('for_primary')), axis=1
                        )
                        filtered_grants_temp = filtered_grants_temp[mask]
                    
                    # Apply keyword type filter
                    filtered_keywords_temp = keywords.copy()
                    if keyword_type_filter:
                        filtered_keywords_temp = filtered_keywords_temp[filtered_keywords_temp['type'].isin(keyword_type_filter)]
                    
                    # Filter keywords based on filtered grants
                    if funder_filter or source_filter or for_code_filter:
                        filtered_grants_ids = set(filtered_grants_temp['id'])
                        filtered_keywords = filtered_keywords_temp[filtered_keywords_temp['grants'].apply(
                            lambda x: len([g for g in x if g in filtered_grants_ids]) > min_count
                        )]
                    else:
                        filtered_keywords = filtered_keywords_temp[filtered_keywords_temp.grants.map(len) > min_count]
                    
                    # Extract keyword terms
                    if 'term' in filtered_keywords.columns:
                        all_available_keywords = [str(term) for term in filtered_keywords['term'].tolist()]
                    else:
                        all_available_keywords = [str(term) for term in filtered_keywords.index.tolist()]
                    
                    # Check if we have keywords to sample
                    if len(all_available_keywords) == 0 or random_sample_size == 0:
                        st.error("No keywords available for random sampling with current filters. Please adjust your filter settings.")
                        st.stop()
                    
                    # Set random seed if specified
                    if use_random_seed:
                        # Include counter in seed to allow new samples even with fixed seed
                        effective_seed = random_seed + st.session_state.get('random_sample_counter', 0)
                        random.seed(effective_seed)
                        np.random.seed(effective_seed)
                    
                    # Randomly sample keywords
                    sampled_keywords = random.sample(
                        all_available_keywords, 
                        min(random_sample_size, len(all_available_keywords))
                    )
                    custom_keywords_for_viz = sampled_keywords
                    count_type = "Cumulative" if use_cumulative else "Yearly"
                    viz_title = f"{count_type} Random Sample of Keywords ({len(sampled_keywords)} keywords)"
                    
                elif keyword_selection_method == "Top N keywords":
                    custom_keywords_for_viz = None
                    count_type = "Cumulative" if use_cumulative else "Yearly"
                    viz_title = f"{count_type} Keyword Trends (min_count={min_count}, top_n={top_n})"
                else:
                    custom_keywords_for_viz = custom_keywords
                    count_type = "Cumulative" if use_cumulative else "Yearly"
                    viz_title = f"{count_type} Keyword Trends - Custom Selection ({len(custom_keywords)} keywords)"
                
                # Add filter information to title
                filter_parts = []
                if funder_filter:
                    filter_parts.append(f"Funders: {', '.join(funder_filter)}")
                if source_filter:
                    filter_parts.append(f"Sources: {', '.join(source_filter)}")
                if for_code_filter:
                    # Show FOR codes with short names in title
                    for_title_parts = []
                    for code in for_code_filter[:2]:  # Show first 2 in title
                        name = for_codes_mapping.get(code, "Unknown")
                        short_name = name[:20] + "..." if len(name) > 20 else name
                        for_title_parts.append(f"{code} ({short_name})")
                    
                    for_display = ', '.join(for_title_parts)
                    if len(for_code_filter) > 2:
                        for_display += f"... +{len(for_code_filter) - 2} more"
                    filter_parts.append(f"FOR: {for_display}")
                if keyword_type_filter:
                    filter_parts.append(f"Types: {', '.join(keyword_type_filter)}")
                if filter_parts:
                    viz_title += f" | Filtered by {' | '.join(filter_parts)}"
                
                fig_trends = create_keyword_trends_visualization(
                    keywords_df=keywords,
                    grants_df=grants,
                    min_count=min_count,
                    top_n=top_n if keyword_selection_method == "Top N keywords" else 0,
                    title=viz_title,
                    height=600,
                    show_average=show_average,
                    show_error_bars=show_error_bars,
                    average_from_population=average_from_population,
                    show_baseline=show_baseline,
                    custom_keywords=custom_keywords_for_viz,
                    funder_filter=funder_filter if funder_filter else None,
                    source_filter=source_filter if source_filter else None,
                    keyword_type_filter=keyword_type_filter if keyword_type_filter else None,
                    for_code_filter=for_code_filter if for_code_filter else None,
                    use_cumulative=use_cumulative
                )
                
                if fig_trends is not None:
                    st.plotly_chart(fig_trends, use_container_width=True)
                    
                    # Debug expander showing underlying data
                    with st.expander("🔍 Debug: View Underlying Data", expanded=False):
                        st.subheader("Filtered Grants Data")
                        # Apply same filters as used in visualization
                        debug_grants = grants.copy()
                        if funder_filter:
                            debug_grants = debug_grants[debug_grants['funder'].isin(funder_filter)]
                        if source_filter:
                            debug_grants = debug_grants[debug_grants['source'].isin(source_filter)]
                        if for_code_filter:
                            def has_for_code_debug(for_codes_str, for_primary_val):
                                if pd.isna(for_codes_str) and pd.isna(for_primary_val):
                                    return False
                                if not pd.isna(for_primary_val):
                                    primary_str = str(int(for_primary_val))
                                    if any(code in primary_str for code in for_code_filter):
                                        return True
                                if not pd.isna(for_codes_str):
                                    for_codes_list = str(for_codes_str).split(',')
                                    for filter_code in for_code_filter:
                                        if any(filter_code in for_code.strip() for for_code in for_codes_list):
                                            return True
                                return False
                            mask = debug_grants.apply(
                                lambda row: has_for_code_debug(row.get('for'), row.get('for_primary')), axis=1
                            )
                            debug_grants = debug_grants[mask]
                        
                        st.write(f"**Shape:** {debug_grants.shape}")
                        st.dataframe(debug_grants, use_container_width=True)
                        
                        st.subheader("Filtered Keywords Data")  
                        debug_keywords = keywords.copy()
                        if keyword_type_filter:
                            debug_keywords = debug_keywords[debug_keywords['type'].isin(keyword_type_filter)]
                        
                        # Filter keywords based on grants
                        if funder_filter or source_filter or for_code_filter:
                            filtered_grants_ids = set(debug_grants['id'])
                            debug_keywords = debug_keywords[debug_keywords['grants'].apply(
                                lambda x: len([g for g in x if g in filtered_grants_ids]) > 0
                            )]
                        
                        st.write(f"**Shape:** {debug_keywords.shape}")
                        st.dataframe(debug_keywords, use_container_width=True)
                    
                    # Calculate filtered grants and keywords count
                    filtered_grants_count = len(grants)
                    filtered_keywords_count = len(keywords)
                    
                    # Apply grant filters
                    filtered_grants_temp = grants.copy()
                    if funder_filter:
                        filtered_grants_temp = filtered_grants_temp[filtered_grants_temp['funder'].isin(funder_filter)]
                    if source_filter:
                        filtered_grants_temp = filtered_grants_temp[filtered_grants_temp['source'].isin(source_filter)]
                    if for_code_filter:
                        # Apply FOR code filtering
                        def has_for_code(for_codes_str, for_primary_val):
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
                        
                        mask = filtered_grants_temp.apply(
                            lambda row: has_for_code(row.get('for'), row.get('for_primary')), axis=1
                        )
                        filtered_grants_temp = filtered_grants_temp[mask]
                    
                    filtered_grants_count = len(filtered_grants_temp)
                    
                    # Apply keyword filters
                    filtered_keywords_temp = keywords.copy()
                    if keyword_type_filter:
                        filtered_keywords_temp = filtered_keywords_temp[filtered_keywords_temp['type'].isin(keyword_type_filter)]
                    
                    # Filter keywords based on filtered grants
                    if funder_filter or source_filter or for_code_filter:
                        filtered_grants_ids = set(filtered_grants_temp['id'])
                        filtered_keywords_temp = filtered_keywords_temp[filtered_keywords_temp['grants'].apply(
                            lambda x: len([g for g in x if g in filtered_grants_ids]) > min_count
                        )]
                    else:
                        filtered_keywords_temp = filtered_keywords_temp[filtered_keywords_temp.grants.map(len) > min_count]
                    
                    filtered_keywords_count = len(filtered_keywords_temp)
                    
                    # Show some statistics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Total Keywords", len(keywords))
                    with col2:
                        st.metric("Filtered Keywords", filtered_keywords_count)
                    with col3:
                        st.metric("Total Grants", len(grants))
                    with col4:
                        st.metric("Filtered Grants", filtered_grants_count)
                    with col5:
                        if keyword_selection_method == "Top N keywords" and top_n > 0:
                            st.metric("Displayed Keywords", min(top_n, filtered_keywords_count))
                        elif keyword_selection_method == "Random sample":
                            st.metric("Displayed Keywords", len(sampled_keywords))
                        elif keyword_selection_method == "Specify custom keywords":
                            st.metric("Displayed Keywords", len(custom_keywords))
                        else:
                            st.metric("Displayed Keywords", 0)
                            
                    # Show selected keywords
                    if keyword_selection_method == "Random sample":
                        st.subheader("Randomly Selected Keywords:")
                        st.write(", ".join(str(kw) for kw in sampled_keywords))
                        if use_random_seed:
                            st.info(f"Random seed used: {effective_seed}")
                    elif keyword_selection_method == "Specify custom keywords" and custom_keywords:
                        st.subheader("Selected Keywords:")
                        st.write(", ".join(str(kw) for kw in custom_keywords))
                else:
                    st.warning("No data available for the selected parameters.")

# Tab 2: Research Landscape Treemap
with tab2:
    st.header("Research Landscape Treemap")
    st.markdown("Explore the hierarchical structure of research categories and keywords.")
    
    # Sidebar controls for treemap
    with st.sidebar:
        st.subheader("Treemap Settings")
        
        max_for_classes = st.selectbox(
            "Maximum FOR classes",
            options=[None, 5, 10, 15, 20],
            index=2,
            format_func=lambda x: "All" if x is None else str(x),
            help="Maximum number of FOR (Field of Research) classes to include"
        )
        
        max_categories_per_for = st.selectbox(
            "Maximum categories per FOR class",
            options=[None, 3, 5, 10, 15],
            index=2,
            format_func=lambda x: "All" if x is None else str(x),
            help="Maximum number of categories to show per FOR class"
        )
        
        max_keywords_per_category = st.selectbox(
            "Maximum keywords per category",
            options=[None, 5, 10, 20, 30],
            index=2,
            format_func=lambda x: "All" if x is None else str(x),
            help="Maximum number of keywords to show per category"
        )
        
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
    
    # Generate treemap visualization
    if st.button("Generate Research Landscape", type="primary"):
        with st.spinner("Creating research landscape treemap..."):
            categories_list = categories.to_dict('records')
            
            fig_treemap = create_research_landscape_treemap(
                categories=categories_list,
                classification_results=[],  # Empty for now
                title="Research Landscape: FOR Classes → Categories → Keywords",
                height=treemap_height,
                font_size=font_size,
                max_for_classes=max_for_classes,
                max_categories_per_for=max_categories_per_for,
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
                    unique_fors = len(set(cat.get('for_code', 'Unknown') for cat in categories_list))
                    st.metric("Unique FOR Classes", unique_fors)
            else:
                st.warning("No data available for the selected parameters.")

# Tab 3: Grant Distributions
with tab3:
    st.header("Grant Distributions Over Time")
    st.markdown("Visualize the distribution of research grants by year with filtering capabilities.")
    
    # Sidebar controls for grant distributions
    with st.sidebar:
        st.subheader("Grant Distribution Settings")
        
        st.markdown("### Grant Filters")
        
        # Get unique values for filter options (reuse from tab1 cached functions)
        unique_funders_grants = sorted(grants['funder'].dropna().unique().tolist())
        unique_sources_grants = sorted(grants['source'].dropna().unique().tolist())
        unique_for_codes_grants = get_unique_for_codes()
        for_code_options_grants, for_code_values_grants = create_for_code_options(unique_for_codes_grants, for_codes_mapping)
        option_to_code_grants = dict(zip(for_code_options_grants, for_code_values_grants))
        
        # Filter by funder
        funder_filter_grants = st.multiselect(
            "Filter by Funder (Grants)",
            options=unique_funders_grants,
            default=[],
            help="Select specific funders to filter grants by (leave empty for all funders)"
        )
        
        # Filter by source
        source_filter_grants = st.multiselect(
            "Filter by Source (Grants)",
            options=unique_sources_grants,
            default=[],
            help="Select specific sources to filter grants by (leave empty for all sources)"
        )
        
        # Filter by FOR codes
        st.markdown("**FOR (Field of Research) Codes (Grants):**")
        st.markdown("📁 = Divisions (2-digit) | 📂 = Groups (4-digit) | 📄 = Fields (6-digit)")
        
        selected_for_code_options_grants = st.multiselect(
            "Select FOR codes to filter by (Grants)",
            options=for_code_options_grants,
            default=[],
            help="FOR codes are organized hierarchically. Selecting a division includes all its groups and fields."
        )
        
        # Convert selected options back to codes
        for_code_filter_grants = [option_to_code_grants[option] for option in selected_for_code_options_grants]
        
        # Show current filter status for grants
        active_filters_grants = []
        if funder_filter_grants:
            active_filters_grants.append(f"Funders: {', '.join(funder_filter_grants)}")
        if source_filter_grants:
            active_filters_grants.append(f"Sources: {', '.join(source_filter_grants)}")
        if for_code_filter_grants:
            # Show FOR codes with names in the active filters
            for_display_parts_grants = []
            for code in for_code_filter_grants[:3]:  # Show first 3
                name = for_codes_mapping.get(code, "Unknown")
                short_name = name[:30] + "..." if len(name) > 30 else name
                for_display_parts_grants.append(f"{code} ({short_name})")
            
            for_display_grants = ', '.join(for_display_parts_grants)
            if len(for_code_filter_grants) > 3:
                for_display_grants += f"... and {len(for_code_filter_grants) - 3} more"
            active_filters_grants.append(f"FOR Codes: {for_display_grants}")
            
        if active_filters_grants:
            st.info(f"Active filters: {' | '.join(active_filters_grants)}")
    
    # Generate grant distribution visualization
    if st.button("Generate Grant Distribution", type="primary"):
        with st.spinner("Creating grant distribution visualization..."):
            # Apply filters to grants
            filtered_grants = grants.copy()
            
            # Apply funder filter
            if funder_filter_grants:
                filtered_grants = filtered_grants[filtered_grants['funder'].isin(funder_filter_grants)]
            
            # Apply source filter
            if source_filter_grants:
                filtered_grants = filtered_grants[filtered_grants['source'].isin(source_filter_grants)]
            
            # Apply FOR code filter
            if for_code_filter_grants:
                def has_for_code_grants(for_codes_str, for_primary_val):
                    if pd.isna(for_codes_str) and pd.isna(for_primary_val):
                        return False
                    
                    # Check primary FOR code
                    if not pd.isna(for_primary_val):
                        primary_str = str(int(for_primary_val))
                        if any(code in primary_str for code in for_code_filter_grants):
                            return True
                    
                    # Check secondary FOR codes
                    if not pd.isna(for_codes_str):
                        for_codes_list = str(for_codes_str).split(',')
                        for filter_code in for_code_filter_grants:
                            if any(filter_code in for_code.strip() for for_code in for_codes_list):
                                return True
                    return False
                
                mask = filtered_grants.apply(
                    lambda row: has_for_code_grants(row.get('for'), row.get('for_primary')), axis=1
                )
                filtered_grants = filtered_grants[mask]
            
            if len(filtered_grants) == 0:
                st.error("No grants found with the current filters. Please adjust your filter settings.")
            else:
                # Count grants per year
                grants_per_year = filtered_grants.groupby('start_year').size().reset_index(name='count')
                
                # Create line chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=grants_per_year['start_year'],
                    y=grants_per_year['count'],
                    mode='lines+markers',
                    name='Grants per Year',
                    line=dict(width=3, color='#1f77b4'),
                    marker=dict(size=8, color='#1f77b4'),
                    hovertemplate='<b>Year:</b> %{x}<br><b>Grants:</b> %{y}<extra></extra>'
                ))
                
                # Create title with filter information
                title_parts = ["Grant Distribution by Year"]
                if funder_filter_grants:
                    title_parts.append(f"Funders: {', '.join(funder_filter_grants)}")
                if source_filter_grants:
                    title_parts.append(f"Sources: {', '.join(source_filter_grants)}")
                if for_code_filter_grants:
                    for_title_parts = []
                    for code in for_code_filter_grants[:2]:  # Show first 2 in title
                        name = for_codes_mapping.get(code, "Unknown")
                        short_name = name[:20] + "..." if len(name) > 20 else name
                        for_title_parts.append(f"{code} ({short_name})")
                    
                    for_display = ', '.join(for_title_parts)
                    if len(for_code_filter_grants) > 2:
                        for_display += f"... +{len(for_code_filter_grants) - 2} more"
                    title_parts.append(f"FOR: {for_display}")
                
                title = title_parts[0] if len(title_parts) == 1 else f"{title_parts[0]} | Filtered by {' | '.join(title_parts[1:])}"
                
                fig.update_layout(
                    title=title,
                    xaxis_title='Year',
                    yaxis_title='Number of Grants',
                    height=600,
                    hovermode='x unified',
                    showlegend=False
                )
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Debug expander showing underlying data
                with st.expander("🔍 Debug: View Underlying Data", expanded=False):
                    st.subheader("Filtered Grants DataFrame")
                    st.write(f"**Shape:** {filtered_grants.shape}")
                    st.dataframe(filtered_grants, use_container_width=True)
                    
                    st.subheader("Grants Per Year (Chart Data)")
                    st.write(f"**Shape:** {grants_per_year.shape}")  
                    st.dataframe(grants_per_year, use_container_width=True)
                
                # Show statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Grants", len(grants))
                with col2:
                    st.metric("Filtered Grants", len(filtered_grants))
                with col3:
                    st.metric("Year Range", f"{grants_per_year['start_year'].min()}-{grants_per_year['start_year'].max()}")
                with col4:
                    st.metric("Peak Year", f"{grants_per_year.loc[grants_per_year['count'].idxmax(), 'start_year']} ({grants_per_year['count'].max()} grants)")

# Tab 4: Search
with tab4:
    st.header("Search Keywords and Grants")
    st.markdown("Search through the research keywords and grants database to find specific information.")
    
    # Search controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Search Keywords")
        
        # Keyword search
        keyword_search_term = st.text_input(
            "Search keyword terms",
            placeholder="e.g., machine learning, artificial intelligence, ^neural.*",
            help="Search for keywords by term name. Use regex patterns like ^neural.* or .*learning$ for advanced matching"
        )
        
        # Regex toggle for keyword search
        use_keyword_regex = st.checkbox(
            "Use regex for keyword search",
            value=False,
            help="Enable regular expression patterns in keyword search"
        )
        
        # Keyword type filter
        unique_keyword_types_search = sorted(keywords['type'].dropna().unique().tolist())
        keyword_type_filter_search = st.multiselect(
            "Filter by keyword type",
            options=unique_keyword_types_search,
            default=[],
            help="Filter keywords by their type"
        )
        
        # Minimum grant count for keywords
        min_grant_count = st.slider(
            "Minimum number of grants",
            min_value=1,
            max_value=50,
            value=5,
            help="Show keywords that appear in at least this many grants"
        )
        
        # Search keywords button
        if st.button("Search Keywords", type="primary"):
            with st.spinner("Searching keywords..."):
                # Start with all keywords
                filtered_keywords = keywords.copy()
                
                # Apply keyword type filter
                if keyword_type_filter_search:
                    filtered_keywords = filtered_keywords[filtered_keywords['type'].isin(keyword_type_filter_search)]
                
                # Apply grant count filter
                filtered_keywords = filtered_keywords[filtered_keywords['grants'].apply(len) >= min_grant_count]
                
                # Apply text search
                if keyword_search_term:
                    try:
                        # Check if keyword has 'term' column
                        if 'term' in filtered_keywords.columns:
                            if use_keyword_regex:
                                mask = filtered_keywords['term'].astype(str).str.contains(
                                    keyword_search_term, case=False, na=False, regex=True
                                )
                            else:
                                mask = filtered_keywords['term'].astype(str).str.contains(
                                    keyword_search_term, case=False, na=False, regex=False
                                )
                            filtered_keywords = filtered_keywords[mask]
                        else:
                            # If no 'term' column, search in index
                            if use_keyword_regex:
                                mask = filtered_keywords.index.astype(str).str.contains(
                                    keyword_search_term, case=False, na=False, regex=True
                                )
                            else:
                                mask = filtered_keywords.index.astype(str).str.contains(
                                    keyword_search_term, case=False, na=False, regex=False
                                )
                            filtered_keywords = filtered_keywords[mask]
                    except Exception as e:
                        if use_keyword_regex:
                            st.error(f"Invalid regex pattern: {e}")
                            st.stop()
                        else:
                            # If not using regex but still got an error, show it
                            st.error(f"Search error: {e}")
                            st.stop()
                
                # Display results
                st.subheader(f"Keywords Search Results ({len(filtered_keywords)} found)")
                
                if len(filtered_keywords) > 0:
                    # Add display limit warning and controls
                    MAX_DISPLAY_KEYWORDS = 1000
                    if len(filtered_keywords) > MAX_DISPLAY_KEYWORDS:
                        st.warning(f"⚠️ Found {len(filtered_keywords)} keywords, displaying first {MAX_DISPLAY_KEYWORDS} results. Use more specific filters to see all results.")
                        display_limit = MAX_DISPLAY_KEYWORDS
                    else:
                        display_limit = len(filtered_keywords)
                    
                    # Add a column showing grant count for easier viewing
                    display_keywords = filtered_keywords.copy()
                    display_keywords['grant_count'] = display_keywords['grants'].apply(len)
                    
                    # Sort by grant count descending
                    display_keywords = display_keywords.sort_values('grant_count', ascending=False)
                    
                    # Apply display limit
                    display_keywords_limited = display_keywords.head(display_limit)
                    
                    st.dataframe(display_keywords_limited, use_container_width=True, height=400)
                    
                    if len(filtered_keywords) > display_limit:
                        st.info(f"📊 Showing {display_limit} of {len(filtered_keywords)} results. Refine your search to see specific keywords.")
                    
                    # Summary stats
                    col1_stat, col2_stat, col3_stat = st.columns(3)
                    with col1_stat:
                        st.metric("Total Keywords Found", len(filtered_keywords))
                    with col2_stat:
                        avg_grants = display_keywords['grant_count'].mean()
                        st.metric("Avg Grants per Keyword", f"{avg_grants:.1f}")
                    with col3_stat:
                        max_grants = display_keywords['grant_count'].max()
                        st.metric("Max Grants per Keyword", max_grants)
                else:
                    st.warning("No keywords found matching your search criteria. Try adjusting your filters or search terms.")
    
    with col2:
        st.subheader("🎓 Search Grants")
        
        # Direct Grant ID lookup
        st.markdown("**Quick Grant Lookup:**")
        grant_id_lookup = st.text_input(
            "Look up grant by ID",
            placeholder="Enter grant ID (e.g., arc/DP220100606) or regex pattern (e.g., ^arc/DP.*)",
            help="Enter exact grant ID or use regex patterns like ^arc/.* for advanced matching"
        )
        
        # Regex toggle for grant ID lookup
        use_grant_id_regex = st.checkbox(
            "Use regex for grant ID lookup",
            value=False,
            help="Enable regular expression patterns in grant ID lookup"
        )
        
        if grant_id_lookup:
            try:
                # Try to find the grant by ID
                if use_grant_id_regex:
                    matching_grants = grants[grants['id'].astype(str).str.contains(
                        grant_id_lookup, case=False, na=False, regex=True
                    )]
                else:
                    matching_grants = grants[grants['id'].astype(str) == str(grant_id_lookup)]
                
                if len(matching_grants) > 0:
                    if use_grant_id_regex:
                        st.success(f"✅ Found {len(matching_grants)} grant(s) matching pattern: {grant_id_lookup}")
                    else:
                        st.success(f"✅ Found grant with ID: {grant_id_lookup}")
                    
                    # Display the grant details
                    for idx, (_, grant_details) in enumerate(matching_grants.iterrows()):
                        if len(matching_grants) > 1:
                            st.markdown(f"**Grant {idx + 1}:**")
                        
                        # Create a nice display of grant information
                        with st.container():
                            st.markdown(f"**ID:** {grant_details['id']}")
                            st.markdown(f"**Title:** {grant_details['title']}")
                            if 'grant_summary' in grant_details and pd.notna(grant_details['grant_summary']):
                                st.markdown(f"**Summary:** {grant_details['grant_summary']}")
                            
                            # Display other grant details in columns
                            detail_col1, detail_col2, detail_col3 = st.columns(3)
                            
                            with detail_col1:
                                if 'start_year' in grant_details and pd.notna(grant_details['start_year']):
                                    st.metric("Year", int(grant_details['start_year']))
                            
                            with detail_col2:
                                if 'funder' in grant_details and pd.notna(grant_details['funder']):
                                    st.metric("Funder", grant_details['funder'])
                            
                            with detail_col3:
                                if 'funding_amount' in grant_details and pd.notna(grant_details['funding_amount']):
                                    st.metric("Funding", f"${grant_details['funding_amount']:,.0f}")
                        
                        if len(matching_grants) > 1:
                            st.markdown("---")
                    
                    # Show full grant details in expander
                    with st.expander("📄 View All Grant Details", expanded=False):
                        st.dataframe(matching_grants, use_container_width=True)
                        
                elif grant_id_lookup.strip():  # Only show error if user actually entered something
                    if use_grant_id_regex:
                        st.error(f"❌ No grants found matching pattern: {grant_id_lookup}")
                    else:
                        st.error(f"❌ No grant found with ID: {grant_id_lookup}")
            except Exception as e:
                if use_grant_id_regex:
                    st.error(f"Invalid regex pattern: {e}")
                else:
                    st.error(f"Lookup error: {e}")
        
        st.markdown("---")
        st.markdown("**Advanced Grant Search:**")
        
        # Grant search
        grant_search_term = st.text_input(
            "Search grant titles/descriptions",
            placeholder="e.g., climate change, neural networks, ^.*machine.*learning.*$",
            help="Search for grants by title or summary. Use regex patterns for advanced matching"
        )
        
        # Regex toggle for grant search
        use_grant_regex = st.checkbox(
            "Use regex for grant search",
            value=False,
            help="Enable regular expression patterns in grant title/description search"
        )
        
        # Grant filters
        unique_funders_search = sorted(grants['funder'].dropna().unique().tolist())
        funder_filter_search = st.multiselect(
            "Filter by funder",
            options=unique_funders_search,
            default=[],
            help="Filter grants by funding organization"
        )
        
        unique_sources_search = sorted(grants['source'].dropna().unique().tolist())
        source_filter_search = st.multiselect(
            "Filter by source",
            options=unique_sources_search,
            default=[],
            help="Filter grants by data source"
        )
        
        # Year range filter
        if 'start_year' in grants.columns:
            min_year = int(grants['start_year'].min())
            max_year = int(grants['start_year'].max())
            year_range = st.slider(
                "Year range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                help="Filter grants by year range"
            )
        else:
            year_range = None
        
        # Search grants button
        if st.button("Search Grants", type="primary"):
            with st.spinner("Searching grants..."):
                # Start with all grants
                filtered_grants = grants.copy()
                
                # Apply funder filter
                if funder_filter_search:
                    filtered_grants = filtered_grants[filtered_grants['funder'].isin(funder_filter_search)]
                
                # Apply source filter
                if source_filter_search:
                    filtered_grants = filtered_grants[filtered_grants['source'].isin(source_filter_search)]
                
                # Apply year range filter
                if year_range and 'start_year' in filtered_grants.columns:
                    filtered_grants = filtered_grants[
                        (filtered_grants['start_year'] >= year_range[0]) & 
                        (filtered_grants['start_year'] <= year_range[1])
                    ]
                
                # Apply text search
                if grant_search_term:
                    try:
                        # Search in title and grant_summary columns
                        if use_grant_regex:
                            title_mask = filtered_grants['title'].astype(str).str.contains(
                                grant_search_term, case=False, na=False, regex=True
                            )
                        else:
                            title_mask = filtered_grants['title'].astype(str).str.contains(
                                grant_search_term, case=False, na=False, regex=False
                            )
                        
                        summary_mask = pd.Series([False] * len(filtered_grants))
                        if 'grant_summary' in filtered_grants.columns:
                            if use_grant_regex:
                                summary_mask = filtered_grants['grant_summary'].astype(str).str.contains(
                                    grant_search_term, case=False, na=False, regex=True
                                )
                            else:
                                summary_mask = filtered_grants['grant_summary'].astype(str).str.contains(
                                    grant_search_term, case=False, na=False, regex=False
                                )
                        
                        # Combine masks with OR logic
                        text_mask = title_mask | summary_mask
                        filtered_grants = filtered_grants[text_mask]
                    except Exception as e:
                        if use_grant_regex:
                            st.error(f"Invalid regex pattern: {e}")
                            st.stop()
                        else:
                            # If not using regex but still got an error, show it
                            st.error(f"Search error: {e}")
                            st.stop()
                
                # Display results
                st.subheader(f"Grants Search Results ({len(filtered_grants)} found)")
                
                if len(filtered_grants) > 0:
                    # Add display limit warning and controls
                    MAX_DISPLAY_GRANTS = 500
                    if len(filtered_grants) > MAX_DISPLAY_GRANTS:
                        st.warning(f"⚠️ Found {len(filtered_grants)} grants, displaying first {MAX_DISPLAY_GRANTS} results. Use more specific filters to see all results.")
                        display_limit = MAX_DISPLAY_GRANTS
                    else:
                        display_limit = len(filtered_grants)
                    
                    # Sort by start_year descending if available
                    if 'start_year' in filtered_grants.columns:
                        filtered_grants = filtered_grants.sort_values('start_year', ascending=False)
                    
                    # Apply display limit
                    filtered_grants_limited = filtered_grants.head(display_limit)
                    
                    # Select key columns for display
                    display_columns = ['title', 'start_year', 'funder', 'source']
                    if 'funding_amount' in filtered_grants.columns:
                        display_columns.insert(2, 'funding_amount')
                    
                    # Only show columns that exist
                    display_columns = [col for col in display_columns if col in filtered_grants.columns]
                    
                    st.dataframe(
                        filtered_grants_limited[display_columns], 
                        use_container_width=True, 
                        height=400
                    )
                    
                    if len(filtered_grants) > display_limit:
                        st.info(f"📊 Showing {display_limit} of {len(filtered_grants)} results. Refine your search to see specific grants.")
                    
                    # Summary stats (calculated from all results, not just displayed ones)
                    col1_stat, col2_stat, col3_stat = st.columns(3)
                    with col1_stat:
                        st.metric("Total Grants Found", len(filtered_grants))
                    with col2_stat:
                        if year_range and 'start_year' in filtered_grants.columns:
                            year_span = f"{filtered_grants['start_year'].min()}-{filtered_grants['start_year'].max()}"
                            st.metric("Year Span", year_span)
                        else:
                            st.metric("Year Span", "N/A")
                    with col3_stat:
                        if 'funding_amount' in filtered_grants.columns:
                            total_funding = filtered_grants['funding_amount'].sum()
                            if pd.notna(total_funding) and total_funding > 0:
                                st.metric("Total Funding", f"${total_funding:,.0f}")
                            else:
                                st.metric("Total Funding", "N/A")
                        else:
                            st.metric("Total Funding", "N/A")
                    
                    # Show full grant details in expander
                    with st.expander("📄 View Full Grant Details", expanded=False):
                        if len(filtered_grants) > display_limit:
                            st.info(f"Showing full details for first {display_limit} grants (sorted by most recent)")
                            st.dataframe(filtered_grants_limited, use_container_width=True)
                        else:
                            st.dataframe(filtered_grants, use_container_width=True)
                        
                else:
                    st.warning("No grants found matching your search criteria. Try adjusting your filters or search terms.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Research Link Technology Landscaping Project")

