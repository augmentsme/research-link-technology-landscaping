import streamlit as st
import pandas as pd
import random
import numpy as np
from webutils import load_data, load_for_codes_mapping, get_unique_for_codes, create_for_code_options, has_for_code
from visualisation import create_keyword_trends_visualization

st.set_page_config(page_title="Keyword Trends", page_icon="📈")

st.markdown("# 📈 Keyword Trends")
st.sidebar.header("Keyword Trends")
st.markdown("Analyze how research keywords evolve over time with cumulative occurrence tracking.")

# Load data
keywords, grants, categories = load_data()
for_codes_mapping = load_for_codes_mapping()

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
    
    # Get unique FOR codes efficiently - cached computation
    unique_for_codes = get_unique_for_codes()
    
    # Create options with both code and name, organized by hierarchy - cached
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
            mask = filtered_grants_temp.apply(
                lambda row: has_for_code(row.get('for'), row.get('for_primary'), for_code_filter), axis=1
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
            mask = filtered_grants_temp.apply(
                lambda row: has_for_code(row.get('for'), row.get('for_primary'), for_code_filter), axis=1
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
                # Get available keywords for random sampling (apply filters first)
                filtered_grants_temp = grants.copy()
                if funder_filter:
                    filtered_grants_temp = filtered_grants_temp[filtered_grants_temp['funder'].isin(funder_filter)]
                if source_filter:
                    filtered_grants_temp = filtered_grants_temp[filtered_grants_temp['source'].isin(source_filter)]
                if for_code_filter:
                    # Apply FOR code filtering
                    mask = filtered_grants_temp.apply(
                        lambda row: has_for_code(row.get('for'), row.get('for_primary'), for_code_filter), axis=1
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
                    for_display += f"... and {len(for_code_filter) - 2} more"
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
            else:
                st.error("Failed to create keyword trends visualization. Please check your data and try again.")
