import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path

# Add the web directory to Python path to import shared_utils
web_dir = str(Path(__file__).parent.parent)
if web_dir not in sys.path:
    sys.path.insert(0, web_dir)

from shared_utils import (
    setup_page_config, 
    load_data,
    clear_previous_page_state
)

# Page configuration
setup_page_config("Search")

# Clear any previous page state
clear_previous_page_state()

# Load data
keywords, grants, categories = load_data()

# Main content
st.header("Search Keywords and Grants")
st.markdown("Search through the research keywords and grants database to find specific information.")

# Search controls
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Search Keywords")
    
    # Keyword search
    keyword_search_term = st.text_input(
        "Search keyword names",
        placeholder="e.g., machine learning, artificial intelligence, ^neural.*",
        help="Search for keywords by name. Use regex patterns like ^neural.* or .*learning$ for advanced matching"
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
                    # Check if keyword has 'name' column (was renamed from 'term')
                    if 'name' in filtered_keywords.columns:
                        if use_keyword_regex:
                            mask = filtered_keywords['name'].astype(str).str.contains(
                                keyword_search_term, case=False, na=False, regex=True
                            )
                        else:
                            mask = filtered_keywords['name'].astype(str).str.contains(
                                keyword_search_term, case=False, na=False, regex=False
                            )
                        filtered_keywords = filtered_keywords[mask]
                    else:
                        # If no 'name' column, search in index
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
