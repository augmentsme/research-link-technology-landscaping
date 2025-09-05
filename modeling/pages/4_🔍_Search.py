import streamlit as st
import pandas as pd
import re
from webutils import load_data

st.set_page_config(page_title="Search", page_icon="🔍")

st.markdown("# 🔍 Search")
st.sidebar.header("Search")
st.markdown("Search through the research keywords and grants database to find specific information.")

# Load data
keywords, grants, categories = load_data()

if keywords is None or grants is None or categories is None:
    st.error("Failed to load data. Please check your data files.")
    st.stop()

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
            # Apply search and filters
            filtered_keywords = keywords.copy()
            
            # Apply keyword type filter
            if keyword_type_filter_search:
                filtered_keywords = filtered_keywords[filtered_keywords['type'].isin(keyword_type_filter_search)]
            
            # Apply minimum grant count filter
            filtered_keywords = filtered_keywords[filtered_keywords.grants.map(len) >= min_grant_count]
            
            # Apply search term filter
            if keyword_search_term:
                if 'term' in filtered_keywords.columns:
                    if use_keyword_regex:
                        try:
                            mask = filtered_keywords['term'].str.contains(keyword_search_term, regex=True, case=False, na=False)
                            filtered_keywords = filtered_keywords[mask]
                        except Exception as e:
                            st.error(f"Invalid regex pattern: {e}")
                            st.stop()
                    else:
                        mask = filtered_keywords['term'].str.contains(keyword_search_term, case=False, na=False)
                        filtered_keywords = filtered_keywords[mask]
                else:
                    # Use index if no 'term' column
                    if use_keyword_regex:
                        try:
                            mask = filtered_keywords.index.str.contains(keyword_search_term, regex=True, case=False, na=False)
                            filtered_keywords = filtered_keywords[mask]
                        except Exception as e:
                            st.error(f"Invalid regex pattern: {e}")
                            st.stop()
                    else:
                        mask = filtered_keywords.index.str.contains(keyword_search_term, case=False, na=False)
                        filtered_keywords = filtered_keywords[mask]
            
            # Display results
            if len(filtered_keywords) > 0:
                st.success(f"Found {len(filtered_keywords)} keywords matching your criteria")
                
                # Add grant count column for display
                display_keywords = filtered_keywords.copy()
                display_keywords['grant_count'] = display_keywords['grants'].map(len)
                
                # Sort by grant count (descending) and show relevant columns
                display_keywords = display_keywords.sort_values('grant_count', ascending=False)
                
                if 'term' in display_keywords.columns:
                    columns_to_show = ['term', 'type', 'grant_count'] if 'type' in display_keywords.columns else ['term', 'grant_count']
                else:
                    # Create term column from index
                    display_keywords['term'] = display_keywords.index
                    columns_to_show = ['term', 'type', 'grant_count'] if 'type' in display_keywords.columns else ['term', 'grant_count']
                
                st.dataframe(display_keywords[columns_to_show], use_container_width=True)
            else:
                st.warning("No keywords found matching your search criteria. Try adjusting your search terms or filters.")

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
            if use_grant_id_regex:
                mask = grants['id'].str.contains(grant_id_lookup, regex=True, case=False, na=False)
                matching_grants = grants[mask]
            else:
                mask = grants['id'].str.contains(grant_id_lookup, case=False, na=False)
                matching_grants = grants[mask]
            
            if len(matching_grants) > 0:
                st.success(f"Found {len(matching_grants)} grants")
                display_columns = ['id', 'title', 'funder', 'start_year'] if all(col in matching_grants.columns for col in ['id', 'title', 'funder', 'start_year']) else list(matching_grants.columns)[:5]
                st.dataframe(matching_grants[display_columns], use_container_width=True)
            else:
                st.warning("No grants found with that ID pattern")
        except Exception as e:
            st.error(f"Error in grant ID search: {e}")
    
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
            # Apply filters
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
            
            # Apply search term
            if grant_search_term:
                # Search in title and summary/description columns
                search_columns = []
                for col in ['title', 'summary', 'description', 'abstract']:
                    if col in filtered_grants.columns:
                        search_columns.append(col)
                
                if search_columns:
                    combined_text = filtered_grants[search_columns].fillna('').agg(' '.join, axis=1)
                    
                    if use_grant_regex:
                        try:
                            mask = combined_text.str.contains(grant_search_term, regex=True, case=False, na=False)
                            filtered_grants = filtered_grants[mask]
                        except Exception as e:
                            st.error(f"Invalid regex pattern: {e}")
                            st.stop()
                    else:
                        mask = combined_text.str.contains(grant_search_term, case=False, na=False)
                        filtered_grants = filtered_grants[mask]
                else:
                    st.warning("No searchable text columns found in grants data")
            
            # Display results
            if len(filtered_grants) > 0:
                st.success(f"Found {len(filtered_grants)} grants matching your criteria")
                
                # Show results in manageable chunks
                if len(filtered_grants) > 100:
                    st.info(f"Showing first 100 grants out of {len(filtered_grants)} results")
                    display_grants = filtered_grants.head(100)
                else:
                    display_grants = filtered_grants
                
                # Select appropriate columns for display
                display_columns = []
                for col in ['id', 'title', 'funder', 'start_year', 'summary']:
                    if col in display_grants.columns:
                        display_columns.append(col)
                
                if display_columns:
                    st.dataframe(display_grants[display_columns], use_container_width=True)
                else:
                    st.dataframe(display_grants, use_container_width=True)
                    
            else:
                st.warning("No grants found matching your search criteria. Try adjusting your filters or search terms.")
