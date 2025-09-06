import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from webutils import load_data, load_for_codes_mapping, get_unique_for_codes, create_for_code_options, has_for_code

st.set_page_config(page_title="Grant Distributions", page_icon="📊", layout="wide")

st.markdown("# 📊 Grant Distributions")
st.sidebar.header("Grant Distributions")
st.markdown("Visualize the distribution of research grants by year with filtering capabilities.")

# Load data
keywords, grants, categories = load_data()
for_codes_mapping = load_for_codes_mapping()

if keywords is None or grants is None or categories is None:
    st.error("Failed to load data. Please check your data files.")
    st.stop()

# Sidebar controls for grant distributions
with st.sidebar:
    st.subheader("Grant Distribution Settings")
    
    st.markdown("### Grant Filters")
    
    # Get unique values for filter options
    unique_funders_grants = sorted(grants['funder'].dropna().unique().tolist())
    unique_sources_grants = sorted(grants['source'].dropna().unique().tolist())
    unique_for_codes_grants = get_unique_for_codes()
    for_code_options_grants, for_code_values_grants = create_for_code_options(unique_for_codes_grants, for_codes_mapping)
    option_to_code_grants = dict(zip(for_code_options_grants, for_code_values_grants))
    
    # Filter by funder
    funder_filter_grants = st.multiselect(
        "Filter by Funder",
        options=unique_funders_grants,
        default=[],
        help="Select specific funders to filter grants by (leave empty for all funders)"
    )
    
    # Filter by source
    source_filter_grants = st.multiselect(
        "Filter by Source",
        options=unique_sources_grants,
        default=[],
        help="Select specific sources to filter grants by (leave empty for all sources)"
    )
    
    # Filter by FOR codes
    st.markdown("**FOR (Field of Research) Codes:**")
    st.markdown("📁 = Divisions (2-digit) | 📂 = Groups (4-digit) | 📄 = Fields (6-digit)")
    
    selected_for_code_options_grants = st.multiselect(
        "Select FOR codes to filter by",
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
            # Apply FOR code filtering
            mask = filtered_grants.apply(
                lambda row: has_for_code(row.get('for'), row.get('for_primary'), for_code_filter_grants), axis=1
            )
            filtered_grants = filtered_grants[mask]
        
        if len(filtered_grants) == 0:
            st.error("No grants found matching your filter criteria. Please adjust your filters.")
        else:
            # Create year distribution visualization
            if 'start_year' in filtered_grants.columns:
                year_counts = filtered_grants['start_year'].value_counts().sort_index()
                
                # Create bar chart
                fig_dist = go.Figure(data=[
                    go.Bar(
                        x=year_counts.index,
                        y=year_counts.values,
                        name='Grant Count'
                    )
                ])
                
                fig_dist.update_layout(
                    title=f"Grant Distribution by Year ({len(filtered_grants)} total grants)",
                    xaxis_title="Year",
                    yaxis_title="Number of Grants",
                    height=500
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Additional statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Grants", len(filtered_grants))
                
                with col2:
                    if 'start_year' in filtered_grants.columns:
                        min_year = int(filtered_grants['start_year'].min())
                        max_year = int(filtered_grants['start_year'].max())
                        st.metric("Year Range", f"{min_year}-{max_year}")
                
                with col3:
                    if 'funder' in filtered_grants.columns:
                        unique_funders_count = len(filtered_grants['funder'].unique())
                        st.metric("Unique Funders", unique_funders_count)
                
                with col4:
                    if 'start_year' in filtered_grants.columns:
                        avg_grants_per_year = len(filtered_grants) / (max_year - min_year + 1)
                        st.metric("Avg Grants/Year", f"{avg_grants_per_year:.1f}")
                
                # Show filtered grants table
                st.subheader("Filtered Grants")
                display_columns = []
                for col in ['id', 'title', 'funder', 'start_year', 'for_primary']:
                    if col in filtered_grants.columns:
                        display_columns.append(col)
                
                if display_columns:
                    st.dataframe(filtered_grants[display_columns], use_container_width=True)
                else:
                    st.dataframe(filtered_grants.head(100), use_container_width=True)
            
            else:
                st.error("No 'start_year' column found in grants data. Cannot create year distribution.")
