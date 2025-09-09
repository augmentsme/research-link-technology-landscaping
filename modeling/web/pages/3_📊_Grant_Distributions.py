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

from shared_utils import (
    setup_page_config, 
    load_data,
    field_to_division_codes, 
    has_research_field_simple,
    get_unique_research_fields,
    create_research_field_options,
    clear_previous_page_state
)

# Page configuration
setup_page_config("Grant Distributions")

# Clear any previous page state
clear_previous_page_state()

# Load data
keywords, grants, categories = load_data()

# Main content
st.header("Grant Distributions Over Time")
st.markdown("Visualize the distribution of research grants by year with filtering capabilities.")

# Clear sidebar and add page-specific controls
st.sidebar.empty()

# Sidebar controls for grant distributions
with st.sidebar:
    st.subheader("📊 Grant Distribution Settings")
    
    # Filtering Options in Expander
    with st.expander("🔍 Filtering Options", expanded=True):
        st.markdown("**Grant Filters**")
        
        # Get unique values for filter options
        unique_funders_grants = sorted(grants['funder'].dropna().unique().tolist())
        unique_sources_grants = sorted(grants['source'].dropna().unique().tolist())
        unique_research_fields_grants = get_unique_research_fields()
        field_options_grants, field_values_grants = create_research_field_options(unique_research_fields_grants)
        option_to_field_grants = dict(zip(field_options_grants, field_values_grants))
        
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
        
        # Filter by research fields
        st.markdown("**Research Fields (Grants):**")
        
        selected_field_options_grants = st.multiselect(
            "Select research fields to filter by (Grants)",
            options=field_options_grants,
            default=[],
            help="Select specific research areas to filter grants by. Leave empty to include all research fields."
        )
        
        # Convert selected options back to codes
        field_filter_grants = [option_to_field_grants[option] for option in selected_field_options_grants]
        
        # Show current filter status for grants
        active_filters_grants = []
        if funder_filter_grants:
            active_filters_grants.append(f"Funders: {', '.join(funder_filter_grants)}")
        if source_filter_grants:
            active_filters_grants.append(f"Sources: {', '.join(source_filter_grants)}")
        if field_filter_grants:
            # Show research fields with friendly names in the active filters
            field_display_parts_grants = []
            for field in field_filter_grants[:3]:  # Show first 3
                # Convert field name to display name
                display_name = field.replace('_', ' ').title()
                if display_name.startswith('Agricultural'):
                    display_name = 'Agricultural, Veterinary & Food Sciences'
                elif display_name.startswith('Biomedical'):
                    display_name = 'Biomedical & Clinical Sciences'
                elif display_name.startswith('Built'):
                    display_name = 'Built Environment & Design'
                elif display_name.startswith('Commerce'):
                    display_name = 'Commerce, Management, Tourism & Services'
                elif display_name.startswith('Creative'):
                    display_name = 'Creative Arts & Writing'
                elif display_name.startswith('History'):
                    display_name = 'History, Heritage & Archaeology'
                elif display_name.startswith('Information'):
                    display_name = 'Information & Computing Sciences'
                elif display_name.startswith('Language'):
                    display_name = 'Language, Communication & Culture'
                elif display_name.startswith('Law'):
                    display_name = 'Law & Legal Studies'
                elif display_name.startswith('Philosophy'):
                    display_name = 'Philosophy & Religious Studies'
                
                short_name = display_name[:30] + "..." if len(display_name) > 30 else display_name
                field_display_parts_grants.append(short_name)
            
            field_display_grants = ', '.join(field_display_parts_grants)
            if len(field_filter_grants) > 3:
                field_display_grants += f"... and {len(field_filter_grants) - 3} more"
            active_filters_grants.append(f"Research Fields: {field_display_grants}")
            
        if active_filters_grants:
            st.info(f"Active filters: {' | '.join(active_filters_grants)}")
    
    # Update settings button at bottom of sidebar
    st.markdown("---")
    update_settings = st.button("Update Settings", type="primary", use_container_width=True)

# Auto-generate grant distribution visualization (on page load or when update button is clicked)
# Use session state to track if this is the first load
if "distributions_initialized" not in st.session_state:
    st.session_state.distributions_initialized = True
    should_generate = True
elif update_settings:
    should_generate = True
else:
    should_generate = False

if should_generate:
    with st.spinner("Creating grant distribution visualization..."):
        # Apply filters to grants
        filtered_grants = grants.copy()
        
        # Apply funder filter
        if funder_filter_grants:
            filtered_grants = filtered_grants[filtered_grants['funder'].isin(funder_filter_grants)]
        
        # Apply source filter
        if source_filter_grants:
            filtered_grants = filtered_grants[filtered_grants['source'].isin(source_filter_grants)]
        
        # Apply research field filter
        if field_filter_grants:
            # Convert field names to division codes for filtering  
            division_codes_grants = field_to_division_codes(field_filter_grants)
            # Apply simplified research field filtering (primary FOR code only)
            mask = filtered_grants['for_primary'].apply(
                lambda x: has_research_field_simple(x, division_codes_grants)
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
            if field_filter_grants:
                field_title_parts = []
                for field in field_filter_grants[:2]:  # Show first 2 in title
                    # Convert field name to display name
                    display_name = field.replace('_', ' ').title()
                    if display_name.startswith('Agricultural'):
                        display_name = 'Agricultural, Veterinary & Food Sciences'
                    elif display_name.startswith('Biomedical'):
                        display_name = 'Biomedical & Clinical Sciences'
                    elif display_name.startswith('Built'):
                        display_name = 'Built Environment & Design'
                    elif display_name.startswith('Commerce'):
                        display_name = 'Commerce, Management, Tourism & Services'
                    elif display_name.startswith('Creative'):
                        display_name = 'Creative Arts & Writing'
                    elif display_name.startswith('History'):
                        display_name = 'History, Heritage & Archaeology'
                    elif display_name.startswith('Information'):
                        display_name = 'Information & Computing Sciences'
                    elif display_name.startswith('Language'):
                        display_name = 'Language, Communication & Culture'
                    elif display_name.startswith('Law'):
                        display_name = 'Law & Legal Studies'
                    elif display_name.startswith('Philosophy'):
                        display_name = 'Philosophy & Religious Studies'
                    
                    short_name = display_name[:25] + "..." if len(display_name) > 25 else display_name
                    field_title_parts.append(short_name)
                
                field_display = ', '.join(field_title_parts)
                if len(field_filter_grants) > 2:
                    field_display += f"... +{len(field_filter_grants) - 2} more"
                title_parts.append(f"Research Fields: {field_display}")
            
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
else:
    st.info("💡 Adjust settings in the sidebar and click 'Update Settings' to generate a new visualization.")
