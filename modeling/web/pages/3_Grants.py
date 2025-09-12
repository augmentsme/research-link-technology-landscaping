"""
Grants Analysis Page
Analyze grant distributions over time and view grants with their extracted keywords.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict
from annotated_text import annotated_text

# Add the web directory to Python path to import shared_utils
web_dir = str(Path(__file__).parent.parent)
if web_dir not in sys.path:
    sys.path.insert(0, web_dir)

from shared_utils import (
    setup_page_config, 
    load_data,
    field_to_division_codes, 
    has_research_field_simple,
    has_research_field_all_codes,
    get_unique_research_fields,
    create_research_field_options,
    clear_previous_page_state
)


@dataclass
class FilterConfig:
    """Configuration for data filtering"""
    source_filter: List[str] 
    field_filter: List[str]
    use_all_for_codes: bool
    start_year_min: Optional[int] = None
    start_year_max: Optional[int] = None


@dataclass
class SearchConfig:
    """Configuration for grant search"""
    search_term: str
    max_results: int


class FilterManager:
    """Manages all data filtering operations"""
    
    def __init__(self, grants_df: pd.DataFrame):
        self.grants_df = grants_df
    
    def apply_filters(self, config: FilterConfig) -> pd.DataFrame:
        """Apply filters to grants dataframe"""
        filtered_grants = self.grants_df.copy()
        
        # Apply source filter
        if config.source_filter:
            filtered_grants = filtered_grants[filtered_grants['source'].isin(config.source_filter)]
        
        # Apply research field filter
        if config.field_filter:
            division_codes = field_to_division_codes(config.field_filter)
            
            if config.use_all_for_codes:
                # Apply filtering based on all FOR codes in the 'for' field
                mask = filtered_grants['for'].apply(
                    lambda x: has_research_field_all_codes(x, division_codes)
                )
            else:
                # Apply simplified research field filtering (primary FOR code only)
                mask = filtered_grants['for_primary'].apply(
                    lambda x: has_research_field_simple(x, division_codes)
                )
            
            filtered_grants = filtered_grants[mask]
        
        # Apply date range filter
        if config.start_year_min is not None:
            filtered_grants = filtered_grants[filtered_grants['start_year'] >= config.start_year_min]
        
        if config.start_year_max is not None:
            filtered_grants = filtered_grants[filtered_grants['start_year'] <= config.start_year_max]
        
        return filtered_grants


class SidebarControls:
    """Handles all sidebar UI controls"""
    
    def __init__(self, grants_df: pd.DataFrame):
        self.grants_df = grants_df
        
        # Get unique values for filter options
        self.unique_sources = sorted(grants_df['source'].dropna().unique().tolist())
        
        # Get year range for date filtering
        valid_years = grants_df['start_year'].dropna()
        self.min_year = int(valid_years.min()) if len(valid_years) > 0 else 2000
        self.max_year = int(valid_years.max()) if len(valid_years) > 0 else 2025
        
        # Get research field options
        unique_research_fields = get_unique_research_fields()
        self.field_options, self.field_values = create_research_field_options(unique_research_fields)
        self.option_to_field = dict(zip(self.field_options, self.field_values))
    
    def render_sidebar(self) -> Tuple[FilterConfig, int]:
        """Render all sidebar controls and return configuration"""
        st.sidebar.empty()
        
        with st.sidebar:
            st.subheader("Grant Analysis Settings")
            
            filter_config = self._render_filtering_options()
            max_funders = self._render_chart_options()
            
            # Display active filters
            self._show_active_filters(filter_config)
            
            st.markdown("---")
            
            return filter_config, max_funders
    
    def _render_filtering_options(self) -> FilterConfig:
        """Render filtering options section"""
        with st.expander("Filtering Options", expanded=True):
            st.markdown("**Grant Filters**")
            
            # Filter by source
            source_filter = st.multiselect(
                "Filter by Source",
                options=self.unique_sources,
                default=[],
                help="Select specific sources to filter grants by (leave empty for all sources)",
                key="grants_source_filter"
            )
            
            # Filter by research fields
            st.markdown("**Research Fields:**")
            
            selected_field_options = st.multiselect(
                "Select research fields to filter by",
                options=self.field_options,
                default=[],
                help="Select specific research areas to filter grants by. Leave empty to include all research fields.",
                key="grants_research_fields_filter"
            )
            
            # Convert selected options back to codes
            field_filter = [self.option_to_field[option] for option in selected_field_options]
            
            # Add toggle for FOR code filtering method
            st.markdown("**FOR Code Filtering Method:**")
            use_all_for_codes = st.checkbox(
                "Include all FOR codes (not just primary)",
                value=False,
                help="When checked, filters grants based on all FOR codes in the 'for' field. When unchecked, only uses the primary FOR code.",
                key="grants_for_code_method"
            )
            
            # Date range filter
            st.markdown("**Year Range Filter:**")
            
            col1, col2 = st.columns(2)
            with col1:
                start_year_min = st.number_input(
                    "Start Year (Min)",
                    min_value=self.min_year,
                    max_value=self.max_year,
                    value=2000,
                    step=1,
                    help="Minimum start year for grants",
                    key="grants_start_year_min"
                )
            
            with col2:
                start_year_max = st.number_input(
                    "Start Year (Max)",
                    min_value=self.min_year,
                    max_value=self.max_year,
                    value=self.max_year,
                    step=1,
                    help="Maximum start year for grants",
                    key="grants_start_year_max"
                )
            
            # Ensure min <= max
            if start_year_min > start_year_max:
                st.error("Minimum start year cannot be greater than maximum start year.")
                start_year_min = start_year_max
        
        return FilterConfig(
            source_filter, 
            field_filter, 
            use_all_for_codes,
            start_year_min,
            start_year_max
        )
    
    def _render_chart_options(self) -> int:
        """Render chart display options"""
        with st.expander("Chart Display Options", expanded=False):
            st.markdown("**Funder Display Settings:**")
            
            max_funders = st.slider(
                "Maximum funders to display",
                min_value=3,
                max_value=20,
                value=10,
                step=1,
                help="Controls how many individual funders are shown. Remaining funders will be grouped into 'Others'.",
                key="grants_max_funders_display"
            )
            
            return max_funders
    
    def _show_active_filters(self, config: FilterConfig):
        """Display active filters information"""
        active_filters = []
        
        if config.source_filter:
            source_display = ', '.join(config.source_filter[:3])
            if len(config.source_filter) > 3:
                source_display += f"... and {len(config.source_filter) - 3} more"
            active_filters.append(f"Sources: {source_display}")
            
        if config.field_filter:
            field_display_parts = []
            for field in config.field_filter[:3]:  # Show first 3
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
                field_display_parts.append(short_name)
            
            field_display = ', '.join(field_display_parts)
            if len(config.field_filter) > 3:
                field_display += f"... and {len(config.field_filter) - 3} more"
            
            # Add filtering method info
            filter_method = "All FOR codes" if config.use_all_for_codes else "Primary FOR code only"
            active_filters.append(f"Research Fields ({filter_method}): {field_display}")
        
        # Add date range filter info
        if config.start_year_min is not None or config.start_year_max is not None:
            if config.start_year_min == config.start_year_max:
                active_filters.append(f"Year: {config.start_year_min}")
            else:
                active_filters.append(f"Years: {config.start_year_min}-{config.start_year_max}")
            
        if active_filters:
            st.info(f"Active filters: {' | '.join(active_filters)}")


class GrantDistributionVisualizer:
    """Manages the grant distribution visualization"""
    
    def __init__(self, grants_df: pd.DataFrame):
        self.grants_df = grants_df
        self.filter_manager = FilterManager(grants_df)
    
    def render_tab(self, filter_config: FilterConfig, max_funders: int):
        """Render the grant distribution visualization tab"""
        st.markdown("Visualize the distribution of research grants by year with filtering capabilities.")
        
        should_generate = self._should_generate_visualization()
        
        if should_generate:
            self._generate_visualization(filter_config, max_funders)
        else:
            st.info("💡 Adjust settings in the sidebar to customize the visualization.")
    
    def _should_generate_visualization(self) -> bool:
        """Determine if visualization should be generated"""
        if "distributions_initialized" not in st.session_state:
            st.session_state.distributions_initialized = True
            return True
        return True  # Always generate when filters change
    
    def _generate_visualization(self, filter_config: FilterConfig, max_funders: int):
        """Generate and display the visualization"""
        with st.spinner("Creating grant distribution visualization..."):
            # Apply filters to grants
            filtered_grants = self.filter_manager.apply_filters(filter_config)
            
            if len(filtered_grants) == 0:
                st.error("No grants found with the current filters. Please adjust your filter settings.")
                return
            
            # Group grants by year and funder
            grants_by_year_funder = filtered_grants.groupby(['start_year', 'funder']).size().reset_index(name='count')
            
            # Create stacked area chart
            fig = self._create_stacked_area_chart(grants_by_year_funder, filter_config, max_funders)
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Show statistics
            self._show_statistics(filtered_grants, grants_by_year_funder)
            
            # Debug expander showing underlying data
            self._show_debug_data(filtered_grants, grants_by_year_funder)
    
    def _create_stacked_area_chart(self, grants_by_year_funder: pd.DataFrame, filter_config: FilterConfig, max_funders: int) -> go.Figure:
        """Create the stacked area chart showing funders by color"""
        # Create a pivot table with years as index and funders as columns
        pivot_data = grants_by_year_funder.pivot(index='start_year', columns='funder', values='count').fillna(0)
        
        # Sort funders by total grants (descending) for better stacking order
        funder_totals = pivot_data.sum().sort_values(ascending=False)
        
        # Limit the number of funders displayed
        if len(funder_totals) > max_funders:
            # Keep top funders
            top_funders = funder_totals.head(max_funders - 1).index.tolist()
            
            # Group remaining funders into "Others"
            other_funders = funder_totals.tail(len(funder_totals) - max_funders + 1).index.tolist()
            
            # Calculate "Others" values by summing remaining funders
            others_data = pivot_data[other_funders].sum(axis=1)
            
            # Create new pivot data with limited funders + Others
            limited_pivot = pivot_data[top_funders].copy()
            limited_pivot['Others'] = others_data
            
            # Update the list of funders to display
            sorted_funders = top_funders + ['Others']
            pivot_data = limited_pivot
        else:
            sorted_funders = funder_totals.index.tolist()
        
        # Get years range
        years = pivot_data.index.tolist()
        
        # Define a color palette for funders
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'
        ]
        
        # Use a distinct color for "Others" if it exists
        if 'Others' in sorted_funders:
            colors = colors[:-1] + ['#cccccc']  # Gray color for Others
        
        fig = go.Figure()
        
        # Create traces for each funder
        for i, funder in enumerate(sorted_funders):
            color = colors[i % len(colors)]
            
            # For stacked area chart, use fill='tonexty' except for the first trace
            fill_mode = 'tozeroy' if i == 0 else 'tonexty'
            
            # Special handling for "Others" category
            if funder == 'Others':
                hover_template = f'<b>{funder}</b> ({len(other_funders)} funders)<br>Year: %{{x}}<br>Grants: %{{y}}<extra></extra>'
            else:
                hover_template = f'<b>{funder}</b><br>Year: %{{x}}<br>Grants: %{{y}}<extra></extra>'
            
            fig.add_trace(go.Scatter(
                x=years,
                y=pivot_data[funder].values,
                name=funder,
                mode='lines',
                fill=fill_mode,
                line=dict(width=0.5, color=color),
                fillcolor=color,
                hovertemplate=hover_template,
                stackgroup='one'  # This ensures proper stacking
            ))
        
        # Create title with filter information
        title = self._create_chart_title(filter_config)
        
        # Add information about funders display limit if applicable
        if len(funder_totals) > max_funders:
            title += f" (Top {max_funders - 1} funders + Others)"
        
        fig.update_layout(
            title=title,
            xaxis_title='Year',
            yaxis_title='Number of Grants',
            height=600,
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        return fig
    
    def _create_chart_title(self, filter_config: FilterConfig) -> str:
        """Create title with filter information"""
        title_parts = ["Grant Distribution by Year and Funder"]
        
        if filter_config.source_filter:
            title_parts.append(f"Sources: {', '.join(filter_config.source_filter)}")
        if filter_config.field_filter:
            field_title_parts = []
            for field in filter_config.field_filter[:2]:  # Show first 2 in title
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
            if len(filter_config.field_filter) > 2:
                field_display += f"... +{len(filter_config.field_filter) - 2} more"
            title_parts.append(f"Research Fields: {field_display}")
        
        # Add date range to title if filtered
        if filter_config.start_year_min is not None or filter_config.start_year_max is not None:
            if filter_config.start_year_min == filter_config.start_year_max:
                title_parts.append(f"Year: {filter_config.start_year_min}")
            else:
                title_parts.append(f"Years: {filter_config.start_year_min}-{filter_config.start_year_max}")
        
        return title_parts[0] if len(title_parts) == 1 else f"{title_parts[0]} | Filtered by {' | '.join(title_parts[1:])}"
    
    def _show_statistics(self, filtered_grants: pd.DataFrame, grants_by_year_funder: pd.DataFrame):
        """Display statistics about the grants"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Calculate total grants per year for peak year calculation
        grants_per_year = grants_by_year_funder.groupby('start_year')['count'].sum()
        
        with col1:
            st.metric("Total Grants", len(self.grants_df))
        with col2:
            st.metric("Filtered Grants", len(filtered_grants))
        with col3:
            st.metric("Unique Funders", grants_by_year_funder['funder'].nunique())
        with col4:
            st.metric("Year Range", f"{grants_per_year.index.min()}-{grants_per_year.index.max()}")
        with col5:
            peak_year = grants_per_year.idxmax()
            peak_count = grants_per_year.max()
            st.metric("Peak Year", f"{peak_year} ({peak_count} grants)")
    
    def _show_debug_data(self, filtered_grants: pd.DataFrame, grants_by_year_funder: pd.DataFrame):
        """Show debug data in an expander"""
        with st.expander("Debug: View Underlying Data", expanded=False):
            st.subheader("Filtered Grants DataFrame")
            st.write(f"**Shape:** {filtered_grants.shape}")
            st.dataframe(filtered_grants, use_container_width=True)
            
            st.subheader("Grants By Year and Funder (Chart Data)")
            st.write(f"**Shape:** {grants_by_year_funder.shape}")  
            st.dataframe(grants_by_year_funder, use_container_width=True)
            


class GrantKeywordViewer:
    """Manages the grants and keywords viewer"""
    
    def __init__(self, grants_df: pd.DataFrame, keywords_df: pd.DataFrame):
        self.grants_df = grants_df
        self.keywords_df = keywords_df
        # Create lookup table for fast keyword retrieval
        self.grant_keywords_lookup = self._build_grant_keywords_lookup()
    
    def render_tab(self):
        """Render the grants and keywords tab"""
        st.markdown("View grants and their extracted keywords side-by-side.")
        
        # Check if keywords data is available and has grant information
        if self.keywords_df is not None and 'grants' in self.keywords_df.columns:
            search_config = self._render_search_controls()
            self._display_grants_with_keywords(search_config)
        else:
            st.error("Keywords data not available or missing grant associations. Please ensure the keyword extraction process has been completed.")
    
    def _render_search_controls(self) -> SearchConfig:
        """Render search controls and return configuration"""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_term = st.text_input(
                "Search grants by title or keyword:", 
                placeholder="Enter search term...", 
                key="grants_search_term"
            )
        
        with col2:
            max_results = st.selectbox(
                "Max results:", 
                [10, 25, 50, 100], 
                index=1, 
                key="grants_max_results"
            )
        
        return SearchConfig(search_term, max_results)
    
    def _display_grants_with_keywords(self, search_config: SearchConfig):
        """Display grants with their keywords based on search"""
        # Filter grants based on search
        display_grants = self._filter_grants_by_search(search_config)
        
        if len(display_grants) > 0:
            st.write(f"Showing {len(display_grants)} grants:")
            
            # Display each grant with its keywords
            for _, grant in display_grants.iterrows():
                self._display_single_grant(grant)
        else:
            if search_config.search_term:
                st.info(f"No grants found matching '{search_config.search_term}'")
            else:
                st.info("No grants available to display.")
    
    def _filter_grants_by_search(self, search_config: SearchConfig) -> pd.DataFrame:
        """Filter grants based on search term"""
        if search_config.search_term:
            # Search in grant titles
            matching_grants = self.grants_df[
                self.grants_df['title'].str.contains(search_config.search_term, case=False, na=False)
            ]
            
            # Also search in keywords using vectorized operations
            matching_keywords = self.keywords_df[
                self.keywords_df['name'].str.contains(search_config.search_term, case=False, na=False)
            ]
            
            # Get grant IDs from matching keywords more efficiently
            keyword_grant_ids = set()
            if not matching_keywords.empty:
                # Use vectorized operations instead of iterating
                for grants_list in matching_keywords['grants'].dropna():
                    if isinstance(grants_list, list):
                        keyword_grant_ids.update(grants_list)
            
            # Combine grants from title search and keyword search
            title_grant_ids = set(matching_grants['id'].tolist())
            all_grant_ids = title_grant_ids.union(keyword_grant_ids)
            
            # Filter grants to display
            return self.grants_df[self.grants_df['id'].isin(all_grant_ids)].head(search_config.max_results)
        else:
            # Show recent grants if no search term
            return self.grants_df.head(search_config.max_results)
    
    def _display_single_grant(self, grant: pd.Series):
        """Display a single grant with its keywords"""
        with st.expander(f"**{grant['title']}** (ID: {grant['id']})"):
            # Grant information
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Grant Details:**")
                st.write(f"**Funder:** {grant.get('funder', 'N/A')}")
                st.write(f"**Start Year:** {grant.get('start_year', 'N/A')}")
                funding_amount = grant.get('funding_amount', 'N/A')
                if pd.notna(funding_amount):
                    st.write(f"**Funding Amount:** ${funding_amount:,}")
                else:
                    st.write("**Funding Amount:** N/A")
                
                if pd.notna(grant.get('grant_summary')):
                    st.write("**Summary:**")
                    st.write(grant['grant_summary'])
            
            with col2:
                st.write("**Extracted Keywords:**")
                
                # Find keywords for this grant
                grant_keywords = self._get_keywords_for_grant(grant['id'])
                
                if grant_keywords:
                    # Create annotated text for all keywords
                    annotated_elements = []
                    
                    for i, kw in enumerate(grant_keywords):
                        # Add the keyword with its type as annotation
                        type_color = self._get_type_color(kw['type'])
                        annotated_elements.append((kw['name'], kw['type'], type_color))
                        
                        # Add separator if not the last keyword
                        if i < len(grant_keywords) - 1:
                            annotated_elements.append("  ")
                    
                    # Display all keywords with their type annotations
                    annotated_text(*annotated_elements)
                else:
                    st.write("No keywords extracted for this grant.")
    
    def _build_grant_keywords_lookup(self) -> Dict[str, List[Dict[str, str]]]:
        """Build a lookup table mapping grant IDs to their keywords for fast retrieval"""
        if self.keywords_df is None:
            return {}
            
        lookup = {}
        for _, kw_row in self.keywords_df.iterrows():
            if isinstance(kw_row['grants'], list):
                keyword_data = {
                    'name': kw_row['name'],
                    'type': kw_row['type'],
                    'description': kw_row['description']
                }
                for grant_id in kw_row['grants']:
                    if grant_id not in lookup:
                        lookup[grant_id] = []
                    lookup[grant_id].append(keyword_data)
        return lookup
    
    def _get_keywords_for_grant(self, grant_id: str) -> List[Dict[str, str]]:
        """Get keywords for a specific grant using fast lookup"""
        return self.grant_keywords_lookup.get(grant_id, [])
    
    def _get_type_color(self, keyword_type: str) -> str:
        """Get background color for keyword type annotations"""
        type_colors = {
            'technology': '#1f77b4', 
            'methodology': "#bdbd13",  
            'general': "#14aa3c",    
            'application': "#d82121"
        }
        return type_colors.get(keyword_type.lower(), '#cccccc')


class GrantsPage:
    """Main page class that orchestrates all components"""
    
    def __init__(self):
        self.grants_df = None
        self.keywords_df = None
        self.categories_df = None
        self.setup_page()
    
    def setup_page(self):
        """Setup page configuration and load data"""
        setup_page_config("Grants")
        clear_previous_page_state()
        
        st.header("Grants Analysis")
        
        self._load_data()
        
        if self.grants_df is None:
            st.error("Unable to load grants data. Please check your data files.")
            return
    
    def _load_data(self):
        """Load and store data"""
        self.keywords_df, self.grants_df, self.categories_df = load_data()
    
    def run(self):
        """Main execution method"""
        # Create sidebar controls
        sidebar_controls = SidebarControls(self.grants_df)
        
        # Get configuration from sidebar (now returns tuple)
        filter_config, max_funders = sidebar_controls.render_sidebar()
        
        # Create tabs
        tab1, tab2 = st.tabs(["Grant Distributions", "Grants & Keywords"])
        
        # Render grant distribution tab
        with tab1:
            distribution_visualizer = GrantDistributionVisualizer(self.grants_df)
            distribution_visualizer.render_tab(filter_config, max_funders)
        
        # Render grants and keywords tab
        with tab2:
            keyword_viewer = GrantKeywordViewer(self.grants_df, self.keywords_df)
            keyword_viewer.render_tab()


def main():
    """Main function to run the grants page"""
    grants_page = GrantsPage()
    grants_page.run()


if __name__ == "__main__":
    main()
