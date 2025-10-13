"""
Grants Analysis Page
Analyze grant distributions over time and view grants with their extracted keywords.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

# Add the web directory to Python path to import shared_utils
web_dir = str(Path(__file__).parent.parent)
if web_dir not in sys.path:
    sys.path.insert(0, web_dir)

from shared_utils import (  # noqa: E402
    format_research_field,
    load_data
)
from visualisation import (  # noqa: E402
    DataExplorer,
    DataExplorerConfig,
    TrendsVisualizer,
    TrendsConfig,
    compute_active_years,
)
from web.sidebar import SidebarControl, FilterConfig, DisplayConfig  # noqa: E402

st.set_page_config(
    page_title="Grant",
    layout="wide",
    initial_sidebar_state="expanded"
)
class GrantFilterManager:
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
            if config.field_filter and 'primary_subject' in filtered_grants.columns:
                filtered_grants = filtered_grants[filtered_grants['primary_subject'].isin(config.field_filter)]

        if config.use_active_grant_period and 'start_year' in filtered_grants.columns:
            start_series = filtered_grants['start_year']
            end_series = filtered_grants['end_year'] if 'end_year' in filtered_grants.columns else start_series
            start_series = start_series.fillna(end_series)
            end_series = end_series.fillna(start_series)
            mask = pd.Series(True, index=filtered_grants.index)
            if config.start_year_min is not None:
                mask &= end_series >= config.start_year_min
            if config.start_year_max is not None:
                mask &= start_series <= config.start_year_max
            filtered_grants = filtered_grants[mask]
        else:
            if config.start_year_min is not None and 'start_year' in filtered_grants.columns:
                filtered_grants = filtered_grants[filtered_grants['start_year'] >= config.start_year_min]
            if config.start_year_max is not None and 'start_year' in filtered_grants.columns:
                filtered_grants = filtered_grants[filtered_grants['start_year'] <= config.start_year_max]
        
        return filtered_grants


class GrantDistributionVisualizer:
    """Manages the grant distribution visualization"""
    
    def __init__(self, grants_df: pd.DataFrame):
        self.grants_df = grants_df
        self.filter_manager = GrantFilterManager(grants_df)
    
    def render_tab(self, filter_config: FilterConfig, display_config: DisplayConfig):
        """Render the grant distribution visualization tab"""
        
        should_generate = self._should_generate_visualization()
        
        if should_generate:
            self._generate_visualization(filter_config, display_config)
        else:
            st.info("Adjust settings in the sidebar to customize the visualization.")
    
    def _should_generate_visualization(self) -> bool:
        """Determine if visualization should be generated"""
        if "distributions_initialized" not in st.session_state:
            st.session_state.distributions_initialized = True
            return True
        return True  # Always generate when filters change
    
    def _generate_visualization(self, filter_config: FilterConfig, display_config: DisplayConfig):
        """Generate and display the visualization"""
        with st.spinner("Creating grant distribution visualization..."):
            # Apply filters to grants
            filtered_grants = self.filter_manager.apply_filters(filter_config)
            
            if len(filtered_grants) == 0:
                st.error("No grants found with the current filters. Please adjust your filter settings.")
                return
            
            if filter_config.use_active_grant_period:
                expanded_records = []
                for _, row in filtered_grants.iterrows():
                    years = compute_active_years(
                        row.get('start_year'),
                        row.get('end_year'),
                        use_active_period=True,
                        min_year=filter_config.start_year_min,
                        max_year=filter_config.start_year_max,
                    )
                    if not years or pd.isna(row.get('funder')):
                        continue
                    expanded_records.extend({'start_year': year, 'funder': row['funder']} for year in years)
                if not expanded_records:
                    st.warning("No active grants within the selected period.")
                    return
                expanded_df = pd.DataFrame(expanded_records)
                grants_by_year_funder = expanded_df.groupby(['start_year', 'funder']).size().reset_index(name='count')
            else:
                # Group grants by year and funder
                grants_by_year_funder = filtered_grants.groupby(['start_year', 'funder']).size().reset_index(name='count')
            
            # Create stacked area chart using generalized visualizer
            fig = self._create_grants_visualization(grants_by_year_funder, filter_config, display_config)
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Show statistics
            self._show_statistics(filtered_grants, grants_by_year_funder)
            
            # Debug expander showing underlying data
            self._show_debug_data(filtered_grants, grants_by_year_funder)
    
    def _create_grants_visualization(self, grants_by_year_funder: pd.DataFrame, filter_config: FilterConfig, display_config: DisplayConfig) -> go.Figure:
        """Create the grants visualization using the new TrendsVisualizer"""
        
        # Prepare data for the visualizer
        # Rename columns to match expected format
        viz_data = grants_by_year_funder.rename(columns={
            'start_year': 'year',
            'funder': 'funder',
            'count': 'grant_count'
        })
        
        # Create title with filter information
        title = self._create_chart_title(filter_config, display_config)
        
        # Configure the new visualizer for stacked area chart
        viz_config = TrendsConfig(
            entity_col='funder',
            time_col='year',
            value_col='grant_count',
            max_entities=display_config.num_entities,
            aggregation='sum',
            use_cumulative=display_config.use_cumulative,
            chart_type='stacked_area',
            show_others=True,
            title=title,
            x_label='Year',
            y_label='Number of Grants',
            height=600
        )
        
        # Create the visualization
        visualizer = TrendsVisualizer()
        return visualizer.create_plot(viz_data, viz_config)
    
    def _create_chart_title(self, filter_config: FilterConfig, display_config: DisplayConfig) -> str:
        """Create title with filter information"""
        count_type = "Cumulative" if display_config.use_cumulative else "Yearly"
        title_parts = [f"{count_type} Grant Distribution by Year and Funder"]
        
        if filter_config.source_filter:
            title_parts.append(f"Sources: {', '.join(filter_config.source_filter)}")
        if filter_config.field_filter:
                display_names = [format_research_field(field) for field in filter_config.field_filter[:2]]
                field_display = ', '.join(display_names)
                if len(filter_config.field_filter) > 2:
                    field_display += f"... +{len(filter_config.field_filter) - 2} more"
                title_parts.append(f"Subjects: {field_display}")
        
        # Add date range to title if filtered
        if filter_config.start_year_min is not None or filter_config.start_year_max is not None:
            if filter_config.start_year_min == filter_config.start_year_max:
                title_parts.append(f"Year: {filter_config.start_year_min}")
            else:
                title_parts.append(f"Years: {filter_config.start_year_min}-{filter_config.start_year_max}")
        if filter_config.use_active_grant_period:
            title_parts.append("Active grant period")
        
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
            


class GrantsPage:
    """Main page class that orchestrates all components"""
    
    def __init__(self):
        self.grants_df = None
        self.keywords_df = None
        self.categories_df = None
        self.setup_page()
    
    def setup_page(self):

        
        st.header("Grants Analysis")
        
        self._load_data()
        
        if self.grants_df is None:
            st.error("Unable to load grants data. Please check your data files.")
            return
    
    def _load_data(self):
        """Load and store data"""
        self.keywords_df, self.grants_df, self.categories_df = load_data()
    
    @staticmethod
    def prepare_grants_data(grants_df: pd.DataFrame) -> tuple[pd.DataFrame, DataExplorerConfig]:
        """Prepare grants data for DataExplorer"""
        if grants_df is None or grants_df.empty:
            return pd.DataFrame(), DataExplorerConfig(
                title="Grants Dataset",
                description="No grants data available",
                search_columns=[],
                display_columns=[]
            )
        
        display_df = grants_df.copy()
        display_df = display_df.sort_values(['funding_amount', 'start_year'], ascending=[False, False])
        
        config = DataExplorerConfig(
            title="Grants Dataset",
            description="Explore grants data. Search by grant title, funder, or source.",
            search_columns=['title', 'funder', 'source'],
            search_placeholder="Search by grant title, funder, or source...",
            search_help="Enter text to search across grant titles, funders, and sources",
            display_columns=['title', 'funder', 'source', 'start_year', 'funding_amount'],
            column_formatters={
                'funding_amount': lambda x: f"${x:,.0f}" if pd.notna(x) and x > 0 else "N/A",
                'title': lambda x: x[:80] + "..." if len(str(x)) > 80 else str(x)
            },
            column_renames={
                'title': 'Grant Title',
                'funder': 'Funder',
                'source': 'Source',
                'start_year': 'Start Year',
                'funding_amount': 'Funding Amount'
            },
            statistics_columns=['funding_amount', 'start_year'],
            max_display_rows=50,
            show_statistics=True,
            show_data_info=True,
            enable_download=True
        )
        
        return display_df, config
    
    def run(self):
        """Main execution method"""
        # Create unified sidebar
        sidebar_controls = SidebarControl(
            page_type="grants",
            grants_df=self.grants_df
        )
        filter_config, display_config = sidebar_controls.render_sidebar()
        
        # Apply filters once for use across both tabs
        filter_manager = GrantFilterManager(self.grants_df)
        filtered_grants = filter_manager.apply_filters(filter_config)
        
        tab1, tab2 = st.tabs(["Grant Distributions", "Grants Explorer"])
        
        with tab1:
            distribution_visualizer = GrantDistributionVisualizer(self.grants_df)
            distribution_visualizer.render_tab(filter_config, display_config)
        
        with tab2:
            display_df, config = self.prepare_grants_data(filtered_grants)
            data_explorer = DataExplorer()
            data_explorer.render_explorer(display_df, config)


def main():
    """Main function to run the grants page"""
    grants_page = GrantsPage()
    grants_page.run()


if __name__ == "__main__":
    main()
