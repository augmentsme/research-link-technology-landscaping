"""
Keyword Trends Analysis Page
Analyze how research keywords evolve over time with cumulative occurrence tracking.
"""

import streamlit as st
import pandas as pd
import random
import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

# Add the web directory to Python path to import shared_utils
web_dir = str(Path(__file__).parent.parent)
if web_dir not in sys.path:
    sys.path.insert(0, web_dir)

from shared_utils import (
    format_research_field,
    load_data
)
from web.sidebar import SidebarControl, FilterConfig, DisplayConfig
from visualisation import (
    DataExplorer,
    DataExplorerConfig,
    TrendsVisualizer,
    TrendsConfig,
    TrendsDataPreparation,
)


st.set_page_config(
    page_title="Keywords",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class KeywordFilterConfig:
    """Configuration for data filtering"""
    funder_filter: List[str]
    source_filter: List[str]
    field_filter: List[str]
    keyword_type_filter: List[str]
    min_count: int
    max_count: int


@dataclass
class KeywordSelectionConfig:
    """Configuration for keyword selection"""
    method: str  # "Top N keywords", "Random sample", "Specify custom keywords"
    top_n: int
    random_sample_size: int
    custom_keywords: List[str]
    use_random_seed: bool
    random_seed: Optional[int]


@dataclass
class KeywordDisplayConfig:
    """Configuration for display settings"""
    show_baseline: bool
    use_cumulative: bool


class KeywordFilterManager:
    """Manages all data filtering operations"""
    
    def __init__(self, keywords_df: pd.DataFrame, grants_df: pd.DataFrame):
        self.keywords_df = keywords_df
        self.grants_df = grants_df
    
    def apply_grant_filters(self, config: KeywordFilterConfig) -> pd.DataFrame:
        """Apply filters to grants dataframe"""
        filtered_grants = self.grants_df.copy()
        
        if config.funder_filter:
            filtered_grants = filtered_grants[filtered_grants['funder'].isin(config.funder_filter)]
        
        if config.source_filter:
            filtered_grants = filtered_grants[filtered_grants['source'].isin(config.source_filter)]
        
        if config.field_filter and 'primary_subject' in filtered_grants.columns:
            filtered_grants = filtered_grants[filtered_grants['primary_subject'].isin(config.field_filter)]
        
        return filtered_grants
    
    def apply_keyword_filters(self, config: KeywordFilterConfig, filtered_grants: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Apply filters to keywords dataframe and update grants to reflect filtered results"""
        filtered_keywords = self.keywords_df.copy()
        
        if config.keyword_type_filter:
            filtered_keywords = filtered_keywords[filtered_keywords['type'].isin(config.keyword_type_filter)]
        
        # Apply grant-based filtering if any grant filters are active
        if any([config.funder_filter, config.source_filter, config.field_filter]):
            if filtered_grants is None:
                filtered_grants = self.apply_grant_filters(config)
            
            # Get set of filtered grant IDs (preserve original type)
            filtered_grants_ids = set(filtered_grants['id'])
            
            # Filter keywords and update their grants lists to only include filtered grants
            def filter_keyword_grants(grants_list):
                # Filter to only grants that are in the filtered set
                filtered_grants_for_keyword = [g for g in grants_list if g in filtered_grants_ids]
                return filtered_grants_for_keyword
            
            # Update grants column to only include grants that match the filter
            filtered_keywords['grants'] = filtered_keywords['grants'].apply(filter_keyword_grants)
            
            # Filter keywords based on count requirements using the filtered grants
            filtered_keywords = filtered_keywords[filtered_keywords['grants'].apply(
                lambda x: config.min_count <= len(x) <= config.max_count
            )]
        else:
            grant_counts = filtered_keywords.grants.map(len)
            filtered_keywords = filtered_keywords[(grant_counts >= config.min_count) & (grant_counts <= config.max_count)]
        
        return filtered_keywords
    
    def get_available_keywords(self, config: KeywordFilterConfig) -> List[str]:
        """Get list of available keyword names after filtering, sorted by number of grants (descending)"""
        filtered_keywords = self.apply_keyword_filters(config)
        
        # Sort by number of grants (descending order)
        filtered_keywords = filtered_keywords.copy()
        filtered_keywords['grant_count'] = filtered_keywords['grants'].apply(len)
        filtered_keywords = filtered_keywords.sort_values('grant_count', ascending=False)
        
        if 'name' in filtered_keywords.columns:
            return [str(name) for name in filtered_keywords['name'].tolist()]
        else:
            return [str(name) for name in filtered_keywords.index.tolist()]


class KeywordSelector:
    """Handles different keyword selection strategies"""
    
    def __init__(self, filter_manager: KeywordFilterManager):
        self.filter_manager = filter_manager

    
    def select_keywords(self, filter_config: KeywordFilterConfig, selection_config: KeywordSelectionConfig, display_config: KeywordDisplayConfig) -> Tuple[List[str], str]:
        """Select keywords based on the configured method"""
        available_keywords = self.filter_manager.get_available_keywords(filter_config)
        
        if selection_config.method == "Top N keywords":
            return [], self._create_top_n_title(filter_config, selection_config, display_config)
        
        elif selection_config.method == "Random sample":
            return self._select_random_keywords(available_keywords, selection_config, display_config)
        
        elif selection_config.method == "Specify custom keywords":
            count_type = "Cumulative" if display_config.use_cumulative else "Yearly"
            title = f"{count_type} Keyword Trends - Custom Selection ({len(selection_config.custom_keywords)} keywords)"
            return selection_config.custom_keywords, title
        
        return [], "Unknown Selection Method"
    
    def _select_random_keywords(self, available_keywords: List[str], selection_config: KeywordSelectionConfig, display_config: KeywordDisplayConfig) -> Tuple[List[str], str]:
        """Select random keywords from available list"""
        if len(available_keywords) == 0 or selection_config.random_sample_size == 0:
            return [], "No keywords available for random sampling"
        
        if selection_config.use_random_seed and selection_config.random_seed is not None:
            effective_seed = selection_config.random_seed + st.session_state.get('random_sample_counter', 0)
            random.seed(effective_seed)
            np.random.seed(effective_seed)
        
        sampled_keywords = random.sample(
            available_keywords, 
            min(selection_config.random_sample_size, len(available_keywords))
        )
        
        count_type = "Cumulative" if display_config.use_cumulative else "Yearly"
        title = f"{count_type} Random Sample of Keywords ({len(sampled_keywords)} keywords)"
        
        return sampled_keywords, title
    
    
    def _create_top_n_title(self, filter_config: KeywordFilterConfig, selection_config: KeywordSelectionConfig, display_config: KeywordDisplayConfig) -> str:
        """Create title for top N keyword selection"""
        count_type = "Cumulative" if display_config.use_cumulative else "Yearly"
        return f"{count_type} Keyword Trends (count_range={filter_config.min_count}-{filter_config.max_count}, top_n={selection_config.top_n})"


class KeywordTrendsVisualizer:
    """Manages the trends visualization tab using unified TrendsVisualizer"""
    
    def __init__(self, keywords_df: pd.DataFrame, grants_df: pd.DataFrame):
        self.keywords_df = keywords_df
        self.grants_df = grants_df
        self.filter_manager = KeywordFilterManager(keywords_df, grants_df)
        self.keyword_selector = KeywordSelector(self.filter_manager)
    
    def render_tab(self, filter_config: KeywordFilterConfig, selection_config: KeywordSelectionConfig, 
                  display_config: KeywordDisplayConfig):
        """Render the trends visualization tab"""
        st.markdown("# Keyword Trends Over Time")
        
        if selection_config.method == "Specify custom keywords" and not selection_config.custom_keywords:
            st.error("Please select at least one keyword when using custom keyword selection.")
            return
        
        with st.spinner("Creating keyword trends visualization..."):
            # Apply filters to get filtered grants and keywords
            filtered_grants = self.filter_manager.apply_grant_filters(filter_config)
            filtered_keywords = self.filter_manager.apply_keyword_filters(filter_config, filtered_grants)
            
            selected_keywords, title = self.keyword_selector.select_keywords(filter_config, selection_config, display_config)
            title_with_filters = self._add_filter_info_to_title(title, filter_config)
            
            # Use filtered data for trends visualization
            viz_data = TrendsDataPreparation.from_keyword_grants(filtered_keywords, filtered_grants, selected_keywords)
            
            if viz_data.empty:
                st.warning("No data available for the selected keywords and filters.")
                return
            
            num_entities = (
                selection_config.top_n if selection_config.method == "Top N keywords" else
                len(selected_keywords)
            )
            
            viz_config = TrendsConfig(
                entity_col='keyword',
                time_col='year',
                value_col='grant_count',
                max_entities=num_entities,
                aggregation='sum',
                use_cumulative=display_config.use_cumulative,
                chart_type='line',
                title=title_with_filters,
                x_label='Year',
                y_label='Cumulative Grant Count' if display_config.use_cumulative else 'Yearly Grant Count',
                height=600
            )
            
            visualizer = TrendsVisualizer()
            fig_trends = visualizer.create_plot(viz_data, viz_config)
            
            if fig_trends is not None:
                st.plotly_chart(fig_trends, use_container_width=True)
                self._show_statistics(filter_config, selection_config, selected_keywords)
                self._show_debug_data(filtered_keywords, viz_data)
            else:
                st.warning("No data available for the selected parameters.")
    
    def _add_filter_info_to_title(self, title: str, filter_config: KeywordFilterConfig) -> str:
        """Add filter information to the visualization title"""
        filter_parts = []
        if filter_config.funder_filter:
            filter_parts.append(f"Funders: {', '.join(filter_config.funder_filter)}")
        if filter_config.source_filter:
            filter_parts.append(f"Sources: {', '.join(filter_config.source_filter)}")
        if filter_config.field_filter:
            field_display_parts = [format_research_field(field)[:25] for field in filter_config.field_filter[:2]]
            field_display = ', '.join(field_display_parts)
            if len(filter_config.field_filter) > 2:
                field_display += f"... +{len(filter_config.field_filter) - 2} more"
            filter_parts.append(f"Research Fields: {field_display}")
        if filter_config.keyword_type_filter:
            filter_parts.append(f"Types: {', '.join(filter_config.keyword_type_filter)}")
        
        return f"{title} | Filtered by {' | '.join(filter_parts)}" if filter_parts else title
    
    def _show_statistics(self, filter_config: KeywordFilterConfig, selection_config: KeywordSelectionConfig, 
                        selected_keywords: List[str]):
        """Display statistics about the filtered data"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        filtered_grants = self.filter_manager.apply_grant_filters(filter_config)
        filtered_keywords = self.filter_manager.apply_keyword_filters(filter_config, filtered_grants)
        
        with col1:
            st.metric("Total Keywords", len(self.keywords_df))
        with col2:
            st.metric("Filtered Keywords", len(filtered_keywords))
        with col3:
            st.metric("Total Grants", len(self.grants_df))
        with col4:
            st.metric("Filtered Grants", len(filtered_grants))
        with col5:
            displayed_count = self._get_displayed_keywords_count(selection_config, selected_keywords, filtered_keywords)
            st.metric("Displayed Keywords", displayed_count)
    
    def _get_displayed_keywords_count(self, selection_config: KeywordSelectionConfig, 
                                    selected_keywords: List[str], filtered_keywords: pd.DataFrame) -> int:
        """Get the count of displayed keywords"""
        if selection_config.method == "Top N keywords" and selection_config.top_n > 0:
            return min(selection_config.top_n, len(filtered_keywords))
        elif selection_config.method == "Random sample":
            return len(selected_keywords)
        elif selection_config.method == "Specify custom keywords":
            return len(selection_config.custom_keywords)
        return 0
    
    def _show_debug_data(self, filtered_keywords: pd.DataFrame, viz_data: pd.DataFrame):
        """Show debug data in an expander"""
        with st.expander("Debug: View Underlying Data", expanded=False):
            st.subheader("Filtered Keywords DataFrame")
            st.write(f"**Shape:** {filtered_keywords.shape}")
            st.dataframe(filtered_keywords, use_container_width=True)
            
            st.subheader("Keyword Trends Data (Chart Data)")
            st.write(f"**Shape:** {viz_data.shape}")
            st.dataframe(viz_data, use_container_width=True)
    

class KeywordDataExplorer:
    """Manages the data exploration tab using the generalized DataExplorer"""
    
    def __init__(self, keywords_df: pd.DataFrame, grants_df: pd.DataFrame):
        self.keywords_df = keywords_df
        self.grants_df = grants_df
        self.filter_manager = KeywordFilterManager(keywords_df, grants_df)
        self.data_explorer = DataExplorer()
    
    @staticmethod
    def prepare_data(filtered_keywords: pd.DataFrame) -> tuple[pd.DataFrame, DataExplorerConfig]:
        """Prepare keywords data and configuration for DataExplorer"""
        display_df = filtered_keywords.copy()
        
        if 'name' not in display_df.columns and display_df.index.name == 'name':
            display_df = display_df.reset_index()
        
        display_df['grant_count'] = display_df['grants'].apply(len)
        display_df = display_df.sort_values('grant_count', ascending=False)
        
        config = DataExplorerConfig(
            title="Keywords Dataset",
            description="Explore the complete keywords dataset with filtering, search capabilities, and distribution analysis.",
            search_columns=['name', 'type'],
            search_placeholder="Search in keyword names or types...",
            search_help="Enter text to search for specific keywords by name or type",
            display_columns=['name', 'type', 'grant_count', 'grants'],
            column_formatters={
                'grants': lambda x: ', '.join(str(g) for g in x[:3]) + ('...' if len(x) > 3 else '')
            },
            column_renames={
                'name': 'Keyword Name',
                'type': 'Type',
                'grant_count': 'Grant Count',
                'grants': 'Associated Grants'
            },
            statistics_columns=['grant_count'],
            max_display_rows=100,
            show_statistics=True,
            show_data_info=True,
            enable_download=True
        )
        
        return display_df, config
    
    def render_tab(self, filter_config: KeywordFilterConfig):
        """Render the data exploration tab"""
        filtered_keywords = self.filter_manager.apply_keyword_filters(filter_config)
        
        if len(filtered_keywords) == 0:
            st.warning("No keywords found with current filters. Please adjust your filter settings.")
            return
        
        # Prepare data and config
        display_data, config = self.prepare_data(filtered_keywords)
        
        # Render the explorer
        self.data_explorer.render_explorer(display_data, config)


class KeywordsPage:
    """Main page class that orchestrates all components"""
    
    def __init__(self):
        self.keywords_df = None
        self.grants_df = None
        self.categories_df = None
        self.setup_page()
    
    def setup_page(self):

        
        st.header("Keywords Analysis")
        
        self._load_data()
        
        if self.keywords_df is None or self.grants_df is None:
            st.error("Unable to load data. Please check your data files.")
            st.stop()
        
        # Store data in session state for use in sidebar controls
        st.session_state['keywords'] = self.keywords_df
        st.session_state['grants'] = self.grants_df
    
    def _load_data(self):
        """Load and store data"""
        self.keywords_df, self.grants_df, self.categories_df = load_data()
    
    def _map_selection_method(self, method: str) -> str:
        """Map unified selection method to legacy format"""
        mapping = {
            "top_n": "Top N keywords",
            "random": "Random sample",
            "custom": "Specify custom keywords"
        }
        return mapping.get(method, "Top N keywords")
    
    def run(self):
        """Main execution method"""
        # Create unified sidebar controls
        sidebar = SidebarControl(
            page_type="keywords",
            grants_df=self.grants_df,
            keywords_df=self.keywords_df,
            categories_df=self.categories_df
        )
        
        # Get unified configurations from sidebar
        unified_filter, unified_display = sidebar.render_sidebar()
        
        # Adapt to legacy format for existing code
        filter_config = KeywordFilterConfig(
            funder_filter=unified_filter.funder_filter,
            source_filter=unified_filter.source_filter,
            field_filter=unified_filter.field_filter,
            keyword_type_filter=unified_filter.keyword_type_filter,
            min_count=unified_filter.min_count,
            max_count=unified_filter.max_count
        )
        
        selection_config = KeywordSelectionConfig(
            method=self._map_selection_method(unified_display.selection_method),
            top_n=unified_display.num_entities,
            random_sample_size=unified_display.num_entities,
            custom_keywords=unified_display.custom_entities,
            use_random_seed=unified_display.use_random_seed,
            random_seed=unified_display.random_seed
        )
        
        display_config = KeywordDisplayConfig(
            show_baseline=unified_display.show_baseline,
            use_cumulative=unified_display.use_cumulative
        )
        
        # Create tabs
        tab1, tab2 = st.tabs(["Trends Visualization", "Keywords Data"])
        
        # Render trends visualization tab
        with tab1:
            trends_viz = KeywordTrendsVisualizer(self.keywords_df, self.grants_df)
            trends_viz.render_tab(filter_config, selection_config, display_config)
        
        # Render data exploration tab
        with tab2:
            data_exploration = KeywordDataExplorer(self.keywords_df, self.grants_df)
            data_exploration.render_tab(filter_config)


def main():
    """Main function to run the keywords page"""
    keywords_page = KeywordsPage()
    keywords_page.run()


if __name__ == "__main__":
    main()