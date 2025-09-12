"""
Keyword Trends Analysis Page
Analyze how research keywords evolve over time with cumulative occurrence tracking.
"""

import streamlit as st
import pandas as pd
import random
import numpy as np
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

# Add the web directory to Python path to import shared_utils
web_dir = str(Path(__file__).parent.parent)
if web_dir not in sys.path:
    sys.path.insert(0, web_dir)

from shared_utils import (
    setup_page_config, load_data,
    get_unique_values_from_data, create_research_field_options,
    field_to_division_codes, has_research_field_simple, clear_previous_page_state
)
from visualisation import KeywordTrendsVisualizer, FilterConfig as VisFilterConfig, VisualizationConfig


@dataclass
class FilterConfig:
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
class DisplayConfig:
    """Configuration for display settings"""
    show_baseline: bool
    use_cumulative: bool


class FilterManager:
    """Manages all data filtering operations"""
    
    def __init__(self, keywords_df: pd.DataFrame, grants_df: pd.DataFrame):
        self.keywords_df = keywords_df
        self.grants_df = grants_df
    
    def apply_grant_filters(self, config: FilterConfig) -> pd.DataFrame:
        """Apply filters to grants dataframe"""
        filtered_grants = self.grants_df.copy()
        
        if config.funder_filter:
            filtered_grants = filtered_grants[filtered_grants['funder'].isin(config.funder_filter)]
        
        if config.source_filter:
            filtered_grants = filtered_grants[filtered_grants['source'].isin(config.source_filter)]
        
        if config.field_filter:
            division_codes = field_to_division_codes(config.field_filter)
            mask = filtered_grants['for_primary'].apply(
                lambda x: has_research_field_simple(x, division_codes)
            )
            filtered_grants = filtered_grants[mask]
        
        return filtered_grants
    
    def apply_keyword_filters(self, config: FilterConfig, filtered_grants: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Apply filters to keywords dataframe and update grants to reflect filtered results"""
        filtered_keywords = self.keywords_df.copy()
        
        if config.keyword_type_filter:
            filtered_keywords = filtered_keywords[filtered_keywords['type'].isin(config.keyword_type_filter)]
        
        # Apply grant-based filtering if any grant filters are active
        if any([config.funder_filter, config.source_filter, config.field_filter]):
            if filtered_grants is None:
                filtered_grants = self.apply_grant_filters(config)
            
            # Convert all grant IDs to strings to ensure consistent data types
            filtered_grants_ids = set(str(gid) for gid in filtered_grants['id'])
            
            # Filter keywords and update their grants lists to only include filtered grants
            def filter_keyword_grants(grants_list):
                # Ensure grants_list items are strings and filter them
                filtered_grants_for_keyword = [str(g) for g in grants_list if str(g) in filtered_grants_ids]
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
    
    def get_available_keywords(self, config: FilterConfig) -> List[str]:
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
    
    def __init__(self, filter_manager: FilterManager):
        self.filter_manager = filter_manager
    
    def select_keywords(self, filter_config: FilterConfig, selection_config: KeywordSelectionConfig, display_config: DisplayConfig) -> Tuple[List[str], str]:
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
    
    def _select_random_keywords(self, available_keywords: List[str], selection_config: KeywordSelectionConfig, display_config: DisplayConfig) -> Tuple[List[str], str]:
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
    
    def _create_top_n_title(self, filter_config: FilterConfig, selection_config: KeywordSelectionConfig, display_config: DisplayConfig) -> str:
        """Create title for top N keyword selection"""
        count_type = "Cumulative" if display_config.use_cumulative else "Yearly"
        return f"{count_type} Keyword Trends (count_range={filter_config.min_count}-{filter_config.max_count}, top_n={selection_config.top_n})"


class SidebarControls:
    """Handles all sidebar UI controls"""
    
    def __init__(self, keywords_df: pd.DataFrame, grants_df: pd.DataFrame, categories_df: pd.DataFrame):
        self.keywords_df = keywords_df
        self.grants_df = grants_df
        self.categories_df = categories_df
        
        # Get unique values using the shared utility function
        self.unique_funders, self.unique_sources, self.unique_keyword_types, unique_research_fields = get_unique_values_from_data(
            keywords_df, grants_df, categories_df
        )
        
        # Get research field options
        self.field_options, self.field_values = create_research_field_options(unique_research_fields)
        
        # Create mapping from options to field values
        self.option_to_field = dict(zip(self.field_options, self.field_values))
    
    def _get_keyword_count_range_from_percentiles(self, filtered_keywords: pd.DataFrame, lower_percentile: float, upper_percentile: float) -> Tuple[int, int]:
        """Convert percentile range to actual keyword count range based on keyword ranking"""
        if filtered_keywords.empty:
            return 1, 50
        
        # Calculate grant counts for all keywords and sort by occurrence
        grant_counts = filtered_keywords['grants'].apply(len)
        sorted_counts = grant_counts.sort_values(ascending=True)  # Sort ascending (least to most frequent)
        
        # Get the total number of keywords
        total_keywords = len(sorted_counts)
        
        # Calculate the indices based on percentiles (rank-based)
        lower_index = int(lower_percentile * total_keywords)
        upper_index = int(upper_percentile * total_keywords)
        
        # Ensure indices are within bounds
        lower_index = max(0, min(lower_index, total_keywords - 1))
        upper_index = max(lower_index, min(upper_index, total_keywords - 1))
        
        # Get the actual count thresholds at these positions
        min_count = max(1, sorted_counts.iloc[lower_index])
        max_count = sorted_counts.iloc[upper_index]
        
        # Ensure max_count is at least min_count
        if max_count < min_count:
            max_count = min_count
        
        return min_count, max_count
    
    def _get_keyword_count_range_from_log_scale(self, filtered_keywords: pd.DataFrame, lower_log: float, upper_log: float) -> Tuple[int, int]:
        """Convert log scale values to actual keyword count range"""
        if filtered_keywords.empty:
            return 1, 50
        
        # Calculate grant counts for all keywords
        grant_counts = filtered_keywords['grants'].apply(len)
        min_possible = max(1, grant_counts.min())
        max_possible = grant_counts.max()
        
        # Handle edge case where all keywords have the same count
        if min_possible == max_possible:
            return min_possible, max_possible
        
        # Convert from log scale (0-1) to actual counts
        log_min = np.log(min_possible)
        log_max = np.log(max_possible)
        
        # Map slider values (0-1) to log space
        lower_log_val = log_min + lower_log * (log_max - log_min)
        upper_log_val = log_min + upper_log * (log_max - log_min)
        
        # Convert back to actual counts
        min_count = max(1, int(np.exp(lower_log_val)))
        max_count = max(min_count, int(np.exp(upper_log_val)))
        
        return min_count, max_count
    
    def _format_log_scale_value(self, x: float, filtered_keywords: pd.DataFrame) -> str:
        """Format a single log scale value to show the corresponding occurrence count"""
        if filtered_keywords.empty:
            return f"{x:.0%}"
        
        grant_counts = filtered_keywords['grants'].apply(len)
        min_possible = max(1, grant_counts.min())
        max_possible = grant_counts.max()
        
        if min_possible == max_possible:
            return str(min_possible)
        
        log_min = np.log(min_possible)
        log_max = np.log(max_possible)
        log_val = log_min + x * (log_max - log_min)
        count = max(1, int(np.exp(log_val)))
        
        return str(count)
    
    def _format_percentile_range(self, lower: float, upper: float, filtered_keywords: pd.DataFrame) -> str:
        """Format the percentile range display based on keyword ranking"""
        if filtered_keywords.empty:
            return f"{lower:.0%} - {upper:.0%}"
        
        min_count, max_count = self._get_keyword_count_range_from_percentiles(filtered_keywords, lower, upper)
        
        # Calculate how many keywords fall in this range
        grant_counts = filtered_keywords['grants'].apply(len)
        included_keywords = ((grant_counts >= min_count) & (grant_counts <= max_count)).sum()
        total_keywords = len(grant_counts)
        
        # Calculate the expected number of keywords based on percentile range
        expected_keywords = int((upper - lower) * total_keywords)
        
        return f"{lower:.0%} - {upper:.0%} (rank-based: {min_count}-{max_count} grants, ~{expected_keywords} keywords)"
    
    def render_sidebar(self) -> Tuple[FilterConfig, KeywordSelectionConfig, DisplayConfig]:
        """Render all sidebar controls and return configuration objects"""
        st.sidebar.empty()
        
        with st.sidebar:
            st.subheader("📈 Keyword Trends Settings")
            
            filter_config = self._render_filtering_options()
            selection_config = self._render_keyword_selection(filter_config)
            display_config = self._render_display_settings()
            
            st.markdown("---")
            
            return filter_config, selection_config, display_config
    
    def _render_filtering_options(self) -> FilterConfig:
        """Render filtering options section"""
        with st.expander("🔍 Filtering Options", expanded=False):
            st.markdown("**Grant Filters**")
            
            funder_filter = st.multiselect(
                "Filter by Funder",
                options=self.unique_funders,
                default=[],
                help="Select specific funders to filter grants by (leave empty for all funders)"
            )
            
            source_filter = st.multiselect(
                "Filter by Source",
                options=self.unique_sources,
                default=[],
                help="Select specific sources to filter grants by (leave empty for all sources)"
            )
            
            st.markdown("**Research Fields:**")
            selected_field_options = st.multiselect(
                "Select research fields to filter by",
                options=self.field_options,
                default=[],
                help="Select specific research areas to filter by. Leave empty to include all research fields."
            )
            field_filter = [self.option_to_field[option] for option in selected_field_options]
            
            st.markdown("**Keyword Filters**")
            keyword_type_filter = st.multiselect(
                "Filter by Keyword Type",
                options=self.unique_keyword_types,
                default=[],
                help="Select specific keyword types to filter by (leave empty for all types)"
            )
            
            self._show_active_filters(funder_filter, source_filter, field_filter, keyword_type_filter)
            
            # Create a temporary config to get filtered keywords for calculating count options
            temp_config = FilterConfig(funder_filter, source_filter, field_filter, keyword_type_filter, 1, 999999)
            filter_manager = FilterManager(self.keywords_df, self.grants_df)
            temp_filtered_keywords = filter_manager.apply_keyword_filters(temp_config)
            
            # Use select_slider with log scale range (0-1)
            log_range = st.select_slider(
                "Keyword occurrence range (log scale)", 
                options=[i/20 for i in range(21)],  # 0.0, 0.05, 0.10, ..., 1.0
                value=(0.0, 1.0),  # Default: include all keywords
                format_func=lambda x: self._format_log_scale_value(x, temp_filtered_keywords) if not temp_filtered_keywords.empty else f"{x:.0%}",
                help="Select keywords by occurrence count using a logarithmic scale. This provides better distribution across the range, especially for handling keywords with low occurrence counts. 0.0-1.0 includes all keywords."
            )
            
            # Convert log scale range to actual counts
            min_count, max_count = self._get_keyword_count_range_from_log_scale(
                temp_filtered_keywords, log_range[0], log_range[1]
            )
            
            # Show summary of current selection
            if not temp_filtered_keywords.empty:
                total_keywords = len(temp_filtered_keywords)
                
                # Calculate actual keywords in range for verification
                grant_counts = temp_filtered_keywords['grants'].apply(len)
                actual_included = ((grant_counts >= min_count) & (grant_counts <= max_count)).sum()
                
                # Get the actual occurrence range for display
                if temp_filtered_keywords.empty:
                    occurrence_range_text = "No keywords available"
                else:
                    all_counts = grant_counts.sort_values()
                    if len(all_counts) > 0:
                        min_occurrence = all_counts.min()
                        max_occurrence = all_counts.max()
                        occurrence_range_text = f"Available range: {min_occurrence}-{max_occurrence} grants"
                    else:
                        occurrence_range_text = "No keywords available"
                
                st.caption(f"Selected range: {log_range[0]:.0%}-{log_range[1]:.0%} (log scale) → {min_count}-{max_count} grants → {actual_included} keywords | {occurrence_range_text}")
        
        return FilterConfig(funder_filter, source_filter, field_filter, keyword_type_filter, min_count, max_count)
    
    def _render_keyword_selection(self, filter_config: FilterConfig) -> KeywordSelectionConfig:
        """Render keyword selection options"""
        with st.expander("⚙️ Keyword Selection", expanded=True):
            method = st.radio(
                "Keyword selection method",
                options=["Top N keywords", "Random sample", "Specify custom keywords"],
                index=0,
                help="Choose how to select keywords to display"
            )
            
            if method == "Top N keywords":
                return self._render_top_n_selection()
            elif method == "Random sample":
                return self._render_random_selection(filter_config)
            else:
                return self._render_custom_selection(filter_config)
    
    def _render_display_settings(self) -> DisplayConfig:
        """Render display settings section"""
        with st.expander("📊 Display Settings", expanded=True):
            show_baseline = st.checkbox(
                "Show baseline (avg keywords/grant)", 
                value=True,
                help="Show cumulative average keywords per grant per year as baseline"
            )
            
            use_cumulative = st.checkbox(
                "Use cumulative counts",
                value=True,
                help="Show cumulative keyword counts over time (checked) or raw yearly counts (unchecked)"
            )
        
        return DisplayConfig(show_baseline, use_cumulative)
    
    def _render_top_n_selection(self) -> KeywordSelectionConfig:
        """Render top N keywords selection"""
        top_n = st.slider(
            "Number of top keywords to show", 
            min_value=0, 
            max_value=50, 
            value=10,
            help="Number of individual keyword lines to display (0 = show only averages)"
        )
        return KeywordSelectionConfig("Top N keywords", top_n, 0, [], False, None)
    
    def _render_random_selection(self, filter_config: FilterConfig) -> KeywordSelectionConfig:
        """Render random sampling selection"""
        # Calculate available keywords for the current filters
        filter_manager = FilterManager(st.session_state.get('keywords', pd.DataFrame()), 
                                     st.session_state.get('grants', pd.DataFrame()))
        available_keywords = filter_manager.get_available_keywords(filter_config)
        max_available = len(available_keywords)
        
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
        
        use_random_seed = st.checkbox(
            "Use random seed for reproducible results",
            value=True,
            help="Enable to get the same random sample each time"
        )
        
        random_seed = None
        if use_random_seed:
            random_seed = st.number_input(
                "Random seed",
                min_value=0,
                max_value=9999,
                value=42,
                help="Seed for random number generator"
            )
        
        if st.button("🎲 Generate New Random Sample", help="Click to get a different random selection"):
            if 'random_sample_counter' not in st.session_state:
                st.session_state.random_sample_counter = 0
            st.session_state.random_sample_counter += 1
        
        return KeywordSelectionConfig("Random sample", 0, random_sample_size, [], use_random_seed, random_seed)
    
    def _render_custom_selection(self, filter_config: FilterConfig) -> KeywordSelectionConfig:
        """Render custom keywords selection"""
        filter_manager = FilterManager(st.session_state.get('keywords', pd.DataFrame()), 
                                     st.session_state.get('grants', pd.DataFrame()))
        available_keywords = filter_manager.get_available_keywords(filter_config)
        
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
        
        return KeywordSelectionConfig("Specify custom keywords", 0, 0, custom_keywords, False, None)
    
    def _show_active_filters(self, funder_filter: List[str], source_filter: List[str], 
                           field_filter: List[str], keyword_type_filter: List[str]):
        """Display active filters information"""
        active_filters = []
        if funder_filter:
            active_filters.append(f"Funders: {', '.join(str(f) for f in funder_filter)}")
        if source_filter:
            active_filters.append(f"Sources: {', '.join(str(s) for s in source_filter)}")
        if field_filter:
            field_display_parts = []
            for field in field_filter[:3]:
                display_name = str(field).replace('_', ' ').title()
                field_display_parts.append(display_name)
            
            field_display = ', '.join(field_display_parts)
            if len(field_filter) > 3:
                field_display += f"... and {len(field_filter) - 3} more"
            active_filters.append(f"Research Fields: {field_display}")
        if keyword_type_filter:
            active_filters.append(f"Types: {', '.join(str(k) for k in keyword_type_filter)}")
            
        if active_filters:
            st.info(f"Active filters: {' | '.join(active_filters)}")


class TrendsVisualization:
    """Manages the trends visualization tab"""
    
    def __init__(self, keywords_df: pd.DataFrame, grants_df: pd.DataFrame):
        self.keywords_df = keywords_df
        self.grants_df = grants_df
        self.filter_manager = FilterManager(keywords_df, grants_df)
        self.keyword_selector = KeywordSelector(self.filter_manager)
    
    def render_tab(self, filter_config: FilterConfig, selection_config: KeywordSelectionConfig, 
                  display_config: DisplayConfig):
        """Render the trends visualization tab"""
        st.markdown("### Keyword Trends Over Time")
        st.markdown("Analyze how research keywords evolve over time with cumulative occurrence tracking.")
        
        should_generate = self._should_generate_visualization()
        
        if should_generate:
            if selection_config.method == "Specify custom keywords" and not selection_config.custom_keywords:
                st.error("Please select at least one keyword when using custom keyword selection.")
                return
            
            self._generate_visualization(filter_config, selection_config, display_config)
    
    def _should_generate_visualization(self) -> bool:
        """Determine if visualization should be generated"""
        if "trends_initialized" not in st.session_state:
            st.session_state.trends_initialized = True
            return True
        return True  # Always generate for now
    
    def _generate_visualization(self, filter_config: FilterConfig, selection_config: KeywordSelectionConfig, 
                              display_config: DisplayConfig):
        """Generate and display the visualization"""
        with st.spinner("Creating keyword trends visualization..."):
            selected_keywords, title = self.keyword_selector.select_keywords(filter_config, selection_config, display_config)
            
            title_with_filters = self._add_filter_info_to_title(title, filter_config)
            
            # Create visualization using the KeywordTrendsVisualizer class
            visualizer = KeywordTrendsVisualizer(self.keywords_df, self.grants_df)
            
            # Convert our FilterConfig to VisFilterConfig
            vis_filter_config = VisFilterConfig(
                funder_filter=filter_config.funder_filter if filter_config.funder_filter else None,
                source_filter=filter_config.source_filter if filter_config.source_filter else None,
                keyword_type_filter=filter_config.keyword_type_filter if filter_config.keyword_type_filter else None,
                field_filter=filter_config.field_filter if filter_config.field_filter else None,
                min_count=filter_config.min_count,
                max_count=filter_config.max_count
            )
            
            # Create visualization config
            vis_config = VisualizationConfig(
                title=title_with_filters,
                height=600,
                show_baseline=display_config.show_baseline,
                use_cumulative=display_config.use_cumulative,
                top_n=selection_config.top_n if selection_config.method == "Top N keywords" else 0,
                custom_keywords=selected_keywords if selected_keywords else None
            )
            
            fig_trends = visualizer.create_visualization(vis_filter_config, vis_config)
            
            if fig_trends is not None:
                st.plotly_chart(fig_trends, use_container_width=True)
                self._show_statistics(filter_config, selection_config, selected_keywords)
            else:
                st.warning("No data available for the selected parameters.")
    
    def _add_filter_info_to_title(self, title: str, filter_config: FilterConfig) -> str:
        """Add filter information to the visualization title"""
        filter_parts = []
        if filter_config.funder_filter:
            filter_parts.append(f"Funders: {', '.join(filter_config.funder_filter)}")
        if filter_config.source_filter:
            filter_parts.append(f"Sources: {', '.join(filter_config.source_filter)}")
        if filter_config.field_filter:
            field_display_parts = [field.replace('_', ' ').title()[:25] for field in filter_config.field_filter[:2]]
            field_display = ', '.join(field_display_parts)
            if len(filter_config.field_filter) > 2:
                field_display += f"... +{len(filter_config.field_filter) - 2} more"
            filter_parts.append(f"Research Fields: {field_display}")
        if filter_config.keyword_type_filter:
            filter_parts.append(f"Types: {', '.join(filter_config.keyword_type_filter)}")
        
        if filter_parts:
            return f"{title} | Filtered by {' | '.join(filter_parts)}"
        return title
    
    def _show_statistics(self, filter_config: FilterConfig, selection_config: KeywordSelectionConfig, 
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
    

class DataExploration:
    """Manages the data exploration tab"""
    
    def __init__(self, keywords_df: pd.DataFrame, grants_df: pd.DataFrame):
        self.keywords_df = keywords_df
        self.grants_df = grants_df
        self.filter_manager = FilterManager(keywords_df, grants_df)
    
    def render_tab(self, filter_config: FilterConfig):
        """Render the data exploration tab"""
        st.markdown("### Keywords Dataset")
        st.markdown("Explore the complete keywords dataset with filtering, search capabilities, and distribution analysis.")
        
        filtered_keywords = self.filter_manager.apply_keyword_filters(filter_config)
        
        if len(filtered_keywords) == 0:
            st.warning("No keywords found with current filters. Please adjust your filter settings.")
            return
        
        self._show_summary_statistics(filtered_keywords)
        self._create_histogram_visualization(filtered_keywords)
        search_term = self._render_search_input()
        display_keywords = self._apply_search_filter(filtered_keywords, search_term)
        self._display_keywords_dataframe(display_keywords)
    
    def _create_histogram_visualization(self, filtered_keywords: pd.DataFrame):
        """Create and display the histogram"""
        import plotly.graph_objects as go
        
        st.markdown("#### 📊 Keywords Occurrence Distribution")
        
        grant_counts = filtered_keywords['grants'].apply(len)
        
        # Create controls for histogram customization
        use_log_scale = st.checkbox("Use log scale", help="Useful when distribution is highly skewed", value=True)
        
        # Create histogram with automatic binning
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=grant_counts,
            name="Keyword Frequency",
            marker_color='rgba(55, 83, 109, 0.7)',
            marker_line=dict(color='rgba(55, 83, 109, 1.0)', width=1)
        ))
        
        fig.update_layout(
            title="Distribution of Keyword Occurrences Across Grants",
            xaxis_title="Number of Grants per Keyword",
            yaxis_title="Frequency",
            height=400,
            showlegend=False,
            template="plotly_white"
        )
        
        if use_log_scale:
            fig.update_yaxes(type="log")
            fig.update_xaxes(type="log")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_summary_statistics(self, filtered_keywords: pd.DataFrame):
        """Display summary statistics"""
        grant_counts = filtered_keywords['grants'].apply(len)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Keywords", len(self.keywords_df))
        with col2:
            st.metric("Filtered Keywords", len(filtered_keywords))
        with col3:
            st.metric("Mean Occurrences", f"{grant_counts.mean():.1f}")
        with col4:
            st.metric("Median Occurrences", f"{grant_counts.median():.0f}")
        with col5:
            st.metric("Max Occurrences", f"{grant_counts.max():.0f}")
    
    def _render_search_input(self) -> str:
        """Render search input and return search term"""
        st.markdown("#### 🔍 Search Keywords")
        return st.text_input(
            "Search in keyword names:",
            help="Enter text to search for specific keywords"
        )
    
    def _apply_search_filter(self, filtered_keywords: pd.DataFrame, search_term: str) -> pd.DataFrame:
        """Apply search filter to keywords"""
        if not search_term:
            return filtered_keywords
        
        if 'name' in filtered_keywords.columns:
            return filtered_keywords[
                filtered_keywords['name'].str.contains(search_term, case=False, na=False)
            ]
        else:
            return filtered_keywords[
                filtered_keywords.index.str.contains(search_term, case=False, na=False)
            ]
    
    def _display_keywords_dataframe(self, display_keywords: pd.DataFrame):
        """Display the keywords dataframe with controls"""
        st.markdown("#### 📊 Keywords Data")
        
        if len(display_keywords) > 0:
            # Create columns for controls
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Add row limit control for download
                max_rows_available = len(display_keywords)
                download_limit = st.number_input(
                    "Download row limit",
                    min_value=1,
                    max_value=max_rows_available,
                    value=min(1000, max_rows_available),
                    help=f"Limit the number of rows to download (max available: {max_rows_available})"
                )
            
            with col2:
                # Add download button
                display_df = self._prepare_display_dataframe(display_keywords, show_grants_detail=True)
                # Limit the dataframe for download
                download_df = display_df.head(download_limit)
                csv_data = download_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Keywords Data",
                    data=csv_data,
                    file_name=f"keywords_data_{download_limit}rows_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help=f"Download the top {download_limit} keywords as CSV file"
                )
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            st.info(f"Showing {len(display_keywords)} keywords" + 
                    (f" (filtered from {len(self.keywords_df)} total)" if len(display_keywords) < len(self.keywords_df) else ""))
        else:
            st.warning("No keywords found matching the current filters and search criteria.")
            st.info("Try adjusting your filters or search terms to see results.")

    def _prepare_display_dataframe(self, display_keywords: pd.DataFrame, show_grants_detail: bool = True) -> pd.DataFrame:
        """Prepare dataframe for display, sorted by grant count (descending)"""
        display_df = display_keywords.copy()
        display_df['grant_count'] = display_df['grants'].apply(len)
        
        # Sort by grant count in descending order (most grants first)
        display_df = display_df.sort_values('grant_count', ascending=False)
        
        column_order = []
        if 'name' in display_df.columns:
            column_order.append('name')
        if 'type' in display_df.columns:
            column_order.append('type')
        column_order.append('grant_count')
        
        # Add other columns except 'grants' and 'org' (we'll handle these separately)
        for col in display_df.columns:
            if col not in column_order and col not in ['grants', 'org']:
                column_order.append(col)
        
        # Always include grants column
        column_order.append('grants')
        
        available_columns = [col for col in column_order if col in display_df.columns]
        return display_df[available_columns]


class KeywordsPage:
    """Main page class that orchestrates all components"""
    
    def __init__(self):
        self.keywords_df = None
        self.grants_df = None
        self.categories_df = None
        self.setup_page()
    
    def setup_page(self):
        """Setup page configuration and load data"""
        setup_page_config("Keywords", "📈")
        clear_previous_page_state()
        
        st.header("📈 Keywords Analysis")
        st.markdown("Analyze research keywords with trend visualization and data exploration.")
        
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
    
    def run(self):
        """Main execution method"""
        # Create sidebar controls
        sidebar_controls = SidebarControls(self.keywords_df, self.grants_df, self.categories_df)
        
        # Get configurations from sidebar
        filter_config, selection_config, display_config = sidebar_controls.render_sidebar()
        
        # Create tabs
        tab1, tab2 = st.tabs(["📈 Trends Visualization", "📊 Keywords Data"])
        
        # Render trends visualization tab
        with tab1:
            trends_viz = TrendsVisualization(self.keywords_df, self.grants_df)
            trends_viz.render_tab(filter_config, selection_config, display_config)
        
        # Render data exploration tab
        with tab2:
            data_exploration = DataExploration(self.keywords_df, self.grants_df)
            data_exploration.render_tab(filter_config)
    
    @st.cache_data
    def _get_unique_research_fields_from_grants(_self):
        """Get unique research fields efficiently - cached computation"""
        division_to_field = {
            '30': 'AGRICULTURAL_VETERINARY_FOOD_SCIENCES',
            '31': 'BIOLOGICAL_SCIENCES',
            '32': 'BIOMEDICAL_CLINICAL_SCIENCES',
            '33': 'BUILT_ENVIRONMENT_DESIGN',
            '34': 'CHEMICAL_SCIENCES',
            '35': 'COMMERCE_MANAGEMENT_TOURISM_SERVICES',
            '36': 'CREATIVE_ARTS_WRITING',
            '37': 'EARTH_SCIENCES',
            '38': 'ECONOMICS',
            '39': 'EDUCATION',
            '40': 'ENGINEERING',
            '41': 'ENVIRONMENTAL_SCIENCES',
            '42': 'HEALTH_SCIENCES',
            '43': 'HISTORY_HERITAGE_ARCHAEOLOGY',
            '44': 'HUMAN_SOCIETY',
            '45': 'INDIGENOUS_STUDIES',
            '46': 'INFORMATION_COMPUTING_SCIENCES',
            '47': 'LANGUAGE_COMMUNICATION_CULTURE',
            '48': 'LAW_LEGAL_STUDIES',
            '49': 'MATHEMATICAL_SCIENCES',
            '50': 'PHILOSOPHY_RELIGIOUS_STUDIES',
            '51': 'PHYSICAL_SCIENCES',
            '52': 'PSYCHOLOGY'
        }
        
        unique_fields = set()
        
        # Add fields from primary FOR codes
        for val in _self.grants_df['for_primary'].dropna():
            division_code = str(int(val))[:2]
            if division_code in division_to_field:
                unique_fields.add(division_to_field[division_code])
        
        # Add fields from secondary FOR codes
        for for_codes_str in _self.grants_df['for'].dropna():
            for code in str(for_codes_str).split(','):
                code = code.strip()
                if code and len(code) >= 2:
                    division_code = code[:2]
                    if division_code in division_to_field:
                        unique_fields.add(division_to_field[division_code])
        
        return unique_fields


def main():
    """Main function to run the keywords page"""
    keywords_page = KeywordsPage()
    keywords_page.run()


if __name__ == "__main__":
    main()