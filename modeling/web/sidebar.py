"""
Unified Sidebar Controls for All Pages
Provides consistent filtering and display settings across grants, keywords, and categories pages.
"""
import streamlit as st
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from shared_utils import (
    create_research_field_options,
    get_unique_funders,
    get_unique_keyword_types,
    get_unique_research_fields,
    get_unique_research_fields_from_categories,
    get_unique_sources,
    load_css
)

load_css()

@dataclass
class FilterConfig:
    """Configuration for data filtering - what data to include"""
    funder_filter: List[str] = field(default_factory=list)
    source_filter: List[str] = field(default_factory=list)
    field_filter: List[str] = field(default_factory=list)
    keyword_type_filter: List[str] = field(default_factory=list)
    start_year_min: Optional[int] = 2000
    start_year_max: Optional[int] = None
    min_count: int = 0
    max_count: int = 999999
    search_term: str = ""
    use_all_for_codes: bool = False
    use_active_grant_period: bool = False


@dataclass
class DisplayConfig:
    """Configuration for display settings - how to visualize data"""
    selection_method: str = "top_n"  # top_n, random, custom
    num_entities: int = 10
    custom_entities: List[str] = field(default_factory=list)
    use_random_seed: bool = False
    random_seed: Optional[int] = None
    use_cumulative: bool = True
    show_baseline: bool = False
    metric: str = "funding"  # count, funding
    smooth_trends: bool = True
    smoothing_window: int = 5


@dataclass
class SidebarFeatures:
    """Feature flags to control which sidebar elements to display"""
    # Data filtering features
    show_funder_filter: bool = False
    show_source_filter: bool = False
    show_field_filter: bool = True
    show_keyword_type_filter: bool = False
    show_date_filter: bool = False
    show_count_range: bool = False
    show_search: bool = False
    show_active_period_toggle: bool = False
    
    # Display settings features
    show_selection_methods: List[str] = field(default_factory=lambda: ["top_n"])
    show_baseline: bool = False
    show_metrics: bool = False
    show_smoothing: bool = False
    
    # Default values
    default_num_entities: int = 10
    default_min_count: int = 0
    default_max_count: int = 999999


class SidebarControl:
    """Unified sidebar controls for grants, keywords, and categories pages"""
    
    def __init__(self, 
                 page_type: str,
                 grants_df: Optional[pd.DataFrame] = None,
                 keywords_df: Optional[pd.DataFrame] = None,
                 categories_df: Optional[pd.DataFrame] = None,
                 features: Optional[SidebarFeatures] = None):
        """
        Initialize unified sidebar controls
        
        Args:
            page_type: "grants", "keywords", or "categories"
            grants_df: Grants DataFrame (if applicable)
            keywords_df: Keywords DataFrame (if applicable)
            categories_df: Categories DataFrame (if applicable)
            features: Custom feature configuration (uses page-specific defaults if None)
        """
        self.page_type = page_type
        self.grants_df = grants_df
        self.keywords_df = keywords_df
        self.categories_df = categories_df
        
        # Set features based on page type if not provided
        self.features = features or self._get_default_features(page_type)
        
        # Cache available options
        self._cache_available_options()
    
    def _get_default_features(self, page_type: str) -> SidebarFeatures:
        """Get default feature configuration for each page type"""
        if page_type == "grants":
            return SidebarFeatures(
                show_source_filter=True,
                show_field_filter=True,
                show_date_filter=True,
                show_active_period_toggle=True,
                show_smoothing=True,
                show_metrics=True,
                show_selection_methods=["top_n"],
                default_num_entities=10
            )
        elif page_type == "keywords":
            return SidebarFeatures(
                show_funder_filter=True,
                show_source_filter=True,
                show_field_filter=True,
                show_keyword_type_filter=True,
                show_date_filter=True,
                show_count_range=True,
                show_active_period_toggle=True,
                show_smoothing=True,
                show_selection_methods=["top_n", "random", "custom"],
                show_baseline=True,
                show_metrics=True,
                default_num_entities=10,
                default_min_count=10,
                default_max_count=999999
            )
        elif page_type == "categories":
            return SidebarFeatures(
                show_source_filter=True,
                show_field_filter=True,
                show_count_range=True,
                show_search=True,
                show_active_period_toggle=True,
                show_smoothing=True,
                show_selection_methods=["top_n", "random", "custom"],
                show_metrics=True,
                default_num_entities=10,
                default_min_count=0,
                default_max_count=999999
            )
        else:
            return SidebarFeatures()
    
    def _cache_available_options(self):
        """Cache available filter options from data"""
        # Funders
        if self.features.show_funder_filter and self.grants_df is not None:
            self.unique_funders = get_unique_funders(self.grants_df)
        else:
            self.unique_funders = []
        
        # Sources
        if self.features.show_source_filter and self.grants_df is not None:
            self.unique_sources = get_unique_sources(self.grants_df)
        else:
            self.unique_sources = []
        
        # Research fields
        if self.features.show_field_filter:
            if self.categories_df is not None:
                unique_fields = get_unique_research_fields_from_categories(self.categories_df)
            else:
                unique_fields = get_unique_research_fields()
            self.field_options, self.field_values = create_research_field_options(unique_fields)
            self.option_to_field = dict(zip(self.field_options, self.field_values))
        else:
            self.field_options = []
            self.field_values = []
            self.option_to_field = {}
        
        # Keyword types
        if self.features.show_keyword_type_filter and self.keywords_df is not None:
            self.unique_keyword_types = get_unique_keyword_types(self.keywords_df)
        else:
            self.unique_keyword_types = []
        
        # Year range
        if self.features.show_date_filter and self.grants_df is not None:
            valid_years = self.grants_df['start_year'].dropna()
            self.min_year = int(valid_years.min()) if len(valid_years) > 0 else 2000
            self.max_year = int(valid_years.max()) if len(valid_years) > 0 else 2025
        else:
            self.min_year = 2000
            self.max_year = 2025
        
        # Count range for keywords/categories
        if self.features.show_count_range:
            if self.keywords_df is not None:
                self.max_possible_count = self.keywords_df['grants'].apply(len).max()
            elif self.categories_df is not None and 'keyword_count' in self.categories_df.columns:
                self.max_possible_count = self.categories_df['keyword_count'].max()
            else:
                self.max_possible_count = 1000
        else:
            self.max_possible_count = 1000
        
        # Cache available entities for custom selection
        if self.page_type == "keywords" and self.keywords_df is not None:
            if 'name' in self.keywords_df.columns:
                self.available_entities = sorted(self.keywords_df['name'].tolist())
            else:
                self.available_entities = sorted(self.keywords_df.index.tolist())
        elif self.page_type == "categories" and self.categories_df is not None:
            if 'name' in self.categories_df.columns:
                self.available_entities = sorted(self.categories_df['name'].tolist())
            else:
                self.available_entities = sorted(self.categories_df.index.tolist())
        else:
            self.available_entities = []
    
    def render_sidebar(self) -> Tuple[FilterConfig, DisplayConfig]:
        """Render the unified sidebar and return configurations"""
        st.sidebar.empty()
        st.logo("web/static/media/logo.png", size="large", icon_image="web/static/media/favicon.png")
        
        with st.sidebar:
            st.title("Settings")
            
            # Expander 1: Data Filtering
            filter_config = self._render_data_filters_expander()
            
            # Expander 2: Display Settings
            display_config = self._render_display_settings_expander()
            
            # Show active filters summary
            self._show_active_filters(filter_config)
        
        return filter_config, display_config
    
    def _render_data_filters_expander(self) -> FilterConfig:
        """Render the data filtering expander"""
        with st.expander("Data Filtering", expanded=True):
            st.markdown("Select which data to include in the analysis")
            
            # Funder filter
            if self.features.show_funder_filter:
                funder_filter = st.multiselect(
                    "Filter by Funder",
                    options=self.unique_funders,
                    default=[],
                    help="Select specific funders to include"
                )
            else:
                funder_filter = []
            
            # Source filter
            if self.features.show_source_filter:
                source_filter = st.multiselect(
                    "Filter by Source",
                    options=self.unique_sources,
                    default=["arc.gov.au"],
                    help="Select specific data sources"
                )
            else:
                source_filter = []
            
            # Research field filter
            use_all_for_codes = False
            if self.features.show_field_filter:
                selected_field_options = st.multiselect(
                    "Filter by Subject",
                    options=self.field_options,
                    default=[],
                    help="Select subjects to focus on"
                )
                field_filter = [self.option_to_field[opt] for opt in selected_field_options]
                use_all_for_codes = len(field_filter) == 0
            else:
                field_filter = []
                use_all_for_codes = False
            
            # Keyword type filter
            if self.features.show_keyword_type_filter:
                keyword_type_filter = st.multiselect(
                    "Filter by Keyword Type",
                    options=self.unique_keyword_types,
                    default=[],
                    help="Select keyword types to include"
                )
            else:
                keyword_type_filter = []
            
            # Date range filter
            if self.features.show_date_filter:
                st.markdown("**Date Range**")
                col1, col2 = st.columns(2)
                with col1:
                    default_start_year = min(max(2000, self.min_year), self.max_year)
                    start_year_min = st.number_input(
                        "From Year",
                        min_value=self.min_year,
                        max_value=self.max_year,
                        value=default_start_year,
                        step=1
                    )
                with col2:
                    start_year_max = st.number_input(
                        "To Year",
                        min_value=self.min_year,
                        max_value=self.max_year,
                        value=self.max_year,
                        step=1
                    )
            else:
                start_year_min = None
                start_year_max = None

            if self.features.show_active_period_toggle:
                use_active_grant_period = st.toggle(
                    "Treat grants as active through their end year",
                    value=False,
                    help="When enabled, grant counts include every year between start and end year."
                )
            else:
                use_active_grant_period = False
            
            # Count range filter
            if self.features.show_count_range:
                st.markdown("**Occurrence Count Range**")
                count_label = "Keywords" if self.page_type == "categories" else "Grants"
                col1, col2 = st.columns(2)
                with col1:
                    min_count = st.number_input(
                        f"Min {count_label}",
                        min_value=0,
                        max_value=int(self.max_possible_count),
                        value=self.features.default_min_count,
                        step=1
                    )
                with col2:
                    max_count = st.number_input(
                        f"Max {count_label}",
                        min_value=0,
                        max_value=int(self.max_possible_count),
                        value=min(self.features.default_max_count, int(self.max_possible_count)),
                        step=1
                    )
            else:
                min_count = 0
                max_count = 999999
            
            # Search filter
            if self.features.show_search:
                search_term = st.text_input(
                    "Search Terms",
                    value="",
                    placeholder="Search in names or descriptions...",
                    help="Filter by text search"
                )
            else:
                search_term = ""
        
        return FilterConfig(
            funder_filter=funder_filter,
            source_filter=source_filter,
            field_filter=field_filter,
            keyword_type_filter=keyword_type_filter,
            start_year_min=start_year_min,
            start_year_max=start_year_max,
            min_count=min_count,
            max_count=max_count,
            search_term=search_term,
            use_all_for_codes=use_all_for_codes,
            use_active_grant_period=use_active_grant_period
        )
    
    def _render_display_settings_expander(self) -> DisplayConfig:
        """Render the display settings expander"""
        with st.expander("Display Settings", expanded=True):
            st.markdown("Configure how to visualize the filtered data")
            
            # Selection method
            method_options = {
                "top_n": "Top N by Count",
                "random": "Random Sample",
                "custom": "Custom Selection",
            }
            
            available_methods = {k: v for k, v in method_options.items() 
                                if k in self.features.show_selection_methods}
            
            if len(available_methods) > 1:
                selection_method = st.selectbox(
                    "Selection Method",
                    options=list(available_methods.keys()),
                    format_func=lambda x: available_methods[x],
                    help="How to select entities to display in the trend chart"
                )
            else:
                selection_method = list(available_methods.keys())[0] if available_methods else "top_n"
            
            # Number of entities to display
            entity_label = {
                "grants": "Funders",
                "keywords": "Keywords",
                "categories": "Categories"
            }.get(self.page_type, "Entities")
            
            num_entities = st.number_input(
                f"Number of {entity_label} to Display",
                min_value=1,
                max_value=50,
                value=self.features.default_num_entities,
                step=1,
                help=f"How many {entity_label.lower()} to show in the visualization"
            )
            
            # Custom selection (only if method is custom)
            if selection_method == "custom":
                if hasattr(self, 'available_entities') and self.available_entities:
                    # Set default categories for categories page
                    default_selection = []
                    if self.page_type == "categories":
                        default_categories = [
                            "Noise as a Physical Signal in Sensing and Cosmology",
                            "Nanomechanical Computing and Molecular Machines",
                            "Vacuum and Classical Re-Imagined Nanosystems",
                            "PFAS and Fluorinated Pollutant Remediation and Risk Management",
                            "Emergent Molecular Technologies",
                            "Cognitive and Neural Processing Mechanisms"
                        ]
                        # Only include defaults that exist in available entities
                        default_selection = [cat for cat in default_categories if cat in self.available_entities]
                    
                    custom_entities = st.multiselect(
                        f"Select Custom {entity_label}",
                        options=self.available_entities,
                        default=default_selection,
                        help=f"Choose specific {entity_label.lower()} to display"
                    )
                else:
                    st.warning(f"No {entity_label.lower()} available for custom selection")
                    custom_entities = []
            else:
                custom_entities = []
            
            # Random seed (only if method is random)
            if selection_method == "random":
                use_random_seed = st.checkbox(
                    "Use Random Seed",
                    value=False,
                    help="Use a fixed seed for reproducible random sampling"
                )
                if use_random_seed:
                    random_seed = st.number_input(
                        "Random Seed",
                        min_value=0,
                        max_value=99999,
                        value=42,
                        step=1
                    )
                else:
                    random_seed = None
            else:
                use_random_seed = False
                random_seed = None
            
            # Cumulative option
            use_cumulative = st.checkbox(
                "Show Cumulative Values",
                value=False,
                help="Display cumulative counts over time (vs yearly counts)"
            )
            
            # Baseline option (keywords only)
            if self.features.show_baseline:
                show_baseline = st.checkbox(
                    "Show Baseline",
                    value=False,
                    help="Show average keywords per grant baseline"
                )
            else:
                show_baseline = False
            
            # Metrics options (categories only)
            if self.features.show_metrics:
                st.markdown("**Metrics Configuration**")
                metric = st.selectbox(
                    "Y-axis metric",
                    options=["count", "funding"],
                    format_func=lambda x: "Grant Count" if x == "count" else "Funding Amount",
                    index=["count", "funding"].index("funding"),
                    help="Determines both the displayed values and how entities are ranked"
                )
            else:
                metric = "funding"
            # Smoothing options
            if self.features.show_smoothing:
                smooth_trends = st.toggle(
                    "Smooth trend lines",
                    value=True,
                    help="Apply a rolling average to the trend data"
                )
                if smooth_trends:
                    smoothing_window = st.slider(
                        "Rolling window (years)",
                        min_value=2,
                        max_value=10,
                        value=5,
                        step=1,
                        help="Number of years included in the rolling average"
                    )
                else:
                    smoothing_window = 1
            else:
                smooth_trends = True
                smoothing_window = 5
        
        return DisplayConfig(
            selection_method=selection_method,
            num_entities=num_entities,
            custom_entities=custom_entities,
            use_random_seed=use_random_seed,
            random_seed=random_seed,
            use_cumulative=use_cumulative,
            show_baseline=show_baseline,
            metric=metric,
            smooth_trends=smooth_trends,
            smoothing_window=int(smoothing_window) if smooth_trends else 1
        )
    
    def _show_active_filters(self, filter_config: FilterConfig):
        """Display summary of active filters"""
        active_filters = []
        
        if filter_config.funder_filter:
            active_filters.append(f"Funders: {len(filter_config.funder_filter)} selected")
        
        if filter_config.source_filter:
            active_filters.append(f"Sources: {len(filter_config.source_filter)} selected")
        
        if filter_config.field_filter:
            active_filters.append(f"Fields: {len(filter_config.field_filter)} selected")
        
        if filter_config.keyword_type_filter:
            active_filters.append(f"Types: {len(filter_config.keyword_type_filter)} selected")
        
        if filter_config.start_year_min or filter_config.start_year_max:
            active_filters.append(f"Years: {filter_config.start_year_min}-{filter_config.start_year_max}")
        if filter_config.use_active_grant_period:
            active_filters.append("Active grant period enabled")
        
        if self.features.show_count_range and (filter_config.min_count > 0 or filter_config.max_count < 999999):
            active_filters.append(f"Count: {filter_config.min_count}-{filter_config.max_count}")
        
        if filter_config.search_term:
            active_filters.append(f"Search: '{filter_config.search_term[:20]}...'")
        
        if active_filters:
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Active Filters:**")
            for filter_desc in active_filters:
                st.sidebar.markdown(f"- {filter_desc}")
