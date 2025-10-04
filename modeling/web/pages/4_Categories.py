"""
Category Analysis Page
Visualize and analyze research categories generated from keyword clustering and categorization.
"""
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import json
from collections import Counter

# Add the web directory to Python path to import shared_utils
web_dir = str(Path(__file__).parent.parent)
if web_dir not in sys.path:
    sys.path.insert(0, web_dir)

from shared_utils import (
    setup_page_config, load_data,
    get_unique_values_from_data, create_research_field_options,
    field_to_division_codes, has_research_field_simple, clear_previous_page_state
)

import config
from visualisation import PopularityTrendsVisualizer, EntityTrendsConfig, DataExplorer, DataExplorerConfig


@dataclass
class CategoryFilterConfig:
    """Configuration for category data filtering"""
    field_filter: List[str]
    min_keywords: int
    max_keywords: int
    search_term: str
    category_names: List[str]


@dataclass
class CategoryTrendsConfig:
    """Configuration for category trends visualization"""
    num_categories: int
    use_cumulative: bool
    ranking_metric: str  # "grant_count" or "total_funding" - for selecting which categories to show
    display_metric: str  # "grant_count" or "total_funding" - for y-axis values


class CategoryDataManager:
    """Manages category data loading and filtering operations"""
    
    def __init__(self):
        self.categories_df = None
        self.field_options = None
        self._load_data()
    
    def _load_data(self):
        """Load category data"""
        try:
            self.categories_df = config.Categories.load()
            if self.categories_df is not None and not self.categories_df.empty:
                # Process keywords field if it's stored as string
                if 'keywords' in self.categories_df.columns:
                    self.categories_df['keyword_count'] = self.categories_df['keywords'].apply(
                        lambda x: len(x) if isinstance(x, list) else 0
                    )
                
                # Get unique field of research values
                if 'field_of_research' in self.categories_df.columns:
                    unique_fields = self.categories_df['field_of_research'].dropna().unique()
                    self.field_options = sorted([str(field) for field in unique_fields])
                else:
                    self.field_options = []
            else:
                st.warning("No category data found. Please run the categorization process first.")
                self.field_options = []
        except Exception as e:
            st.error(f"Error loading category data: {e}")
            self.categories_df = pd.DataFrame()
            self.field_options = []
    
    def get_filtered_data(self, filter_config: CategoryFilterConfig) -> pd.DataFrame:
        """Apply filters to category data"""
        if self.categories_df is None or self.categories_df.empty:
            return pd.DataFrame()
        
        filtered_df = self.categories_df.copy()
        
        # Field filter
        if filter_config.field_filter:
            filtered_df = filtered_df[
                filtered_df['field_of_research'].isin(filter_config.field_filter)
            ]
        
        # Keyword count filter
        filtered_df = filtered_df[
            (filtered_df['keyword_count'] >= filter_config.min_keywords) &
            (filtered_df['keyword_count'] <= filter_config.max_keywords)
        ]
        
        # Search term filter
        if filter_config.search_term:
            search_mask = (
                filtered_df['name'].str.contains(filter_config.search_term, case=False, na=False) |
                filtered_df['description'].str.contains(filter_config.search_term, case=False, na=False)
            )
            filtered_df = filtered_df[search_mask]
        
        # Specific category names filter
        if filter_config.category_names:
            filtered_df = filtered_df[
                filtered_df['name'].isin(filter_config.category_names)
            ]
        
        return filtered_df


class CategoryVisualizer:
    """Creates various visualizations for category data"""
    
    @staticmethod
    def create_field_distribution_chart(categories_df: pd.DataFrame) -> go.Figure:
        """Create a bar chart showing category distribution by field of research"""
        if categories_df.empty:
            return go.Figure()
        
        field_counts = categories_df['field_of_research'].value_counts()
        
        fig = px.bar(
            x=field_counts.index,
            y=field_counts.values,
            title="Category Distribution by Field of Research",
            labels={'x': 'Field of Research', 'y': 'Number of Categories'}
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_keyword_size_distribution(categories_df: pd.DataFrame) -> go.Figure:
        """Create a histogram showing distribution of category sizes (number of keywords)"""
        if categories_df.empty:
            return go.Figure()
        
        fig = px.histogram(
            categories_df,
            x='keyword_count',
            nbins=20,
            title="Distribution of Category Sizes (Number of Keywords per Category)",
            labels={'keyword_count': 'Number of Keywords', 'count': 'Number of Categories'}
        )
        
        fig.update_layout(height=400)
        return fig
    
    @staticmethod
    def create_top_categories_chart(categories_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
        """Create a horizontal bar chart of top N largest categories"""
        if categories_df.empty:
            return go.Figure()
        
        top_categories = categories_df.nlargest(top_n, 'keyword_count')
        
        fig = px.bar(
            top_categories,
            x='keyword_count',
            y='name',
            orientation='h',
            title=f"Top {top_n} Largest Categories by Number of Keywords",
            labels={'keyword_count': 'Number of Keywords', 'name': 'Category Name'}
        )
        
        fig.update_layout(
            height=max(400, top_n * 25),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    @staticmethod
    def create_field_keyword_heatmap(categories_df: pd.DataFrame) -> go.Figure:
        """Create a heatmap showing keyword distribution across fields"""
        if categories_df.empty:
            return go.Figure()
        
        # Create bins for keyword counts
        categories_df = categories_df.copy()
        categories_df['size_bin'] = pd.cut(
            categories_df['keyword_count'],
            bins=[0, 5, 10, 20, 50, 100, float('inf')],
            labels=['1-5', '6-10', '11-20', '21-50', '51-100', '100+']
        )
        
        # Create cross-tabulation
        heatmap_data = pd.crosstab(
            categories_df['field_of_research'],
            categories_df['size_bin']
        )
        
        fig = px.imshow(
            heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            title="Category Size Distribution Across Fields of Research",
            labels={'x': 'Category Size (Keywords)', 'y': 'Field of Research', 'color': 'Number of Categories'},
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(height=600)
        return fig


class SidebarControls:
    """Handles sidebar UI controls for category filtering"""
    
    def __init__(self, data_manager: CategoryDataManager):
        self.data_manager = data_manager
    
    def render_sidebar(self) -> Tuple[CategoryFilterConfig, CategoryTrendsConfig]:
        """Render sidebar controls and return filter and trends configurations"""
        st.sidebar.empty()
        
        with st.sidebar:
            st.header("🔧 Category Filters")
            
            # Field of research filter
            field_filter = st.multiselect(
                "Field of Research",
                options=self.data_manager.field_options,
                default=[],
                help="Filter categories by field of research"
            )
            
            # Get keyword count range for sliders
            if not self.data_manager.categories_df.empty:
                min_possible = int(self.data_manager.categories_df['keyword_count'].min())
                max_possible = int(self.data_manager.categories_df['keyword_count'].max())
            else:
                min_possible, max_possible = 0, 100
            
            # Keyword count filter
            st.subheader("Category Size Filter")
            min_keywords, max_keywords = st.slider(
                "Number of keywords in category",
                min_value=min_possible,
                max_value=max_possible,
                value=(min_possible, max_possible),
                help="Filter categories by number of keywords they contain"
            )
            
            # Search filter
            search_term = st.text_input(
                "Search categories",
                placeholder="Search in category names or descriptions...",
                help="Filter categories by text search in names or descriptions"
            )
            
            # Category Trends Settings
            st.markdown("---")
            st.header("📈 Trends Settings")
            
            # Number of categories to display
            num_categories = st.slider(
                "Number of categories to display",
                min_value=1,
                max_value=15,
                value=5,
                help="Select how many top categories to show in the trends plot"
            )
            
            # Display options
            use_cumulative = st.checkbox(
                "Show cumulative trends",
                value=True,
                help="Show cumulative category occurrences over time instead of yearly counts"
            )
            
            # Ranking metric selection (which categories to show)
            ranking_metric = st.radio(
                "Rank categories by:",
                options=["grant_count", "total_funding"],
                format_func=lambda x: "Number of Grants" if x == "grant_count" else "Total Funding Amount",
                help="Choose how to select the top categories to display"
            )
            
            # Display metric selection (y-axis values)
            display_metric = st.radio(
                "Y-axis shows:",
                options=["grant_count", "total_funding"],
                format_func=lambda x: "Grant Count" if x == "grant_count" else "Funding Amount",
                help="Choose what metric to display on the y-axis"
            )
            
            # Update visualization button
            update_trends = st.button("🔄 Update Category Trends", type="secondary")
            
            # Show active filters
            self._show_active_filters(field_filter, min_keywords, max_keywords, search_term)
        
        filter_config = CategoryFilterConfig(
            field_filter=field_filter,
            min_keywords=min_keywords,
            max_keywords=max_keywords,
            search_term=search_term,
            category_names=[]
        )
        
        trends_config = CategoryTrendsConfig(
            num_categories=num_categories,
            use_cumulative=use_cumulative,
            ranking_metric=ranking_metric,
            display_metric=display_metric
        )
        
        return filter_config, trends_config
    
    def _show_active_filters(self, field_filter: List[str], min_keywords: int, 
                           max_keywords: int, search_term: str):
        """Display active filters information"""
        active_filters = []
        
        if field_filter:
            active_filters.append(f"Fields: {len(field_filter)} selected")
        
        if search_term:
            active_filters.append(f"Search: '{search_term}'")
        
        if active_filters:
            st.sidebar.info("**Active Filters:**\n" + "\n".join(f"• {f}" for f in active_filters))


class OverviewTab:
    """Manages the overview tab with summary statistics and distributions"""
    
    def __init__(self, data_manager: CategoryDataManager):
        self.data_manager = data_manager
        self.visualizer = CategoryVisualizer()
    
    def render_tab(self, filter_config: CategoryFilterConfig, trends_config: CategoryTrendsConfig):
        """Render the overview tab"""
        st.markdown("### Category Overview")
        st.markdown("Summary statistics and distributions of research categories.")
        
        filtered_data = self.data_manager.get_filtered_data(filter_config)
        
        # Summary statistics
        self._show_summary_stats(filtered_data)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig_field_dist = self.visualizer.create_field_distribution_chart(filtered_data)
            st.plotly_chart(fig_field_dist, use_container_width=True, key="overview_field_dist")
        
        with col2:
            fig_size_dist = self.visualizer.create_keyword_size_distribution(filtered_data)
            st.plotly_chart(fig_size_dist, use_container_width=True, key="overview_size_dist")
        
        # Top categories
        fig_top_categories = self.visualizer.create_top_categories_chart(filtered_data)
        st.plotly_chart(fig_top_categories, use_container_width=True, key="overview_top_categories")
    
    def _show_summary_stats(self, filtered_data: pd.DataFrame):
        """Display summary statistics"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Categories", len(filtered_data))
        
        with col2:
            total_keywords = filtered_data['keyword_count'].sum() if not filtered_data.empty else 0
            st.metric("Total Keywords", total_keywords)
        
        with col3:
            avg_size = filtered_data['keyword_count'].mean() if not filtered_data.empty else 0
            st.metric("Avg Category Size", f"{avg_size:.1f}")
        
        with col4:
            unique_fields = filtered_data['field_of_research'].nunique() if not filtered_data.empty else 0
            st.metric("Fields Covered", unique_fields)
        
        with col5:
            if not filtered_data.empty:
                largest_category = filtered_data.loc[filtered_data['keyword_count'].idxmax(), 'keyword_count']
            else:
                largest_category = 0
            st.metric("Largest Category", largest_category)


class FieldAnalysisTab:
    """Manages the field analysis tab focusing on field of research distributions"""
    
    def __init__(self, data_manager: CategoryDataManager):
        self.data_manager = data_manager
        self.visualizer = CategoryVisualizer()
    
    def render_tab(self, filter_config: CategoryFilterConfig, trends_config: CategoryTrendsConfig):
        """Render the field analysis tab"""
        st.markdown("### Field of Research Analysis")
        st.markdown("Analyze how categories are distributed across different fields of research.")
        
        filtered_data = self.data_manager.get_filtered_data(filter_config)
        
        if filtered_data.empty:
            st.warning("No data available with current filters.")
            return
        
        # Field distribution
        fig_field_dist = self.visualizer.create_field_distribution_chart(filtered_data)
        st.plotly_chart(fig_field_dist, use_container_width=True, key="field_analysis_field_dist")
        
        # Heatmap showing size distribution across fields
        fig_heatmap = self.visualizer.create_field_keyword_heatmap(filtered_data)
        st.plotly_chart(fig_heatmap, use_container_width=True, key="field_analysis_heatmap")
        
        # Detailed field breakdown
        self._show_field_breakdown(filtered_data)
    
    def _show_field_breakdown(self, filtered_data: pd.DataFrame):
        """Show detailed breakdown by field"""
        st.subheader("Detailed Field Breakdown")
        
        field_stats = filtered_data.groupby('field_of_research').agg({
            'name': 'count',
            'keyword_count': ['sum', 'mean', 'std']
        }).round(2)
        
        field_stats.columns = ['Category Count', 'Total Keywords', 'Avg Keywords', 'Std Keywords']
        field_stats = field_stats.sort_values('Category Count', ascending=False)
        
        st.dataframe(field_stats, use_container_width=True)


class CategoryExplorerTab:
    """Manages the category explorer tab for detailed category browsing"""
    
    def __init__(self, data_manager: CategoryDataManager):
        self.data_manager = data_manager
        self.data_explorer = DataExplorer()
    
    def render_tab(self, filter_config: CategoryFilterConfig, trends_config: CategoryTrendsConfig):
        """Render the category explorer tab"""
        st.markdown("### Category Explorer")
        st.markdown("Browse and explore individual categories in detail.")
        
        filtered_data = self.data_manager.get_filtered_data(filter_config)
        
        if filtered_data.empty:
            st.warning("No categories found with current filters.")
            return
        
        # Category selection for detailed view
        selected_category = self._render_category_selector(filtered_data)
        
        if selected_category:
            self._show_category_details(filtered_data, selected_category)
        
        # Use DataExplorer for category table
        self._show_category_table_with_explorer(filtered_data)
    
    def _render_category_selector(self, filtered_data: pd.DataFrame) -> Optional[str]:
        """Render category selection dropdown"""
        category_options = [""] + sorted(filtered_data['name'].tolist())
        
        selected = st.selectbox(
            "Select a category to explore in detail:",
            options=category_options,
            help="Choose a category to see its detailed information"
        )
        
        return selected if selected else None
    
    def _show_category_details(self, filtered_data: pd.DataFrame, category_name: str):
        """Show detailed information for selected category"""
        category_row = filtered_data[filtered_data['name'] == category_name].iloc[0]
        
        st.subheader(f"📂 {category_row['name']}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Description:**")
            st.write(category_row['description'])
            
            st.markdown("**Keywords:**")
            if isinstance(category_row['keywords'], list):
                keywords_text = ", ".join(category_row['keywords'])
            else:
                keywords_text = str(category_row['keywords'])
            st.write(keywords_text)
        
        with col2:
            st.metric("Field of Research", category_row['field_of_research'])
            st.metric("Number of Keywords", category_row['keyword_count'])
    
    def _show_category_table_with_explorer(self, filtered_data: pd.DataFrame):
        """Show table of all categories using DataExplorer"""
        # Prepare display dataframe
        display_df = self._prepare_category_explorer_data(filtered_data)
        
        # Configure DataExplorer for categories
        config = DataExplorerConfig(
            title="All Categories",
            description="Search and explore research categories with their keywords and descriptions.",
            search_columns=['name', 'description', 'field_of_research', 'keywords_formatted'],
            search_placeholder="Search by category name, description, field of research, or keywords...",
            search_help="Enter text to search across category names, descriptions, fields, and keywords",
            display_columns=['name', 'field_of_research', 'keyword_count', 'keywords_formatted', 'description'],
            column_formatters={
                'description': lambda x: (x[:150] + "...") if len(str(x)) > 150 else str(x),
                'keywords_formatted': lambda x: x  # Already formatted
            },
            column_renames={
                'name': 'Category Name',
                'field_of_research': 'Field of Research',
                'keyword_count': 'Keywords Count',
                'keywords_formatted': 'Keywords',
                'description': 'Description'
            },
            statistics_columns=['keyword_count'],
            max_display_rows=50,
            show_statistics=True,
            show_data_info=True,
            enable_download=True
        )
        
        # Render the explorer
        self.data_explorer.render_explorer(display_df, config)
    
    def _prepare_category_explorer_data(self, filtered_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare categories data for the DataExplorer"""
        display_df = filtered_data[['name', 'field_of_research', 'keyword_count', 'description', 'keywords']].copy()
        display_df = display_df.sort_values('keyword_count', ascending=False)
        
        # Format keywords for display and search
        def format_keywords(keywords):
            if isinstance(keywords, list):
                # Join keywords with commas, but truncate if too long
                keywords_str = ", ".join(keywords)
                if len(keywords_str) > 100:  # Truncate long keyword lists
                    return keywords_str[:97] + "..."
                return keywords_str
            else:
                return str(keywords) if keywords else ""
        
        display_df['keywords_formatted'] = display_df['keywords'].apply(format_keywords)
        
        return display_df


class CategoryTrendsTab:
    """Manages the category trends over time tab"""
    
    def __init__(self, data_manager: CategoryDataManager):
        self.data_manager = data_manager
    
    def render_tab(self, filter_config: CategoryFilterConfig, trends_config: CategoryTrendsConfig):
        """Render the category trends tab with automatic visualization"""
        st.markdown("### Category Trends Over Time")
        st.markdown("Analyze how research categories evolve over time based on their associated grants.")
        
        filtered_data = self.data_manager.get_filtered_data(filter_config)
        
        if filtered_data.empty:
            st.warning("No categories found with current filters.")
            return
        
        # Load additional data needed for trends
        keywords_df, grants_df, _ = load_data()
        
        if keywords_df is None or grants_df is None:
            st.error("Unable to load keywords or grants data needed for trends analysis.")
            return
        
        # Show data info
        st.info(f"Processing {len(filtered_data)} categories with {len(keywords_df)} keywords. "
               f"Ranking by: {'Grant Count' if trends_config.ranking_metric == 'grant_count' else 'Total Funding Amount'}, "
               f"Displaying: {'Grant Count' if trends_config.display_metric == 'grant_count' else 'Funding Amount'}")
        
        # Initialize session state for trends
        if 'trends_data' not in st.session_state:
            st.session_state.trends_data = None
            st.session_state.trends_settings = None
        
        # Check if we need to regenerate (settings changed or first time)
        current_settings = (filter_config, trends_config)
        need_regenerate = (st.session_state.trends_settings != current_settings or 
                          st.session_state.trends_data is None)
        
        # Auto-generate or regenerate if needed
        if need_regenerate:
            with st.spinner("Creating category trends visualization..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    fig, category_data = self._create_category_trends_plot(
                        filtered_data, keywords_df, grants_df, trends_config, progress_bar, status_text
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("Visualization complete!")
                    
                    # Store in session state
                    st.session_state.trends_data = fig
                    st.session_state.trends_category_data = category_data
                    st.session_state.trends_settings = current_settings
                    
                except Exception as e:
                    st.error(f"Error creating visualization: {str(e)}")
                    st.session_state.trends_data = None
                    st.session_state.trends_category_data = None
                    return
                finally:
                    progress_bar.empty()
                    status_text.empty()
        
        # Display the visualization
        if st.session_state.trends_data:
            st.plotly_chart(st.session_state.trends_data, use_container_width=True, key="category_trends_plot")
            # Show statistics if we have category data
            if hasattr(st.session_state, 'trends_category_data') and st.session_state.trends_category_data:
                self._show_trends_statistics(trends_config, st.session_state.trends_category_data)
        else:
            st.warning("Unable to create trends visualization. No valid data found.")
    
    def _create_category_trends_plot(self, categories_df: pd.DataFrame, keywords_df: pd.DataFrame, 
                                   grants_df: pd.DataFrame, trends_config: CategoryTrendsConfig,
                                   progress_bar, status_text) -> Tuple[Optional[go.Figure], Dict[str, Dict[str, Any]]]:
        """Create the category trends plot using the generalized PopularityTrendsVisualizer"""
        
        # Step 1: Prepare indexed DataFrames for efficient access
        status_text.text("Indexing data for efficient access...")
        progress_bar.progress(10)
        
        # Set keyword names as index for fast lookup
        keywords_indexed = keywords_df.set_index('name')
        
        # Set grant IDs as index and clean grant data
        grants_indexed = grants_df.set_index('id').copy()
        grants_indexed = grants_indexed[pd.notna(grants_indexed['start_year'])]
        grants_indexed['start_year'] = grants_indexed['start_year'].astype(int)
        grants_indexed['funding_amount'] = grants_indexed['funding_amount'].fillna(0)
        
        # Step 2: Process categories using pandas operations
        status_text.text("Processing categories...")
        progress_bar.progress(30)
        
        category_grant_data = self._map_categories_to_grants_pandas(
            categories_df, keywords_indexed, grants_indexed, progress_bar, status_text
        )
        
        if not category_grant_data:
            return None, {}
        
        # Step 3: Prepare data for the generalized visualizer
        status_text.text("Preparing data for visualization...")
        progress_bar.progress(70)
        
        # Convert category data to the format expected by PopularityTrendsVisualizer
        viz_data = self._prepare_category_viz_data(category_grant_data, grants_indexed, trends_config)
        
        if viz_data.empty:
            return None, {}
        
        # Step 4: Configure and create visualization using generalized visualizer
        status_text.text("Creating visualization...")
        progress_bar.progress(80)
        
        # Create progress callback for the visualizer
        def progress_callback(message, percent):
            status_text.text(message)
            progress_bar.progress(80 + int(percent * 0.2))  # Use remaining 20% for visualization
        
        # Configure the visualizer
        ranking_text = "Grant Count" if trends_config.ranking_metric == "grant_count" else "Total Funding"
        display_text = "Grant Count" if trends_config.display_metric == "grant_count" else "Funding Amount"
        cumulative_text = "Cumulative" if trends_config.use_cumulative else "Yearly"
        
        viz_config = EntityTrendsConfig(
            entity_column='category',
            time_column='year',
            value_column=trends_config.display_metric,
            ranking_metric=trends_config.ranking_metric,
            max_entities=trends_config.num_categories,
            aggregation_method='sum',
            use_cumulative=trends_config.use_cumulative,
            chart_type='line',
            title=f"{cumulative_text} Category Trends Over Time (Top {trends_config.num_categories} by {ranking_text})",
            x_axis_label='Year',
            y_axis_label=f'{"Cumulative" if trends_config.use_cumulative else "Yearly"} {display_text}',
            height=600
        )
        
        # Create the visualization
        visualizer = PopularityTrendsVisualizer()
        fig = visualizer.create_trends_visualization(viz_data, viz_config, progress_callback)
        
        # Convert category_grant_data back to dictionary format for statistics
        category_data_dict = category_grant_data
        
        return fig, category_data_dict
    
    def _prepare_category_viz_data(self, category_grant_data: Dict[str, Dict[str, Any]], 
                                  grants_indexed: pd.DataFrame, trends_config: CategoryTrendsConfig) -> pd.DataFrame:
        """Prepare category data in the format expected by PopularityTrendsVisualizer"""
        viz_data = []
        
        for category_name, category_info in category_grant_data.items():
            grant_ids = category_info['grant_details']
            
            if not grant_ids:
                continue
            
            # Get grant data for this category
            category_grants = grants_indexed.loc[grant_ids].copy()
            
            # Group by year and calculate both metrics
            yearly_data = category_grants.groupby('start_year').agg({
                'funding_amount': 'sum'
            }).reset_index()
            yearly_data['grant_count'] = category_grants.groupby('start_year').size().values
            
            # Add category name to each row
            yearly_data['category'] = category_name
            
            # Rename columns to match expected format
            yearly_data = yearly_data.rename(columns={'start_year': 'year'})
            
            viz_data.append(yearly_data)
        
        if not viz_data:
            return pd.DataFrame()
        
        # Combine all category data
        combined_data = pd.concat(viz_data, ignore_index=True)
        
        # Ensure we have all required columns with proper names
        # Create both possible column names that might be used
        combined_data['total_funding'] = combined_data['funding_amount']
        
        # Ensure all required columns exist
        required_columns = ['category', 'year', 'grant_count', 'total_funding']
        for col in required_columns:
            if col not in combined_data.columns:
                combined_data[col] = 0
        
        return combined_data
    
    def _map_categories_to_grants_pandas(self, categories_df: pd.DataFrame, keywords_indexed: pd.DataFrame, 
                                        grants_indexed: pd.DataFrame, progress_bar, status_text) -> Dict[str, Dict[str, Any]]:
        """Map categories to grants using pandas operations - OPTIMIZED VERSION"""
        category_grant_mapping = {}
        total_categories = len(categories_df)
        
        # OPTIMIZATION: Create lookup tables upfront
        valid_grant_ids = set(grants_indexed.index)
        keyword_to_grants = {}
        
        # Pre-build keyword to grants mapping
        for keyword, grants_list in keywords_indexed['grants'].items():
            if isinstance(grants_list, list):
                keyword_to_grants[keyword] = set(grants_list)
        
        for idx, (_, category) in enumerate(categories_df.iterrows()):
            # Update progress periodically
            if idx % 100 == 0:
                progress = 30 + int((idx / total_categories) * 40)  # 30-70% range
                progress_bar.progress(progress)
                status_text.text(f"Processing categories... {idx}/{total_categories}")
            
            category_name = category['name']
            category_keywords = category.get('keywords', [])
            
            if not isinstance(category_keywords, list):
                continue
            
            # OPTIMIZATION: Use set operations for fast intersection
            associated_grants = set()
            
            # Get keywords that exist in our keywords DataFrame
            existing_keywords = [kw for kw in category_keywords if kw in keyword_to_grants]
            
            if existing_keywords:
                # OPTIMIZATION: Use set union operations instead of list operations
                for keyword in existing_keywords:
                    associated_grants.update(keyword_to_grants[keyword])
            
            # OPTIMIZATION: Filter grants using set intersection
            valid_grant_ids_for_category = list(associated_grants & valid_grant_ids)
            
            if valid_grant_ids_for_category:
                # Use pandas operations to calculate funding
                grant_subset = grants_indexed.loc[valid_grant_ids_for_category]
                total_funding = grant_subset['funding_amount'].sum()
            else:
                total_funding = 0
            
            category_grant_mapping[category_name] = {
                'grant_details': valid_grant_ids_for_category,
                'grant_count': len(valid_grant_ids_for_category),
                'total_funding': total_funding,
                'keywords': category_keywords,
                'keyword_count': len(category_keywords)
            }
        
        return category_grant_mapping
    
    def _show_trends_statistics(self, config: CategoryTrendsConfig, category_data: Dict[str, Dict[str, Any]]) -> None:
        """Display statistics about the trends data"""
        if not category_data:
            st.warning("No data available for statistics")
            return
        
        st.subheader("Trends Statistics")
        
        total_categories = len(category_data)
        total_grants = sum(data['grant_count'] for data in category_data.values())
        total_funding = sum(data['total_funding'] for data in category_data.values())
        avg_grants_per_category = total_grants / total_categories if total_categories > 0 else 0
        avg_funding_per_category = total_funding / total_categories if total_categories > 0 else 0
        
        # Create metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Categories", total_categories)
        with col2:
            st.metric("Total Grants", f"{total_grants:,}")
        with col3:
            st.metric("Total Funding", f"${total_funding/1e6:.1f}M" if total_funding >= 1e6 else f"${total_funding/1e3:.1f}K")
        with col4:
            ranking_label = "Avg Grants/Category" if config.ranking_metric == "grant_count" else "Avg Funding/Category"
            ranking_value = avg_grants_per_category if config.ranking_metric == "grant_count" else avg_funding_per_category
            display_value = f"{ranking_value:.1f}" if config.ranking_metric == "grant_count" else f"${ranking_value/1e6:.2f}M" if ranking_value >= 1e6 else f"${ranking_value/1e3:.1f}K"
            st.metric(ranking_label, display_value)


class CategoriesPage:
    """Main page class that orchestrates all components"""
    
    def __init__(self):
        self.data_manager = CategoryDataManager()
        self.sidebar_controls = SidebarControls(self.data_manager)
        self.overview_tab = OverviewTab(self.data_manager)
        self.field_analysis_tab = FieldAnalysisTab(self.data_manager)
        self.explorer_tab = CategoryExplorerTab(self.data_manager)
        self.trends_tab = CategoryTrendsTab(self.data_manager)
    
    def setup_page(self):
        """Set up page configuration"""
        st.set_page_config(
            page_title="Category Analysis",
            page_icon="📂",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("📂 Category Analysis")
        st.markdown("""
        Explore and analyze research categories generated from keyword clustering and categorization.
        Use the sidebar to filter categories and the tabs below to explore different aspects of the data.
        """)
    
    def run(self):
        """Main function to run the categories page"""
        self.setup_page()
        
        # Check if data is available
        if self.data_manager.categories_df is None or self.data_manager.categories_df.empty:
            st.error("No category data available. Please run the categorization process first.")
            st.info("Run: `make categorise` to generate categories from keywords, or `make merge` to merge similar categories.")
            return
        
        # Render sidebar controls
        filter_config, trends_config = self.sidebar_controls.render_sidebar()
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔬 Field Analysis", "🔍 Category Explorer", "📈 Category Trends"])
        
        with tab1:
            self.overview_tab.render_tab(filter_config, trends_config)
        
        with tab2:
            self.field_analysis_tab.render_tab(filter_config, trends_config)
        
        with tab3:
            self.explorer_tab.render_tab(filter_config, trends_config)
        
        with tab4:
            self.trends_tab.render_tab(filter_config, trends_config)


def main():
    """Main function to run the categories page"""
    categories_page = CategoriesPage()
    categories_page.run()


if __name__ == "__main__":
    main()
