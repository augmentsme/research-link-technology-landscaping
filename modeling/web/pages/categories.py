"""
Category Analysis Page
Visualize and analyze research categories generated from keyword clustering and categorization.
"""
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Optional

# Add the web directory to Python path to import shared_utils
web_dir = str(Path(__file__).parent.parent)
if web_dir not in sys.path:
    sys.path.insert(0, web_dir)

from shared_utils import load_data  # noqa: E402
from web.sidebar import SidebarControl  # noqa: E402

import config
from visualisation import (  # noqa: E402
    DataExplorer,
    DataExplorerConfig,
    TrendsVisualizer,
    TrendsConfig,
    TrendsDataPreparation,
)
st.set_page_config(
    page_title="Category",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class CategoryFilterConfig:
    """Configuration for category data filtering"""
    field_filter: List[str]
    source_filter: List[str]
    min_keywords: int
    max_keywords: int
    search_term: str
    category_names: List[str]
    start_year_min: Optional[int] = None
    start_year_max: Optional[int] = None
    use_active_grant_period: bool = False


@dataclass
class CategoryTrendsConfig:
    """Configuration for category trends visualization"""
    num_categories: int
    use_cumulative: bool
    ranking_metric: str  # "grant_count" or "total_funding" - for selecting which categories to show
    display_metric: str  # "grant_count" or "total_funding" - for y-axis values
    selection_method: str  # "top_n", "random", or "custom"


class CategoryDataManager:
    """Manages category data loading and filtering operations"""
    
    def __init__(self):
        self.categories_df = None
        self.field_options = None
        self.grants_df = None
        self.keywords_df = None
        self._load_data()
    
    def _load_data(self):
        """Load category data"""
        try:
            self.categories_df = config.Categories.load()
            # Load grants and keywords for source filtering
            self.grants_df = config.Grants.load()
            self.keywords_df = config.Keywords.load()
            
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
        
        # Apply source filtering through grants if source filter is active
        if filter_config.source_filter and self.grants_df is not None and self.keywords_df is not None:
            # Filter grants by source
            filtered_grants = self.grants_df[self.grants_df['source'].isin(filter_config.source_filter)]
            filtered_grant_ids = set(filtered_grants['id'])
            
            # Build keyword-to-grants lookup
            keyword_grants_lookup = {}
            for _, kw_row in self.keywords_df.iterrows():
                if 'grants' in kw_row and kw_row['grants']:
                    # Filter keyword grants to only those matching source filter
                    filtered_kw_grants = [g for g in kw_row['grants'] if g in filtered_grant_ids]
                    if filtered_kw_grants:
                        keyword_grants_lookup[kw_row['name']] = filtered_kw_grants
            
            # Filter categories based on whether their keywords have any filtered grants
            def category_has_valid_grants(keywords_list):
                if not keywords_list:
                    return False
                for keyword in keywords_list:
                    if keyword in keyword_grants_lookup:
                        return True
                return False
            
            filtered_df = filtered_df[filtered_df['keywords'].apply(category_has_valid_grants)]
        
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


class CategoryExplorerTab:
    """Manages the category explorer tab for detailed category browsing"""
    
    def __init__(self, data_manager: CategoryDataManager):
        self.data_manager = data_manager
        self.data_explorer = DataExplorer()
    
    @staticmethod
    def prepare_data(filtered_categories: pd.DataFrame) -> tuple[pd.DataFrame, DataExplorerConfig]:
        """Prepare categories data and configuration for DataExplorer"""
        display_df = filtered_categories[['name', 'field_of_research', 'keyword_count', 'description', 'keywords']].copy()
        display_df = display_df.sort_values('keyword_count', ascending=False)
        
        def format_keywords(keywords):
            if isinstance(keywords, list):
                keywords_str = ", ".join(keywords)
                if len(keywords_str) > 100:
                    return keywords_str[:97] + "..."
                return keywords_str
            else:
                return str(keywords) if keywords else ""
        
        display_df['keywords_formatted'] = display_df['keywords'].apply(format_keywords)
        
        config = DataExplorerConfig(
            title="All Categories",
            description="Search and explore research categories with their keywords and descriptions.",
            search_columns=['name', 'description', 'field_of_research', 'keywords_formatted'],
            search_placeholder="Search by category name, description, field of research, or keywords...",
            search_help="Enter text to search across category names, descriptions, fields, and keywords",
            display_columns=['name', 'field_of_research', 'keyword_count', 'keywords_formatted', 'description'],
            column_formatters={
                'description': lambda x: (x[:150] + "...") if len(str(x)) > 150 else str(x),
                'keywords_formatted': lambda x: x
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
        
        return display_df, config
    
    def render_tab(self, filter_config: CategoryFilterConfig):
        """Render the category explorer tab"""
        st.markdown("# Category Explorer")
        
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
        
        st.subheader(f"{category_row['name']}")
        
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
        # Prepare data and config
        display_df, config = self.prepare_data(filtered_data)
        
        # Render the explorer
        self.data_explorer.render_explorer(display_df, config)


class CategoryTrendsTab:
    """Manages the category trends over time tab"""
    
    def __init__(self, data_manager: CategoryDataManager):
        self.data_manager = data_manager
    
    def render_tab(self, filter_config: CategoryFilterConfig, trends_config: CategoryTrendsConfig):
        """Render the category trends tab"""
        st.markdown("# Category Trends Over Time")
        
        filtered_data = self.data_manager.get_filtered_data(filter_config)
        
        if filtered_data.empty:
            st.warning("No categories found with current filters.")
            return
        
        # Validate custom selection
        if trends_config.selection_method == "custom" and not filter_config.category_names:
            st.error("Please select at least one category when using custom category selection.")
            return
        
        # Load additional data needed for trends
        _, grants_df, _ = load_data()
        
        if grants_df is None:
            st.error("Unable to load grants data needed for trends analysis.")
            return
        
        # Show data info
        st.info(f"Processing {len(filtered_data)} categories. "
               f"Ranking by: {'Grant Count' if trends_config.ranking_metric == 'grant_count' else 'Total Funding Amount'}, "
               f"Displaying: {'Grant Count' if trends_config.display_metric == 'grant_count' else 'Funding Amount'}")
        
        with st.spinner("Creating category trends visualization..."):
            # Determine selected categories based on selection method
            selected_categories = None
            if trends_config.selection_method == "custom" and filter_config.category_names:
                selected_categories = filter_config.category_names
            
            # Prepare data using the helper from TrendsDataPreparation
            viz_data = TrendsDataPreparation.from_category_grants(
                filtered_data,
                grants_df,
                selected_categories,
                use_active_period=filter_config.use_active_grant_period,
                year_min=filter_config.start_year_min,
                year_max=filter_config.start_year_max,
            )
            
            if viz_data.empty:
                st.warning("No data available for visualization.")
                return
            
            # Configure visualization
            ranking_text = "Grant Count" if trends_config.ranking_metric == "grant_count" else "Total Funding"
            display_text = "Grant Count" if trends_config.display_metric == "grant_count" else "Funding Amount"
            cumulative_text = "Cumulative" if trends_config.use_cumulative else "Yearly"
            
            # Create appropriate title based on selection method
            if trends_config.selection_method == "custom":
                title = f"{cumulative_text} Category Trends Over Time - Custom Selection ({len(selected_categories)} categories)"
                max_entities_to_show = len(selected_categories)
            else:
                title = f"{cumulative_text} Category Trends Over Time (Top {trends_config.num_categories} by {ranking_text})"
                max_entities_to_show = trends_config.num_categories
            
            viz_config = TrendsConfig(
                entity_col='category',
                time_col='year',
                value_col=trends_config.display_metric,
                ranking_col=trends_config.ranking_metric,
                max_entities=max_entities_to_show,
                aggregation='sum',
                use_cumulative=trends_config.use_cumulative,
                chart_type='line',
                title=title,
                x_label='Year',
                y_label=f'{"Cumulative" if trends_config.use_cumulative else "Yearly"} {display_text}',
                height=600
            )
            
            # Create visualization
            visualizer = TrendsVisualizer()
            fig = visualizer.create_plot(viz_data, viz_config)
            
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, key="category_trends_plot")
                self._show_statistics(filtered_data, viz_data)
                self._show_debug_data(filtered_data, viz_data)
            else:
                st.warning("Unable to create trends visualization.")
    
    def _show_statistics(self, filtered_data: pd.DataFrame, viz_data: pd.DataFrame):
        """Display statistics about the filtered data"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Categories", len(filtered_data))
        with col2:
            st.metric("Categories in Chart", viz_data['category'].nunique())
        with col3:
            year_range = f"{viz_data['year'].min()}-{viz_data['year'].max()}"
            st.metric("Year Range", year_range)
    
    def _show_debug_data(self, filtered_data: pd.DataFrame, viz_data: pd.DataFrame):
        """Show debug data in an expander"""
        with st.expander("Debug: View Underlying Data", expanded=False):
            st.subheader("Filtered Categories DataFrame")
            st.write(f"**Shape:** {filtered_data.shape}")
            st.dataframe(filtered_data, use_container_width=True)
            
            st.subheader("Category Trends Data (Chart Data)")
            st.write(f"**Shape:** {viz_data.shape}")
            st.dataframe(viz_data, use_container_width=True)


class CategoriesPage:
    """Main page class that orchestrates all components"""
    
    def __init__(self):
        self.data_manager = CategoryDataManager()
        self.explorer_tab = CategoryExplorerTab(self.data_manager)
        self.trends_tab = CategoryTrendsTab(self.data_manager)

    
    def run(self):

        
        # Check if data is available
        if self.data_manager.categories_df is None or self.data_manager.categories_df.empty:
            st.error("No category data available. Please run the categorization process first.")
            st.info("Run: `make categorise` to generate categories from keywords, or `make merge` to merge similar categories.")
            return
        
        # Load grants data for unified sidebar
        _, grants_df, _ = load_data()
        
        # Create unified sidebar controls
        sidebar = SidebarControl(
            page_type="categories",
            grants_df=grants_df,
            categories_df=self.data_manager.categories_df
        )
        
        # Get unified configurations from sidebar
        unified_filter, unified_display = sidebar.render_sidebar()
        
        # Adapt to legacy format for existing code
        filter_config = CategoryFilterConfig(
            field_filter=unified_filter.field_filter,
            source_filter=unified_filter.source_filter,
            min_keywords=unified_filter.min_count,
            max_keywords=unified_filter.max_count,
            search_term=unified_filter.search_term,
            category_names=unified_display.custom_entities if unified_display.selection_method == "custom" else [],
            start_year_min=unified_filter.start_year_min,
            start_year_max=unified_filter.start_year_max,
            use_active_grant_period=unified_filter.use_active_grant_period
        )
        
        trends_config = CategoryTrendsConfig(
            num_categories=unified_display.num_entities,
            use_cumulative=unified_display.use_cumulative,
            ranking_metric="grant_count" if unified_display.ranking_metric == "count" else "total_funding",
            display_metric="grant_count" if unified_display.display_metric == "count" else "total_funding",
            selection_method=unified_display.selection_method
        )
        
        # Create tabs
        tab1, tab2 = st.tabs(["Category Trends", "Category Explorer"])
        
        with tab1:
            self.trends_tab.render_tab(filter_config, trends_config)
            
        
        with tab2:
            self.explorer_tab.render_tab(filter_config)
            


def main():
    """Main function to run the categories page"""
    categories_page = CategoriesPage()
    categories_page.run()


if __name__ == "__main__":
    main()
