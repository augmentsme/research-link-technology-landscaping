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
from typing import List, Optional, Dict, Any, Tuple

# Add the web directory to Python path to import shared_utils
web_dir = str(Path(__file__).parent.parent)
if web_dir not in sys.path:
    sys.path.insert(0, web_dir)

from shared_utils import load_data

import config
from visualisation import DataExplorer, DataExplorerConfig
from trends_visualizer import TrendsVisualizer, TrendsConfig, TrendsDataPreparation
from data_explorer_helper import DataExplorerPreparation


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
            st.subheader("� Category Analysis Settings")
            
            filter_config = self._render_filtering_options()
            trends_config = self._render_trends_settings()
            
            # Show active filters
            self._show_active_filters(filter_config)
            
            st.markdown("---")
            
            return filter_config, trends_config
    
    def _render_filtering_options(self) -> CategoryFilterConfig:
        """Render filtering options section"""
        with st.expander("🔍 Filtering Options", expanded=True):
            st.markdown("**Category Filters**")
            
            # Field of research filter
            field_filter = st.multiselect(
                "Filter by Field of Research",
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
            
            st.markdown("**Category Size Filter:**")
            min_keywords, max_keywords = st.slider(
                "Number of keywords in category",
                min_value=min_possible,
                max_value=max_possible,
                value=(min_possible, max_possible),
                help="Filter categories by number of keywords they contain"
            )
            
            # Search filter
            st.markdown("**Search:**")
            search_term = st.text_input(
                "Search categories",
                placeholder="Search in category names or descriptions...",
                help="Filter categories by text search in names or descriptions"
            )
        
        return CategoryFilterConfig(
            field_filter=field_filter,
            min_keywords=min_keywords,
            max_keywords=max_keywords,
            search_term=search_term,
            category_names=[]
        )
    
    def _render_trends_settings(self) -> CategoryTrendsConfig:
        """Render trends visualization settings"""
        with st.expander("📈 Trends Settings", expanded=True):
            st.markdown("**Display Options:**")
            
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
            
            st.markdown("**Ranking & Display Metrics:**")
            
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
        
        return CategoryTrendsConfig(
            num_categories=num_categories,
            use_cumulative=use_cumulative,
            ranking_metric=ranking_metric,
            display_metric=display_metric
        )
    
    def _show_active_filters(self, filter_config: CategoryFilterConfig) -> None:
        """Display active filters summary"""
        active_filters = []
        
        if filter_config.field_filter:
            active_filters.append(f"🔬 Fields: {len(filter_config.field_filter)}")
        
        if filter_config.search_term:
            active_filters.append(f"🔍 Search: '{filter_config.search_term}'")
        
        # Check if keyword range is not at max
        if not self.data_manager.categories_df.empty:
            min_possible = int(self.data_manager.categories_df['keyword_count'].min())
            max_possible = int(self.data_manager.categories_df['keyword_count'].max())
            if filter_config.min_keywords != min_possible or filter_config.max_keywords != max_possible:
                active_filters.append(
                    f"📊 Keywords: {filter_config.min_keywords}-{filter_config.max_keywords}"
                )
        
        if active_filters:
            for filter_desc in active_filters:
                st.sidebar.markdown(f"- {filter_desc}")
        else:
            st.sidebar.markdown("*No filters active*")


class CategoryExplorerTab:
    """Manages the category explorer tab for detailed category browsing"""
    
    def __init__(self, data_manager: CategoryDataManager):
        self.data_manager = data_manager
        self.data_explorer = DataExplorer()
    
    def render_tab(self, filter_config: CategoryFilterConfig):
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
        # Prepare data and config using helper
        display_df, config = DataExplorerPreparation.prepare_categories_data(filtered_data)
        
        # Render the explorer
        self.data_explorer.render_explorer(display_df, config)


class CategoryTrendsTab:
    """Manages the category trends over time tab"""
    
    def __init__(self, data_manager: CategoryDataManager):
        self.data_manager = data_manager
    
    def render_tab(self, filter_config: CategoryFilterConfig, trends_config: CategoryTrendsConfig):
        """Render the category trends tab"""
        st.markdown("### Category Trends Over Time")
        st.markdown("Analyze how research categories evolve over time based on their associated grants.")
        
        filtered_data = self.data_manager.get_filtered_data(filter_config)
        
        if filtered_data.empty:
            st.warning("No categories found with current filters.")
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
            # Prepare data using the helper from TrendsDataPreparation
            viz_data = TrendsDataPreparation.from_category_grants(
                filtered_data,
                grants_df
            )
            
            if viz_data.empty:
                st.warning("No data available for visualization.")
                return
            
            # Configure visualization
            ranking_text = "Grant Count" if trends_config.ranking_metric == "grant_count" else "Total Funding"
            display_text = "Grant Count" if trends_config.display_metric == "grant_count" else "Funding Amount"
            cumulative_text = "Cumulative" if trends_config.use_cumulative else "Yearly"
            
            viz_config = TrendsConfig(
                entity_col='category',
                time_col='year',
                value_col=trends_config.display_metric,
                ranking_col=trends_config.ranking_metric,
                max_entities=trends_config.num_categories,
                aggregation='sum',
                use_cumulative=trends_config.use_cumulative,
                chart_type='line',
                title=f"{cumulative_text} Category Trends Over Time (Top {trends_config.num_categories} by {ranking_text})",
                x_label='Year',
                y_label=f'{"Cumulative" if trends_config.use_cumulative else "Yearly"} {display_text}',
                height=600
            )
            
            # Create visualization
            visualizer = TrendsVisualizer()
            fig = visualizer.create_plot(viz_data, viz_config)
            
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, key="category_trends_plot")
            else:
                st.warning("Unable to create trends visualization.")
    


class CategoriesPage:
    """Main page class that orchestrates all components"""
    
    def __init__(self):
        self.data_manager = CategoryDataManager()
        self.sidebar_controls = SidebarControls(self.data_manager)
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
        tab1, tab2 = st.tabs([" Category Explorer", "📈 Category Trends"])
        
        with tab1:
            self.explorer_tab.render_tab(filter_config)
        
        with tab2:
            self.trends_tab.render_tab(filter_config, trends_config)


def main():
    """Main function to run the categories page"""
    categories_page = CategoriesPage()
    categories_page.run()


if __name__ == "__main__":
    main()
