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


@dataclass
class CategoryFilterConfig:
    """Configuration for category data filtering"""
    field_filter: List[str]
    min_keywords: int
    max_keywords: int
    search_term: str
    category_names: List[str]


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
    
    def render_sidebar(self) -> CategoryFilterConfig:
        """Render sidebar controls and return filter configuration"""
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
            
            # Show active filters
            self._show_active_filters(field_filter, min_keywords, max_keywords, search_term)
        
        return CategoryFilterConfig(
            field_filter=field_filter,
            min_keywords=min_keywords,
            max_keywords=max_keywords,
            search_term=search_term,
            category_names=[]
        )
    
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
    
    def render_tab(self, filter_config: CategoryFilterConfig):
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
    
    def render_tab(self, filter_config: CategoryFilterConfig):
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
    
    def render_tab(self, filter_config: CategoryFilterConfig):
        """Render the category explorer tab"""
        st.markdown("### Category Explorer")
        st.markdown("Browse and explore individual categories in detail.")
        
        filtered_data = self.data_manager.get_filtered_data(filter_config)
        
        if filtered_data.empty:
            st.warning("No categories found with current filters.")
            return
        
        # Category selection
        selected_category = self._render_category_selector(filtered_data)
        
        if selected_category:
            self._show_category_details(filtered_data, selected_category)
        
        # Category list
        self._show_category_table(filtered_data)
    
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
    
    def _show_category_table(self, filtered_data: pd.DataFrame):
        """Show table of all categories"""
        st.subheader("All Categories")
        
        # Prepare display dataframe
        display_df = filtered_data[['name', 'field_of_research', 'keyword_count', 'description', 'keywords']].copy()
        display_df = display_df.sort_values('keyword_count', ascending=False)
        
        # Format keywords for display
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
        
        # Select and rename columns for display
        final_df = display_df[['name', 'field_of_research', 'keyword_count', 'keywords_formatted', 'description']].copy()
        final_df.columns = ['Category Name', 'Field of Research', 'Keywords Count', 'Keywords', 'Description']
        
        st.dataframe(
            final_df,
            use_container_width=True,
            height=400
        )


class CategoryTrendsTab:
    """Manages the category trends over time tab"""
    
    def __init__(self, data_manager: CategoryDataManager):
        self.data_manager = data_manager
    
    def render_tab(self, filter_config: CategoryFilterConfig):
        """Render the category trends tab"""
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
        
        # Category selection controls
        self._render_trend_controls(filtered_data, keywords_df, grants_df)
    
    def _render_trend_controls(self, filtered_data: pd.DataFrame, keywords_df: pd.DataFrame, grants_df: pd.DataFrame):
        """Render controls for category trends"""
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Select categories to display
            max_categories = min(15, len(filtered_data))
            num_categories = st.slider(
                "Number of categories to display",
                min_value=1,
                max_value=max_categories,
                value=min(5, max_categories),  # Reduced default for performance
                help="Select how many top categories to show in the trends plot"
            )
        
        with col2:
            # Display options
            use_cumulative = st.checkbox(
                "Show cumulative trends",
                value=True,
                help="Show cumulative category occurrences over time instead of yearly counts"
            )
        
        # Show data size warning
        st.info(f"Processing {len(filtered_data)} categories with {len(keywords_df)} keywords. Large datasets may take a moment to process.")
        
        # Generate trends visualization
        if st.button("Generate Category Trends", type="primary"):
            with st.spinner("Creating category trends visualization..."):
                # Add progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    fig = self._create_category_trends_plot(
                        filtered_data, keywords_df, grants_df, num_categories, use_cumulative, progress_bar, status_text
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("Visualization complete!")
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="category_trends_plot")
                        self._show_trends_statistics(filtered_data, keywords_df, grants_df, num_categories)
                    else:
                        st.warning("Unable to create trends visualization. No valid data found.")
                        
                except Exception as e:
                    st.error(f"Error creating visualization: {str(e)}")
                finally:
                    # Clean up progress indicators
                    progress_bar.empty()
                    status_text.empty()
    
    def _create_category_trends_plot(self, categories_df: pd.DataFrame, keywords_df: pd.DataFrame, 
                                   grants_df: pd.DataFrame, num_categories: int, use_cumulative: bool,
                                   progress_bar, status_text) -> Optional[go.Figure]:
        """Create the category trends plot with optimized performance"""
        
        # Step 1: Create keyword lookup dictionary for fast access
        status_text.text("Building keyword index...")
        progress_bar.progress(10)
        
        keyword_to_grants = {}
        for _, keyword_row in keywords_df.iterrows():
            keyword_name = keyword_row['name']
            grants_list = keyword_row.get('grants', [])
            if isinstance(grants_list, list):
                keyword_to_grants[keyword_name] = grants_list
        
        # Step 2: Create grants lookup dictionary for fast temporal access
        status_text.text("Building grants index...")
        progress_bar.progress(20)
        
        grants_lookup = {}
        for _, grant_row in grants_df.iterrows():
            grant_id = grant_row['id']
            start_year = grant_row.get('start_year')
            if pd.notna(start_year):
                grants_lookup[grant_id] = int(start_year)
        
        # Step 3: Process categories efficiently
        status_text.text("Processing categories...")
        progress_bar.progress(30)
        
        category_grant_data = self._map_categories_to_grants_optimized(
            categories_df, keyword_to_grants, grants_lookup, progress_bar, status_text
        )
        
        if not category_grant_data:
            return None
        
        # Step 4: Select top categories
        status_text.text("Selecting top categories...")
        progress_bar.progress(70)
        
        top_categories = sorted(category_grant_data.items(), 
                              key=lambda x: len(x[1]['grants']), 
                              reverse=True)[:num_categories]
        
        # Step 5: Create time series data
        status_text.text("Creating time series...")
        progress_bar.progress(80)
        
        time_series_data = []
        for category_name, category_info in top_categories:
            yearly_data = self._create_yearly_data_for_category_optimized(
                category_info['grants'], grants_lookup
            )
            
            for year, count in yearly_data.items():
                time_series_data.append({
                    'category': category_name,
                    'year': year,
                    'count': count
                })
        
        if not time_series_data:
            return None
        
        # Step 6: Create visualization
        status_text.text("Creating visualization...")
        progress_bar.progress(90)
        
        # Convert to DataFrame and create matrix
        df = pd.DataFrame(time_series_data)
        
        # Create occurrence matrix
        years = sorted(df['year'].unique())
        categories = [cat[0] for cat in top_categories]
        
        fig = go.Figure()
        
        # Create traces for each category
        for category in categories:
            category_data = df[df['category'] == category]
            
            # Create full year range with zeros for missing years
            year_counts = dict(zip(category_data['year'], category_data['count']))
            full_counts = [year_counts.get(year, 0) for year in years]
            
            if use_cumulative:
                full_counts = np.cumsum(full_counts)
            
            fig.add_trace(go.Scatter(
                x=years,
                y=full_counts,
                mode='lines+markers',
                name=category,
                line=dict(width=2),
                marker=dict(size=6),
                hovertemplate=f'<b>{category}</b><br>' +
                            'Year: %{x}<br>' +
                            ('Cumulative Grants: %{y}' if use_cumulative else 'Yearly Grants: %{y}') +
                            '<extra></extra>'
            ))
        
        # Update layout
        y_title = 'Cumulative Grant Count' if use_cumulative else 'Yearly Grant Count'
        title = f"{'Cumulative' if use_cumulative else 'Yearly'} Category Trends Over Time (Top {num_categories} Categories)"
        
        fig.update_layout(
            title=title,
            xaxis_title='Year',
            yaxis_title=y_title,
            legend_title='Category',
            height=600,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def _map_categories_to_grants_optimized(self, categories_df: pd.DataFrame, keyword_to_grants: Dict[str, List[str]], 
                                          grants_lookup: Dict[str, int], progress_bar, status_text) -> Dict[str, Dict[str, Any]]:
        """Optimized mapping of categories to grants through keywords"""
        category_grant_mapping = {}
        total_categories = len(categories_df)
        
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
            
            # Find grants associated with these keywords using the pre-built lookup
            associated_grants = set()
            
            for keyword_name in category_keywords:
                if keyword_name in keyword_to_grants:
                    grants_list = keyword_to_grants[keyword_name]
                    associated_grants.update(grants_list)
            
            category_grant_mapping[category_name] = {
                'grants': list(associated_grants),
                'keywords': category_keywords,
                'keyword_count': len(category_keywords)
            }
        
        return category_grant_mapping
    
    def _create_yearly_data_for_category_optimized(self, grant_ids: List[str], grants_lookup: Dict[str, int]) -> Dict[int, int]:
        """Optimized creation of yearly occurrence data for a category"""
        yearly_counts = {}
        
        for grant_id in grant_ids:
            if grant_id in grants_lookup:
                year = grants_lookup[grant_id]
                yearly_counts[year] = yearly_counts.get(year, 0) + 1
        
        return yearly_counts
    
    def _show_trends_statistics(self, categories_df: pd.DataFrame, keywords_df: pd.DataFrame, 
                              grants_df: pd.DataFrame, num_categories: int):
        """Display statistics about the trends data"""
        st.subheader("Trends Statistics")
        
        # For statistics, we'll use a smaller sample to avoid performance issues
        sample_size = min(100, len(categories_df))
        sample_categories = categories_df.head(sample_size)
        
        # Create optimized lookups for the sample
        keyword_to_grants = {}
        for _, keyword_row in keywords_df.iterrows():
            keyword_name = keyword_row['name']
            grants_list = keyword_row.get('grants', [])
            if isinstance(grants_list, list):
                keyword_to_grants[keyword_name] = grants_list
        
        grants_lookup = {}
        for _, grant_row in grants_df.iterrows():
            grant_id = grant_row['id']
            start_year = grant_row.get('start_year')
            if pd.notna(start_year):
                grants_lookup[grant_id] = int(start_year)
        
        # Calculate statistics for sample
        category_grant_data = {}
        for _, category in sample_categories.iterrows():
            category_name = category['name']
            category_keywords = category.get('keywords', [])
            
            if isinstance(category_keywords, list):
                associated_grants = set()
                for keyword_name in category_keywords:
                    if keyword_name in keyword_to_grants:
                        associated_grants.update(keyword_to_grants[keyword_name])
                
                category_grant_data[category_name] = {
                    'grants': list(associated_grants),
                    'keywords': category_keywords
                }
        
        if not category_grant_data:
            st.warning("No category-grant mappings found in sample.")
            return
        
        total_grants_mapped = sum(len(data['grants']) for data in category_grant_data.values())
        categories_with_grants = sum(1 for data in category_grant_data.values() if data['grants'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Categories Displayed", num_categories)
        
        with col2:
            st.metric("Sample Categories with Grants", f"{categories_with_grants}/{sample_size}")
        
        with col3:
            st.metric("Sample Grant Mappings", total_grants_mapped)
        
        with col4:
            if category_grant_data:
                top_category = max(category_grant_data.items(), key=lambda x: len(x[1]['grants']))
                st.metric("Most Active Category", len(top_category[1]['grants']))
        
        if sample_size < len(categories_df):
            st.info(f"Statistics based on sample of {sample_size} categories (out of {len(categories_df)} total)")


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
            st.info("Run: `uv run python categorise.py keywords` to generate categories from keywords.")
            return
        
        # Render sidebar controls
        filter_config = self.sidebar_controls.render_sidebar()
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔬 Field Analysis", "🔍 Category Explorer", "📈 Category Trends"])
        
        with tab1:
            self.overview_tab.render_tab(filter_config)
        
        with tab2:
            self.field_analysis_tab.render_tab(filter_config)
        
        with tab3:
            self.explorer_tab.render_tab(filter_config)
        
        with tab4:
            self.trends_tab.render_tab(filter_config)


def main():
    """Main function to run the categories page"""
    categories_page = CategoriesPage()
    categories_page.run()


if __name__ == "__main__":
    main()
