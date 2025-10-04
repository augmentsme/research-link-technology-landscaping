"""
Research Landscape Visualizations - Refactored

This module contains visualization functions for the research link technology 
landscaping analysis, including treemap visualizations of research categories 
and keywords, and keyword trends over time.

Refactored to use modular classes and methods for better maintainability.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import Counter
from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import streamlit as st

import config


@dataclass
class FilterConfig:
    """Configuration for data filtering"""
    funder_filter: Optional[List[str]] = None
    source_filter: Optional[List[str]] = None
    keyword_type_filter: Optional[List[str]] = None
    field_filter: Optional[List[str]] = None
    min_count: int = 10
    max_count: int = 999999


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings"""
    title: str = "Cumulative keyword occurrences over time — Top keywords"
    height: int = 600
    show_baseline: bool = False
    use_cumulative: bool = True
    top_n: int = 20
    custom_keywords: Optional[List[str]] = None


class DataFilterManager:
    """Manages data filtering operations for visualizations"""
    
    @staticmethod
    def apply_grant_filters(grants_df: pd.DataFrame, filter_config: FilterConfig) -> pd.DataFrame:
        """Apply filters to grants dataframe"""
        filtered_grants = grants_df.copy()
        
        if filter_config.funder_filter:
            filtered_grants = filtered_grants[filtered_grants['funder'].isin(filter_config.funder_filter)]
        
        if filter_config.source_filter:
            filtered_grants = filtered_grants[filtered_grants['source'].isin(filter_config.source_filter)]
        
        if filter_config.field_filter:
            division_codes = DataFilterManager._field_to_division_codes(filter_config.field_filter)
            mask = filtered_grants['for_primary'].apply(
                lambda x: DataFilterManager._has_research_field(x, division_codes)
            )
            filtered_grants = filtered_grants[mask]
        
        return filtered_grants
    
    @staticmethod
    def apply_keyword_filters(keywords_df: pd.DataFrame, filter_config: FilterConfig) -> pd.DataFrame:
        """Apply filters to keywords dataframe"""
        filtered_keywords = keywords_df.copy()
        
        if filter_config.keyword_type_filter:
            filtered_keywords = filtered_keywords[filtered_keywords['type'].isin(filter_config.keyword_type_filter)]
        
        return filtered_keywords
    
    @staticmethod
    def _field_to_division_codes(field_names: List[str]) -> List[str]:
        """Convert field names to division codes for filtering"""
        field_to_division = {
            'AGRICULTURAL_VETERINARY_FOOD_SCIENCES': '30',
            'BIOLOGICAL_SCIENCES': '31',
            'BIOMEDICAL_CLINICAL_SCIENCES': '32',
            'BUILT_ENVIRONMENT_DESIGN': '33',
            'CHEMICAL_SCIENCES': '34',
            'COMMERCE_MANAGEMENT_TOURISM_SERVICES': '35',
            'CREATIVE_ARTS_WRITING': '36',
            'EARTH_SCIENCES': '37',
            'ECONOMICS': '38',
            'EDUCATION': '39',
            'ENGINEERING': '40',
            'ENVIRONMENTAL_SCIENCES': '41',
            'HEALTH_SCIENCES': '42',
            'HISTORY_HERITAGE_ARCHAEOLOGY': '43',
            'HUMAN_SOCIETY': '44',
            'INDIGENOUS_STUDIES': '45',
            'INFORMATION_COMPUTING_SCIENCES': '46',
            'LANGUAGE_COMMUNICATION_CULTURE': '47',
            'LAW_LEGAL_STUDIES': '48',
            'MATHEMATICAL_SCIENCES': '49',
            'PHILOSOPHY_RELIGIOUS_STUDIES': '50',
            'PHYSICAL_SCIENCES': '51',
            'PSYCHOLOGY': '52'
        }
        return [field_to_division[field] for field in field_names if field in field_to_division]
    
    @staticmethod
    def _has_research_field(for_primary_val: float, division_codes: List[str]) -> bool:
        """Check if primary FOR code matches any of the division codes"""
        if pd.isna(for_primary_val):
            return False
        division_code = str(int(for_primary_val))[:2]
        return division_code in division_codes


class KeywordTrendsDataProcessor:
    """Processes keyword trends data for visualization"""
    
    def __init__(self, keywords_df: pd.DataFrame, grants_df: pd.DataFrame):
        self.keywords_df = keywords_df
        self.grants_df = grants_df
    
    def create_trends_data(self, filter_config: FilterConfig) -> pd.DataFrame:
        """Create keyword trends data from keywords and grants dataframes"""
        # Prepare grants data
        grants_df = self._prepare_grants_data()
        
        # Apply filters
        filtered_grants = DataFilterManager.apply_grant_filters(grants_df, filter_config)
        filtered_keywords = DataFilterManager.apply_keyword_filters(self.keywords_df, filter_config)
        
        # Filter keywords to only include those with grants in the filtered set
        filtered_keywords = self._filter_keywords_by_grants(filtered_keywords, filtered_grants, filter_config.min_count, filter_config.max_count)
        
        # Create keyword-year pairs
        return self._create_keyword_year_pairs(filtered_keywords, filtered_grants)
    
    def _prepare_grants_data(self) -> pd.DataFrame:
        """Prepare grants data by setting index if needed"""
        grants_df = self.grants_df.copy()
        if 'id' in grants_df.columns:
            grants_df = grants_df.set_index('id')
        return grants_df
    
    def _filter_keywords_by_grants(self, keywords_df: pd.DataFrame, grants_df: pd.DataFrame, min_count: int, max_count: int = 999999) -> pd.DataFrame:
        """Filter keywords to only include those with grants in the filtered set"""
        keywords_df['filtered_grants'] = keywords_df['grants'].apply(
            lambda x: [grant_id for grant_id in x if grant_id in grants_df.index]
        )
        keywords_df['filtered_count'] = keywords_df['filtered_grants'].map(len)
        
        return keywords_df[(keywords_df['filtered_count'] > min_count) & (keywords_df['filtered_count'] <= max_count)].copy()
    
    def _create_keyword_year_pairs(self, keywords_df: pd.DataFrame, grants_df: pd.DataFrame) -> pd.DataFrame:
        """Create keyword-year pairs from filtered data"""
        # Create start_year mapping for each keyword using filtered grants
        if 'start_year' not in keywords_df.columns:
            keywords_df['start_year'] = keywords_df['filtered_grants'].map(
                lambda x: [grants_df.loc[i, "start_year"] for i in x 
                          if i in grants_df.index and not np.isnan(grants_df.loc[i, "start_year"])]
            )
        
        # Explode the keywords to create name-year pairs
        kw_years_list = []
        for idx, row in keywords_df.iterrows():
            name = row['name'] if 'name' in row else row.name
            for year in row['start_year']:
                kw_years_list.append({'name': name, 'start_year': int(year)})
        
        return pd.DataFrame(kw_years_list)


class KeywordSelector:
    """Handles keyword selection for visualization"""
    
    @staticmethod
    def select_keywords_for_display(kw_years: pd.DataFrame, viz_config: VisualizationConfig) -> pd.DataFrame:
        """Select keywords for individual line display"""
        if viz_config.custom_keywords:
            return KeywordSelector._select_custom_keywords(kw_years, viz_config.custom_keywords)
        elif viz_config.top_n > 0:
            return KeywordSelector._select_top_n_keywords(kw_years, viz_config.top_n)
        else:
            return pd.DataFrame()
    
    @staticmethod
    def _select_custom_keywords(kw_years: pd.DataFrame, custom_keywords: List[str]) -> pd.DataFrame:
        """Select user-specified custom keywords"""
        available_keywords = kw_years['name'].value_counts()
        valid_custom_keywords = [kw for kw in custom_keywords if kw in available_keywords]
        
        if valid_custom_keywords:
            return (
                kw_years[kw_years['name'].isin(valid_custom_keywords)]
                .groupby(['name', 'start_year'])
                .size()
                .reset_index(name='occurrences')
                .sort_values(['name', 'start_year'])
            )
        return pd.DataFrame()
    
    @staticmethod
    def _select_top_n_keywords(kw_years: pd.DataFrame, top_n: int) -> pd.DataFrame:
        """Select top N keywords by occurrence count"""
        top_names = kw_years['name'].value_counts().head(top_n).index.tolist()
        
        return (
            kw_years[kw_years['name'].isin(top_names)]
            .groupby(['name', 'start_year'])
            .size()
            .reset_index(name='occurrences')
            .sort_values(['name', 'start_year'])
        )


class BaselineCalculator:
    """Calculates baseline data for visualization"""
    
    @staticmethod
    def calculate_baseline(kw_years: pd.DataFrame, grants_df: pd.DataFrame, 
                          viz_config: VisualizationConfig, year_range: Tuple[int, int]) -> Optional[pd.Series]:
        """Calculate baseline: cumulative average keywords per grant per year"""
        if not viz_config.show_baseline:
            return None
        
        # Count total keywords generated per year across all grants
        yearly_keyword_counts = (
            kw_years.groupby('start_year')
            .size()
            .reset_index(name='total_keywords')
        )
        
        # Count number of grants per year
        yearly_grant_counts = (
            grants_df.groupby('start_year')
            .size()
            .reset_index(name='total_grants')
        )
        
        # Merge and calculate keywords per grant per year
        yearly_baseline = yearly_keyword_counts.merge(
            yearly_grant_counts, 
            on='start_year', 
            how='inner'
        )
        yearly_baseline['keywords_per_grant'] = (
            yearly_baseline['total_keywords'] / yearly_baseline['total_grants']
        )
        
        # Create baseline series for the specified year range
        baseline_years = np.arange(year_range[0], year_range[1] + 1)
        baseline_series = (
            yearly_baseline.set_index('start_year')['keywords_per_grant']
            .reindex(baseline_years, fill_value=0)
        )
        
        if viz_config.use_cumulative:
            baseline_series = baseline_series.cumsum()
        
        return baseline_series


class MatrixProcessor:
    """Processes data matrices for visualization"""
    
    @staticmethod
    def create_occurrence_matrix(df: pd.DataFrame, use_cumulative: bool) -> pd.DataFrame:
        """Create occurrence matrix from dataframe"""
        if df.empty:
            return pd.DataFrame()
        
        years = np.arange(df['start_year'].min(), df['start_year'].max() + 1)
        occ_matrix = (
            df.groupby(['start_year', 'name'])['occurrences']
            .sum()
            .unstack(fill_value=0)
            .reindex(index=years, fill_value=0)
        )
        
        if use_cumulative:
            occ_matrix = occ_matrix.cumsum()
        
        return occ_matrix
    
    @staticmethod
    def sort_matrix_columns(matrix: pd.DataFrame, use_cumulative: bool) -> List[str]:
        """Sort matrix columns for legend ordering - top performers first"""
        if matrix.empty:
            return []
        
        if use_cumulative:
            # For cumulative data, sort by final cumulative value (last row)
            # Use descending=True so highest values appear first in legend (top)
            return matrix.iloc[-1].sort_values(ascending=False).index.tolist()
        else:
            # For yearly data, sort by total sum across all years
            # Use descending=True so highest values appear first in legend (top)
            return matrix.sum(axis=0).sort_values(ascending=False).index.tolist()


class PlotBuilder:
    """Builds plotly figures for visualization"""
    
    def __init__(self, viz_config: VisualizationConfig):
        self.viz_config = viz_config
    
    def create_keyword_trends_figure(self, occ_matrix_display: pd.DataFrame, 
                                   baseline_data: Optional[pd.Series]) -> go.Figure:
        """Create the main keyword trends figure"""
        fig = go.Figure()
        
        # Add individual keyword traces
        self._add_keyword_traces(fig, occ_matrix_display)
        
        # Add baseline
        self._add_baseline(fig, baseline_data)
        
        # Update layout
        self._update_layout(fig)
        
        return fig
    
    def _add_keyword_traces(self, fig: go.Figure, occ_matrix_display: pd.DataFrame):
        """Add individual keyword traces to the figure"""
        if occ_matrix_display.empty:
            return
        
        # Sort columns for legend ordering (highest values first)
        sorted_columns = MatrixProcessor.sort_matrix_columns(occ_matrix_display, self.viz_config.use_cumulative)
        
        # Add traces in order - highest performing keywords will appear at top of legend
        for name in sorted_columns:
            fig.add_trace(go.Scatter(
                x=occ_matrix_display.index,
                y=occ_matrix_display[name],
                mode='lines+markers',
                name=name,
                opacity=0.6,
                line=dict(width=1.5),
                marker=dict(size=4)
            ))
    
    def _add_baseline(self, fig: go.Figure, baseline_data: Optional[pd.Series]):
        """Add baseline to the figure"""
        if baseline_data is None or not self.viz_config.show_baseline:
            return
        
        fig.add_trace(go.Scatter(
            x=baseline_data.index,
            y=baseline_data.values,
            mode='lines+markers',
            name='Baseline (Keywords/Grant/Year)',
            line=dict(width=2, color='red', dash='dash'),
            marker=dict(size=4, color='red'),
            opacity=0.8
        ))
    
    def _update_layout(self, fig: go.Figure):
        """Update figure layout"""
        y_axis_title = 'Cumulative occurrences' if self.viz_config.use_cumulative else 'Yearly occurrences'
        
        fig.update_layout(
            title=self.viz_config.title,
            xaxis_title='Year',
            yaxis_title=y_axis_title,
            legend_title='Keyword',
            height=self.viz_config.height,
            hovermode='x unified'
        )


class KeywordTrendsVisualizer:
    """Main class for creating keyword trends visualizations"""
    
    def __init__(self, keywords_df: Optional[pd.DataFrame] = None, 
                 grants_df: Optional[pd.DataFrame] = None):
        self.keywords_df = keywords_df if keywords_df is not None else config.Keywords.load()
        self.grants_df = grants_df if grants_df is not None else config.Grants.load()
        self.data_processor = KeywordTrendsDataProcessor(self.keywords_df, self.grants_df)
    
    def create_visualization(self, filter_config: FilterConfig, 
                           viz_config: VisualizationConfig) -> Optional[go.Figure]:
        """Create keyword trends visualization"""
        # Create keyword trends data
        kw_years = self.data_processor.create_trends_data(filter_config)
        
        if kw_years.empty:
            return None
        
        # Select keywords for display
        df_top = KeywordSelector.select_keywords_for_display(kw_years, viz_config)
        
        # Create matrices for visualization
        occ_matrix_display = MatrixProcessor.create_occurrence_matrix(df_top, viz_config.use_cumulative)
        
        # Calculate baseline
        year_range = self._determine_year_range(df_top)
        baseline_data = BaselineCalculator.calculate_baseline(kw_years, self.grants_df, viz_config, year_range)
        
        # Create and return figure
        plot_builder = PlotBuilder(viz_config)
        return plot_builder.create_keyword_trends_figure(occ_matrix_display, baseline_data)
    
    def _determine_year_range(self, df_top: pd.DataFrame) -> Tuple[int, int]:
        """Determine the year range for baseline calculation"""
        if not df_top.empty:
            return df_top['start_year'].min(), df_top['start_year'].max()
        else:
            # Fallback to grants data range
            return self.grants_df['start_year'].min(), self.grants_df['start_year'].max()


class TreemapDataProcessor:
    """Processes data for treemap visualizations"""
    
    @staticmethod
    def create_treemap_data(categories: List[Dict[str, Any]], 
                          max_research_fields: Optional[int] = None,
                          max_categories_per_field: Optional[int] = None,
                          max_keywords_per_category: Optional[int] = None) -> pd.DataFrame:
        """Create hierarchical data structure for treemap visualization"""
        treemap_data = []
        
        # Group categories by FOR division
        for_groups = TreemapDataProcessor._group_categories_by_field(categories)
        
        # Apply research field limit
        if max_research_fields is not None:
            for_groups = TreemapDataProcessor._limit_research_fields(for_groups, max_research_fields)
        
        # Create treemap data with 3-level hierarchy
        for for_code_value, for_data in for_groups.items():
            TreemapDataProcessor._add_field_level_data(
                treemap_data, for_code_value, for_data, 
                max_categories_per_field, max_keywords_per_category
            )
        
        return pd.DataFrame(treemap_data)
    
    @staticmethod
    def _group_categories_by_field(categories: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Group categories by research field"""
        for_groups = {}
        
        for category in categories:
            field_of_research = category.get('field_of_research')
            
            if field_of_research:
                for_code_value = field_of_research
                for_division_name = field_of_research.replace('_', ' ').title()
            else:
                # Fallback to old for_code structure
                for_code = category.get('for_code', 'Unknown')
                for_code_value, for_division_name = TreemapDataProcessor._parse_for_code(for_code)
            
            if for_code_value not in for_groups:
                for_groups[for_code_value] = {
                    'name': for_division_name,
                    'categories': []
                }
            for_groups[for_code_value]['categories'].append(category)
        
        return for_groups
    
    @staticmethod
    def _parse_for_code(for_code: Union[str, Dict[str, Any]]) -> Tuple[str, str]:
        """Parse for_code to extract code value and name"""
        if isinstance(for_code, str):
            return for_code, f'FOR {for_code}'
        elif isinstance(for_code, dict):
            code_value = for_code.get('code', 'Unknown')
            division_name = for_code.get('name', f'FOR {code_value}')
            return code_value, division_name
        else:
            return 'Unknown', 'Unknown FOR Division'
    
    @staticmethod
    def _limit_research_fields(for_groups: Dict[str, Dict[str, Any]], 
                             max_research_fields: int) -> Dict[str, Dict[str, Any]]:
        """Limit research fields by sorting by total keyword count"""
        for_items = []
        for for_code_value, for_data in for_groups.items():
            total_keywords = sum(len(cat.get('keywords', [])) for cat in for_data['categories'])
            for_items.append((for_code_value, for_data, total_keywords))
        
        # Sort by keyword count (descending) and limit
        for_items.sort(key=lambda x: x[2], reverse=True)
        return {item[0]: item[1] for item in for_items[:max_research_fields]}
    
    @staticmethod
    def _add_field_level_data(treemap_data: List[Dict[str, Any]], for_code_value: str, 
                            for_data: Dict[str, Any], max_categories_per_field: Optional[int],
                            max_keywords_per_category: Optional[int]):
        """Add field-level data to treemap"""
        for_name = for_data['name']
        
        # Get categories for this research field (apply category limit if specified)
        categories_for_this_for = for_data['categories']
        if max_categories_per_field is not None:
            categories_for_this_for = TreemapDataProcessor._limit_categories(
                categories_for_this_for, max_categories_per_field
            )
        
        # Calculate total keywords for this research field
        for_keyword_count = sum(
            len(cat.get('keywords', [])[:max_keywords_per_category] if max_keywords_per_category else cat.get('keywords', []))
            for cat in categories_for_this_for
        )
        for_size = max(1, for_keyword_count)
        
        # Level 1: Research Fields (top level)
        treemap_data.append({
            'id': f"FIELD_{for_code_value}",
            'parent': '',
            'name': for_name,
            'value': for_size,
            'level': 'Research Field',
            'item_type': 'research_field'
        })
        
        # Level 2 & 3: Categories and Keywords
        TreemapDataProcessor._add_category_and_keyword_data(
            treemap_data, for_code_value, categories_for_this_for, max_keywords_per_category
        )
    
    @staticmethod
    def _limit_categories(categories: List[Dict[str, Any]], max_categories: int) -> List[Dict[str, Any]]:
        """Limit categories by keyword count"""
        category_items = []
        for category in categories:
            keyword_count = len(category.get('keywords', []))
            category_items.append((category, keyword_count))
        
        # Sort by keyword count (descending) and limit
        category_items.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in category_items[:max_categories]]
    
    @staticmethod
    def _add_category_and_keyword_data(treemap_data: List[Dict[str, Any]], for_code_value: str,
                                     categories: List[Dict[str, Any]], max_keywords_per_category: Optional[int]):
        """Add category and keyword level data to treemap"""
        for category in categories:
            category_name = category['name']
            keywords = category.get('keywords', [])
            
            # Apply keyword limit
            if max_keywords_per_category is not None:
                keywords = keywords[:max_keywords_per_category]
            
            # Size by number of keywords (minimum 1)
            category_size = max(1, len(keywords))
            
            # Level 2: Categories
            treemap_data.append({
                'id': f"FIELD_{for_code_value} >> {category_name}",
                'parent': f"FIELD_{for_code_value}",
                'name': category_name,
                'value': category_size,
                'level': 'Category',
                'item_type': 'category'
            })
            
            # Level 3: Keywords
            for keyword in keywords:
                treemap_data.append({
                    'id': f"FIELD_{for_code_value} >> {category_name} >> KW: {keyword}",
                    'parent': f"FIELD_{for_code_value} >> {category_name}",
                    'name': keyword,
                    'value': 1,
                    'level': 'Keyword',
                    'item_type': 'keyword'
                })


class TreemapVisualizer:
    """Creates treemap visualizations"""
    
    def __init__(self):
        pass
    
    def create_research_landscape_treemap(self, categories: Optional[List[Dict[str, Any]]] = None,
                                        classification_results: Optional[List[Dict[str, Any]]] = None,
                                        title: Optional[str] = None,
                                        height: int = 800,
                                        font_size: int = 9,
                                        color_map: Optional[Dict[str, str]] = None,
                                        max_research_fields: Optional[int] = None,
                                        max_categories_per_field: Optional[int] = None,
                                        max_keywords_per_category: Optional[int] = None) -> Optional[go.Figure]:
        """Create a comprehensive treemap visualization of the research landscape"""
        
        # Load data if not provided
        if categories is None:
            categories = config.Categories.load().to_dict('records')
        
        if not categories:
            return None
        
        # Create treemap data
        treemap_df = TreemapDataProcessor.create_treemap_data(
            categories, 
            max_research_fields=max_research_fields,
            max_categories_per_field=max_categories_per_field,
            max_keywords_per_category=max_keywords_per_category
        )
        
        if treemap_df.empty:
            return None
        
        # Setup visualization parameters
        color_map = color_map or self._get_default_color_map()
        title = title or 'Research Landscape: Research Fields → Categories → Keywords'
        
        # Create the treemap
        fig = px.treemap(
            treemap_df,
            ids='id',
            names='name',
            parents='parent', 
            values='value',
            title=title,
            color='level',
            color_discrete_map=color_map,
            hover_data=['level', 'value', 'item_type']
        )
        
        # Update layout and traces
        fig.update_layout(
            font_size=font_size,
            title_font_size=font_size + 7,
            height=height,
            margin=dict(t=60, l=25, r=25, b=25)
        )
        
        fig.update_traces(
            textinfo="label",
            textfont_size=font_size,
            textposition="middle center"
        )
        
        return fig
    
    @staticmethod
    def _get_default_color_map() -> Dict[str, str]:
        """Get default color mapping for treemap levels"""
        return {
            'Research Field': '#1f77b4',  # Blue
            'Category': '#ff7f0e',        # Orange  
            'Keyword': '#2ca02c',         # Green
        }


@dataclass
class EntityTrendsConfig:
    """Configuration for entity trends visualization"""
    # Data structure
    entity_column: str
    time_column: str
    value_column: str
    
    # Selection and ranking
    max_entities: int
    ranking_metric: str = None  # which metric to use for ranking (if different from value_column)
    display_metric: str = None  # which metric to display on y-axis (if different from value_column)
    
    # Aggregation
    aggregation_method: str = 'sum'  # 'sum', 'count', 'mean'
    use_cumulative: bool = True
    
    # Visualization
    chart_type: str = 'line'  # 'line', 'area_stacked'
    show_others_group: bool = False  # for area charts
    
    # Styling
    title: str = "Entity Trends Over Time"
    x_axis_label: str = "Year"
    y_axis_label: str = "Count"
    height: int = 600


class PopularityTrendsVisualizer:
    """
    Generalized visualizer for tracking entity popularity/trends over time.
    
    This class abstracts the common pattern found across Keywords, Grants, and Categories pages
    where we track multiple entities (keywords, funders, categories) over time with various metrics.
    """
    
    def __init__(self):
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'
        ]
    
    def create_trends_visualization(self, 
                                  data: pd.DataFrame, 
                                  config: EntityTrendsConfig,
                                  progress_callback=None) -> go.Figure:
        """
        Create a trends visualization from generic data.
        
        Args:
            data: DataFrame with entity, time, and value columns
            config: Configuration specifying how to process and visualize data
            progress_callback: Optional callback for progress updates
            
        Returns:
            Plotly Figure object
        """
        if progress_callback:
            progress_callback("Preparing time series data...", 20)
        
        # Prepare time series data
        time_series_df = self._prepare_time_series_data(data, config)
        
        if time_series_df.empty:
            return self._create_empty_figure(config)
        
        if progress_callback:
            progress_callback("Selecting top entities...", 40)
        
        # Select top entities
        top_entities_df = self._select_top_entities(time_series_df, config)
        
        if progress_callback:
            progress_callback("Creating visualization...", 80)
        
        # Create visualization based on chart type
        if config.chart_type == 'line':
            fig = self._create_line_chart(top_entities_df, config)
        elif config.chart_type == 'area_stacked':
            fig = self._create_stacked_area_chart(top_entities_df, config)
        else:
            raise ValueError(f"Unsupported chart type: {config.chart_type}")
        
        if progress_callback:
            progress_callback("Visualization complete!", 100)
        
        return fig
    
    def _prepare_time_series_data(self, data: pd.DataFrame, config: EntityTrendsConfig) -> pd.DataFrame:
        """Convert input data to standardized time series format"""
        # Ensure required columns exist
        required_cols = [config.entity_column, config.time_column, config.value_column]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check if ranking_metric is different from value_column and needs to be preserved
        ranking_metric = config.ranking_metric or 'value'
        preserve_ranking = ranking_metric != 'value' and ranking_metric != config.value_column and ranking_metric in data.columns
        
        # Aggregate data by entity and time
        if config.aggregation_method == 'sum':
            agg_cols = [config.value_column]
            if preserve_ranking:
                agg_cols.append(ranking_metric)
            time_series = data.groupby([config.entity_column, config.time_column])[agg_cols].sum().reset_index()
        elif config.aggregation_method == 'count':
            time_series = data.groupby([config.entity_column, config.time_column]).size().reset_index(name=config.value_column)
            if preserve_ranking and ranking_metric in data.columns:
                ranking_agg = data.groupby([config.entity_column, config.time_column])[ranking_metric].sum().reset_index()
                time_series[ranking_metric] = ranking_agg[ranking_metric]
        elif config.aggregation_method == 'mean':
            agg_cols = [config.value_column]
            if preserve_ranking:
                agg_cols.append(ranking_metric)
            time_series = data.groupby([config.entity_column, config.time_column])[agg_cols].mean().reset_index()
        else:
            raise ValueError(f"Unsupported aggregation method: {config.aggregation_method}")
        
        # Rename columns to standard names for easier processing
        if preserve_ranking:
            time_series = time_series.rename(columns={
                config.entity_column: 'entity',
                config.time_column: 'time',
                config.value_column: 'value'
            })
        else:
            time_series.columns = ['entity', 'time', 'value']
        
        return time_series
    
    def _select_top_entities(self, time_series_df: pd.DataFrame, config: EntityTrendsConfig) -> pd.DataFrame:
        """Select top N entities based on ranking metric"""
        # Calculate ranking metric for each entity
        ranking_metric = config.ranking_metric or 'value'
        
        # If ranking_metric is the same as value_column, it was renamed to 'value'
        if ranking_metric == config.value_column:
            ranking_column = 'value'
        elif ranking_metric == 'value':
            ranking_column = 'value'
        else:
            # ranking_metric should still be preserved in the dataframe
            if ranking_metric not in time_series_df.columns:
                raise ValueError(f"Column not found: {ranking_metric}. Available columns: {time_series_df.columns.tolist()}")
            ranking_column = ranking_metric
        
        entity_rankings = time_series_df.groupby('entity')[ranking_column].sum().sort_values(ascending=False)
        
        # Select top entities
        if config.show_others_group and len(entity_rankings) > config.max_entities:
            top_entities = entity_rankings.head(config.max_entities - 1).index.tolist()
            
            # Group remaining entities as "Others"
            others_data = time_series_df[~time_series_df['entity'].isin(top_entities)].copy()
            if not others_data.empty:
                others_aggregated = others_data.groupby('time')['value'].sum().reset_index()
                others_aggregated['entity'] = 'Others'
                
                # Combine top entities with Others
                top_data = time_series_df[time_series_df['entity'].isin(top_entities)]
                return pd.concat([top_data, others_aggregated], ignore_index=True)
        else:
            top_entities = entity_rankings.head(config.max_entities).index.tolist()
        
        return time_series_df[time_series_df['entity'].isin(top_entities)]
    
    def _create_line_chart(self, data: pd.DataFrame, config: EntityTrendsConfig) -> go.Figure:
        """Create line chart for individual entity tracking - OPTIMIZED VERSION"""
        fig = go.Figure()
        
        # Get time range and entities
        times = sorted(data['time'].unique())
        entities = data['entity'].unique()
        
        # OPTIMIZATION: Create pivot table once instead of filtering for each entity
        pivot_data = data.pivot_table(index='time', columns='entity', values='value', aggfunc='sum', fill_value=0)
        
        # Ensure all times are present
        time_index = pd.Index(times)
        pivot_data = pivot_data.reindex(time_index, fill_value=0)
        
        # Create traces for each entity
        for i, entity in enumerate(entities):
            if entity not in pivot_data.columns:
                continue
                
            full_values = pivot_data[entity].values
            
            # Apply cumulative if requested
            if config.use_cumulative:
                full_values = np.cumsum(full_values)
            
            # Choose color
            color = self.color_palette[i % len(self.color_palette)]
            
            # Create hover template
            hover_template = f'<b>{entity}</b><br>' + \
                           f'{config.x_axis_label}: %{{x}}<br>' + \
                           f'{config.y_axis_label}: %{{y}}<br>' + \
                           '<extra></extra>'
            
            fig.add_trace(go.Scatter(
                x=times,
                y=full_values,
                mode='lines+markers',
                name=entity,
                line=dict(width=2, color=color),
                marker=dict(size=6),
                hovertemplate=hover_template
            ))
        
        # Update layout
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_axis_label,
            yaxis_title=config.y_axis_label,
            height=config.height,
            hovermode='x unified',
            legend_title='Entity',
            template='plotly_white'
        )
        
        return fig
    
    def _create_stacked_area_chart(self, data: pd.DataFrame, config: EntityTrendsConfig) -> go.Figure:
        """Create stacked area chart for aggregate tracking"""
        # Create pivot table
        pivot_data = data.pivot(index='time', columns='entity', values='value').fillna(0)
        
        # Sort entities by total for better stacking order
        entity_totals = pivot_data.sum().sort_values(ascending=False)
        pivot_data = pivot_data[entity_totals.index]
        
        fig = go.Figure()
        
        # Create traces for each entity
        for i, entity in enumerate(pivot_data.columns):
            values = pivot_data[entity].values
            
            # Apply cumulative if requested (note: for stacked areas, usually not cumulative per entity)
            if config.use_cumulative:
                values = np.cumsum(values)
            
            # Choose color, use distinct color for "Others"
            if entity == 'Others':
                color = '#cccccc'
            else:
                color = self.color_palette[i % len(self.color_palette)]
            
            # Create hover template
            hover_template = f'<b>{entity}</b><br>' + \
                           f'{config.x_axis_label}: %{{x}}<br>' + \
                           f'{config.y_axis_label}: %{{y}}<br>' + \
                           '<extra></extra>'
            
            fig.add_trace(go.Scatter(
                x=pivot_data.index,
                y=values,
                mode='lines',
                stackgroup='one',
                name=entity,
                line=dict(width=0),
                fillcolor=color,
                hovertemplate=hover_template
            ))
        
        # Update layout
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_axis_label,
            yaxis_title=config.y_axis_label,
            height=config.height,
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            template='plotly_white'
        )
        
        return fig
    
    def _create_empty_figure(self, config: EntityTrendsConfig) -> go.Figure:
        """Create empty figure when no data is available"""
        fig = go.Figure()
        fig.update_layout(
            title=f"{config.title} - No Data Available",
            xaxis_title=config.x_axis_label,
            yaxis_title=config.y_axis_label,
            height=config.height,
            template='plotly_white'
        )
        fig.add_annotation(
            text="No data available for the selected criteria",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig


@dataclass
class DataExplorerConfig:
    """Configuration for data exploration functionality"""
    # Display settings
    title: str = "Data Explorer"
    description: str = "Explore the dataset with search and filtering capabilities"
    max_display_rows: int = 100
    
    # Search settings
    search_columns: List[str] = None  # Columns to search in (None = all text columns)
    search_placeholder: str = "Enter search terms..."
    search_help: str = "Search across multiple columns. Use spaces to separate terms."
    
    # Column settings
    display_columns: List[str] = None  # Columns to display (None = all columns)
    column_formatters: Dict[str, callable] = None  # Custom formatters for specific columns
    column_renames: Dict[str, str] = None  # Column name mappings for display
    
    # Statistics settings
    show_statistics: bool = True
    statistics_columns: List[str] = None  # Columns to include in statistics
    
    # Additional features
    show_data_info: bool = True
    enable_download: bool = True


class DataExplorer:
    """
    Generalized data exploration component for displaying searchable DataFrames.
    
    This class abstracts the common pattern found across Keywords, Grants, and Categories pages
    where we display filtered data in tables with search functionality.
    """
    
    def __init__(self):
        pass
    
    def render_explorer(self, data: pd.DataFrame, config: DataExplorerConfig) -> None:
        """
        Render the complete data exploration interface.
        
        Args:
            data: DataFrame to explore
            config: Configuration for display and behavior
        """
        if data.empty:
            st.warning(f"No data available")
            return
        
        # Search functionality
        search_term = self._render_search_input(config)
        
        # Apply search filter
        filtered_data = self._apply_search_filter(data, search_term, config)
        
        # Display the DataFrame
        self._display_dataframe(filtered_data, config)
        
        # Download option
        if config.enable_download:
            self._render_download_option(filtered_data, config.title)
    
    def _render_search_input(self, config: DataExplorerConfig) -> str:
        """Render search input field"""
        search_term = st.text_input(
            "Search",
            placeholder=config.search_placeholder,
            help=config.search_help,
            key=f"search_{hash(config.title)}"
        )
        
        return search_term.strip()
    
    def _apply_search_filter(self, data: pd.DataFrame, search_term: str, config: DataExplorerConfig) -> pd.DataFrame:
        """Apply search filter to the data"""
        if not search_term:
            return data
        
        # Determine columns to search
        search_columns = config.search_columns
        if search_columns is None:
            # Default to text/object columns
            search_columns = data.select_dtypes(include=['object', 'string']).columns.tolist()
        
        # Ensure search columns exist in data
        search_columns = [col for col in search_columns if col in data.columns]
        
        if not search_columns:
            st.warning("No searchable columns found in the data.")
            return data
        
        # Split search terms
        search_terms = search_term.lower().split()
        
        # Create mask for each search term
        masks = []
        for term in search_terms:
            term_masks = []
            for col in search_columns:
                # Convert column to string and search
                col_mask = data[col].astype(str).str.lower().str.contains(term, na=False, regex=False)
                term_masks.append(col_mask)
            
            # Combine masks for this term (OR across columns)
            if term_masks:
                term_mask = pd.concat(term_masks, axis=1).any(axis=1)
                masks.append(term_mask)
        
        # Combine all term masks (AND across terms)
        if masks:
            final_mask = pd.concat(masks, axis=1).all(axis=1)
            filtered_data = data[final_mask]
        else:
            filtered_data = data
        
        # Show search results info
        if search_term:
            st.info(f"Found {len(filtered_data):,} records matching '{search_term}' (searched in: {', '.join(search_columns)})")
        
        return filtered_data
    
    def _display_dataframe(self, data: pd.DataFrame, config: DataExplorerConfig) -> None:
        """Display the DataFrame with proper formatting"""
        if data.empty:
            st.warning("No records match your search criteria.")
            return
        
        # Prepare display data
        display_data = data.copy()
        
        # Apply column selection
        if config.display_columns:
            available_columns = [col for col in config.display_columns if col in display_data.columns]
            if available_columns:
                display_data = display_data[available_columns]
        
        # Apply column formatters
        if config.column_formatters:
            for col, formatter in config.column_formatters.items():
                if col in display_data.columns:
                    try:
                        display_data[col] = display_data[col].apply(formatter)
                    except Exception as e:
                        st.warning(f"Error formatting column {col}: {e}")
        
        # Apply column renames
        if config.column_renames:
            rename_dict = {k: v for k, v in config.column_renames.items() if k in display_data.columns}
            display_data = display_data.rename(columns=rename_dict)
        
        # Limit displayed rows
        if len(display_data) > config.max_display_rows:
            st.warning(f"Showing first {config.max_display_rows:,} of {len(display_data):,} records. Use search to narrow results.")
            display_data = display_data.head(config.max_display_rows)
        
        # Display the data
        st.dataframe(
            display_data,
            use_container_width=True,
            height=400,
            key=f"dataframe_{hash(config.title)}"
        )
    
    def _render_download_option(self, data: pd.DataFrame, title: str) -> None:
        """Render download option for the filtered data"""
        if data.empty:
            return
        
        csv_data = data.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name=f"{title.lower().replace(' ', '_')}_data.csv",
            mime="text/csv",
            help="Download the current filtered data as a CSV file"
        )


# Convenience functions for backward compatibility
def create_research_landscape_treemap(categories: Optional[List[Dict[str, Any]]] = None,
                                    classification_results: Optional[List[Dict[str, Any]]] = None,
                                    title: Optional[str] = None,
                                    height: int = 800,
                                    font_size: int = 9,
                                    color_map: Optional[Dict[str, str]] = None,
                                    max_research_fields: Optional[int] = None,
                                    max_categories_per_field: Optional[int] = None,
                                    max_keywords_per_category: Optional[int] = None) -> Optional[go.Figure]:
    """Create a research landscape treemap - convenience function"""
    visualizer = TreemapVisualizer()
    return visualizer.create_research_landscape_treemap(
        categories=categories,
        classification_results=classification_results,
        title=title,
        height=height,
        font_size=font_size,
        color_map=color_map,
        max_research_fields=max_research_fields,
        max_categories_per_field=max_categories_per_field,
        max_keywords_per_category=max_keywords_per_category
    )
