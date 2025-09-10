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
        """Sort matrix columns for legend ordering"""
        if matrix.empty:
            return []
        
        if use_cumulative:
            # For cumulative data, sort by final cumulative value (last row)
            return matrix.iloc[-1].sort_values(ascending=True).index.tolist()
        else:
            # For yearly data, sort by total sum across all years
            return matrix.sum(axis=0).sort_values(ascending=True).index.tolist()


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
        
        # Sort columns for legend ordering
        sorted_columns = MatrixProcessor.sort_matrix_columns(occ_matrix_display, self.viz_config.use_cumulative)
        
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
