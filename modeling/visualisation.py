"""
Research Landscape Visualizations - Refactored

This module contains visualization functions for the research link technology 
landscaping analysis, including treemap visualizations of research categories 
and keywords, and data exploration components.

"""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import streamlit as st

import config


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
            textinfo="label+value",
            textfont_size=font_size,
            textposition="middle center",
            marker=dict(
                line=dict(width=2, color='white'),
                pad=dict(t=10, l=10, r=10, b=10)
            )
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


@dataclass
class TrendsConfig:
    """
    Configuration for trend visualization
    
    This dataclass encapsulates all parameters needed to create a trend plot,
    providing a clean interface for customization.
    """
    entity_col: str
    time_col: str
    value_col: str
    max_entities: int = 10
    ranking_col: Optional[str] = None
    aggregation: str = 'sum'
    use_cumulative: bool = True
    chart_type: str = 'line'
    show_others: bool = False
    title: str = "Trends Over Time"
    x_label: str = "Time"
    y_label: str = "Value"
    height: int = 600
    color_palette: List[str] = field(default_factory=lambda: [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'
    ])


class TrendsVisualizer:
    """Unified visualizer for temporal trends across different entity types"""
    
    def create_plot(self, data: pd.DataFrame, config: TrendsConfig) -> go.Figure:
        """Create a trends visualization from input data"""
        self._validate_input(data, config)
        time_series_df = self._prepare_time_series(data, config)
        if time_series_df.empty:
            return self._create_empty_figure(config)
        selected_df = self._select_top_entities(time_series_df, config)
        if config.chart_type == 'line':
            return self._create_line_chart(selected_df, config)
        if config.chart_type == 'area':
            return self._create_area_chart(selected_df, config)
        if config.chart_type == 'stacked_area':
            return self._create_stacked_area_chart(selected_df, config)
        raise ValueError(f"Unsupported chart type: {config.chart_type}")
    
    def _validate_input(self, data: pd.DataFrame, config: TrendsConfig) -> None:
        required_cols = [config.entity_col, config.time_col, config.value_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            available = list(data.columns)
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {available}"
            )
    
    def _prepare_time_series(self, data: pd.DataFrame, config: TrendsConfig) -> pd.DataFrame:
        ranking_col = config.ranking_col or config.value_col
        preserve_ranking = (
            ranking_col != config.value_col and 
            ranking_col in data.columns
        )
        group_cols = [config.entity_col, config.time_col]
        if config.aggregation == 'sum':
            agg_dict = {config.value_col: 'sum'}
            if preserve_ranking:
                agg_dict[ranking_col] = 'sum'
            time_series = data.groupby(group_cols).agg(agg_dict).reset_index()
        elif config.aggregation == 'count':
            time_series = data.groupby(group_cols).size().reset_index(name=config.value_col)
            if preserve_ranking and ranking_col in data.columns:
                ranking_agg = data.groupby(group_cols)[ranking_col].sum().reset_index()
                time_series[ranking_col] = ranking_agg[ranking_col]
        elif config.aggregation == 'mean':
            agg_dict = {config.value_col: 'mean'}
            if preserve_ranking:
                agg_dict[ranking_col] = 'mean'
            time_series = data.groupby(group_cols).agg(agg_dict).reset_index()
        else:
            raise ValueError(f"Unsupported aggregation: {config.aggregation}")
        time_series = time_series.rename(columns={
            config.entity_col: 'entity',
            config.time_col: 'time',
            config.value_col: 'value'
        })
        return time_series
    
    def _select_top_entities(self, time_series_df: pd.DataFrame, config: TrendsConfig) -> pd.DataFrame:
        ranking_col = config.ranking_col or 'value'
        if ranking_col == config.value_col or ranking_col not in time_series_df.columns:
            ranking_col = 'value'
        entity_rankings = (
            time_series_df.groupby('entity')[ranking_col]
            .sum()
            .sort_values(ascending=False)
        )
        if config.show_others and len(entity_rankings) > config.max_entities:
            top_entities = entity_rankings.head(config.max_entities - 1).index.tolist()
            others_data = time_series_df[~time_series_df['entity'].isin(top_entities)].copy()
            if not others_data.empty:
                others_aggregated = (
                    others_data.groupby('time')['value']
                    .sum()
                    .reset_index()
                )
                others_aggregated['entity'] = 'Others'
                top_data = time_series_df[time_series_df['entity'].isin(top_entities)]
                return pd.concat([top_data, others_aggregated], ignore_index=True)
        else:
            top_entities = entity_rankings.head(config.max_entities).index.tolist()
        return time_series_df[time_series_df['entity'].isin(top_entities)]
    
    def _create_line_chart(self, data: pd.DataFrame, config: TrendsConfig) -> go.Figure:
        fig = go.Figure()
        times = sorted(data['time'].unique())
        pivot_data = data.pivot_table(
            index='time',
            columns='entity',
            values='value',
            aggfunc='sum',
            fill_value=0
        )
        time_index = pd.Index(times)
        pivot_data = pivot_data.reindex(time_index, fill_value=0)
        entity_totals = pivot_data.sum().sort_values(ascending=False)
        entities = entity_totals.index.tolist()
        for i, entity in enumerate(entities):
            if entity not in pivot_data.columns:
                continue
            values = pivot_data[entity].values
            if config.use_cumulative:
                values = np.cumsum(values)
            color = config.color_palette[i % len(config.color_palette)]
            hover_template = (
                f'<b>{entity}</b><br>'
                f'{config.x_label}: %{{x}}<br>'
                f'{config.y_label}: %{{y}}<br>'
                '<extra></extra>'
            )
            fig.add_trace(go.Scatter(
                x=times,
                y=values,
                mode='lines+markers',
                name=str(entity),
                line=dict(width=2, color=color),
                marker=dict(size=6),
                hovertemplate=hover_template
            ))
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_label,
            yaxis_title=config.y_label,
            height=config.height,
            hovermode='x unified',
            legend_title='Entity',
            template='plotly_white'
        )
        return fig
    
    def _create_area_chart(self, data: pd.DataFrame, config: TrendsConfig) -> go.Figure:
        fig = self._create_line_chart(data, config)
        for trace in fig.data:
            trace.fill = 'tozeroy'
            trace.mode = 'lines'
        return fig
    
    def _create_stacked_area_chart(self, data: pd.DataFrame, config: TrendsConfig) -> go.Figure:
        pivot_data = data.pivot(
            index='time',
            columns='entity',
            values='value'
        ).fillna(0)
        entity_totals = pivot_data.sum().sort_values(ascending=False)
        pivot_data = pivot_data[entity_totals.index]
        fig = go.Figure()
        for i, entity in enumerate(reversed(pivot_data.columns.tolist())):
            values = pivot_data[entity].values
            if config.use_cumulative:
                values = np.cumsum(values)
            color_index = len(pivot_data.columns) - 1 - i
            if entity == 'Others':
                color = '#cccccc'
            else:
                color = config.color_palette[color_index % len(config.color_palette)]
            hover_template = (
                f'<b>{entity}</b><br>'
                f'{config.x_label}: %{{x}}<br>'
                f'{config.y_label}: %{{y}}<br>'
                '<extra></extra>'
            )
            fig.add_trace(go.Scatter(
                x=pivot_data.index,
                y=values,
                mode='lines',
                stackgroup='one',
                name=str(entity),
                line=dict(width=0),
                fillcolor=color,
                hovertemplate=hover_template
            ))
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_label,
            yaxis_title=config.y_label,
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
    
    def _create_empty_figure(self, config: TrendsConfig) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(
            title=f"{config.title} - No Data Available",
            xaxis_title=config.x_label,
            yaxis_title=config.y_label,
            height=config.height,
            template='plotly_white'
        )
        fig.add_annotation(
            text="No data available for the selected filters",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig


class TrendsDataPreparation:
    """Helper class for preparing data from different sources"""

    @staticmethod
    def from_keyword_grants(
        keywords_df: pd.DataFrame,
        grants_df: pd.DataFrame,
        selected_keywords: Optional[List[str]] = None
    ) -> pd.DataFrame:
        grant_years_lookup = dict(zip(grants_df['id'], grants_df['start_year']))
        grant_years_lookup = {k: v for k, v in grant_years_lookup.items() if pd.notna(v)}
        valid_grant_ids = set(grant_years_lookup.keys())
        if selected_keywords:
            keywords_df = keywords_df[keywords_df['name'].isin(selected_keywords)]
        viz_data = []
        for keyword_name, grants_list in zip(keywords_df['name'], keywords_df['grants']):
            if not grants_list:
                continue
            valid_grants = [g for g in grants_list if g in valid_grant_ids]
            if valid_grants:
                for grant_id in valid_grants:
                    year = grant_years_lookup[grant_id]
                    viz_data.append((keyword_name, int(year), 1))
        if viz_data:
            return pd.DataFrame(viz_data, columns=['keyword', 'year', 'grant_count'])
        return pd.DataFrame(columns=['keyword', 'year', 'grant_count'])

    @staticmethod
    def from_category_grants(
        categories_df: pd.DataFrame,
        grants_df: pd.DataFrame,
        selected_categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        from config import Keywords
        keywords_df = Keywords.load()
        if keywords_df is None or keywords_df.empty:
            return pd.DataFrame(columns=['category', 'year', 'grant_count', 'total_funding'])
        keyword_grants_lookup = {}
        for _, kw_row in keywords_df.iterrows():
            if 'grants' in kw_row and kw_row['grants']:
                keyword_grants_lookup[kw_row['name']] = kw_row['grants']
        if selected_categories:
            categories_df = categories_df[categories_df['name'].isin(selected_categories)]
        viz_data = []
        grant_years = dict(zip(grants_df['id'], grants_df['start_year']))
        grant_funding = dict(zip(grants_df['id'], grants_df.get('funding_amount', [0] * len(grants_df))))
        for _, category in categories_df.iterrows():
            category_name = category['name']
            keywords = category.get('keywords', [])
            if not keywords:
                continue
            category_grant_ids = set()
            for keyword in keywords:
                if keyword in keyword_grants_lookup:
                    category_grant_ids.update(keyword_grants_lookup[keyword])
            for grant_id in category_grant_ids:
                if grant_id in grant_years and pd.notna(grant_years[grant_id]):
                    year = int(grant_years[grant_id])
                    funding = grant_funding.get(grant_id, 0)
                    viz_data.append((category_name, year, 1, funding))
        if viz_data:
            return pd.DataFrame(viz_data, columns=['category', 'year', 'grant_count', 'total_funding'])
        return pd.DataFrame(columns=['category', 'year', 'grant_count', 'total_funding'])

    @staticmethod
    def from_grants_by_attribute(
        grants_df: pd.DataFrame,
        attribute_col: str
    ) -> pd.DataFrame:
        if 'start_year' not in grants_df.columns:
            raise ValueError("grants_df must have 'start_year' column")
        if attribute_col not in grants_df.columns:
            raise ValueError(f"grants_df must have '{attribute_col}' column")
        filtered_df = grants_df[[attribute_col, 'start_year']].dropna()
        filtered_df['grant_count'] = 1
        return filtered_df.rename(columns={'start_year': 'year'})


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
