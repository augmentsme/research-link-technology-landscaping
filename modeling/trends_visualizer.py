"""
Trends Visualizer - Unified module for temporal trends visualization

This module provides a clean, unified interface for creating trend visualizations
across different entity types (keywords, categories, grants, funders, etc.).

The module accepts filtered pandas DataFrames with timestamped entities and produces
Plotly visualizations with flexible configuration options.

Usage:
    from trends_visualizer import TrendsVisualizer, TrendsConfig
    
    # Prepare your data (entity, time, value columns)
    data = pd.DataFrame({
        'keyword': ['AI', 'ML', 'AI', 'ML'],
        'year': [2020, 2020, 2021, 2021],
        'count': [10, 5, 15, 8]
    })
    
    # Configure visualization
    config = TrendsConfig(
        entity_col='keyword',
        time_col='year',
        value_col='count',
        max_entities=10,
        title='Keyword Trends Over Time'
    )
    
    # Create visualization
    visualizer = TrendsVisualizer()
    fig = visualizer.create_plot(data, config)
    fig.show()
"""

from dataclasses import dataclass, field
from typing import Optional, List, Callable
import pandas as pd
import numpy as np
import plotly.graph_objects as go


@dataclass
class TrendsConfig:
    """
    Configuration for trend visualization
    
    This dataclass encapsulates all parameters needed to create a trend plot,
    providing a clean interface for customization.
    """
    # Required: Data structure
    entity_col: str
    """Column name for entity identifier (e.g., 'keyword', 'funder', 'category')"""
    
    time_col: str
    """Column name for time dimension (e.g., 'year', 'date', 'month')"""
    
    value_col: str
    """Column name for the metric to visualize (e.g., 'count', 'funding', 'occurrences')"""
    
    # Selection and filtering
    max_entities: int = 10
    """Maximum number of top entities to display"""
    
    ranking_col: Optional[str] = None
    """Optional: Different column to use for ranking entities (defaults to value_col)"""
    
    # Aggregation and transformation
    aggregation: str = 'sum'
    """How to aggregate values: 'sum', 'count', 'mean'"""
    
    use_cumulative: bool = True
    """Whether to show cumulative values over time"""
    
    # Visualization type
    chart_type: str = 'line'
    """Chart type: 'line', 'area', 'stacked_area'"""
    
    show_others: bool = False
    """Whether to group remaining entities as 'Others' (useful for area charts)"""
    
    # Styling
    title: str = "Trends Over Time"
    """Plot title"""
    
    x_label: str = "Time"
    """X-axis label"""
    
    y_label: str = "Value"
    """Y-axis label"""
    
    height: int = 600
    """Plot height in pixels"""
    
    color_palette: List[str] = field(default_factory=lambda: [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'
    ])
    """Color palette for entities"""


class TrendsVisualizer:
    """
    Unified visualizer for temporal trends across different entity types
    
    This class provides a clean interface for creating trend visualizations from
    DataFrames containing timestamped entity data. It handles data aggregation,
    entity selection, and chart creation with flexible configuration.
    """
    
    def create_plot(self, data: pd.DataFrame, config: TrendsConfig) -> go.Figure:
        """
        Create a trends visualization from input data
        
        This is the main entry point for creating trend plots. It accepts a DataFrame
        with entity, time, and value columns, and produces a Plotly figure based on
        the provided configuration.
        
        Args:
            data: DataFrame with columns matching config (entity_col, time_col, value_col)
            config: TrendsConfig object specifying how to visualize the data
            
        Returns:
            Plotly Figure object ready to display or save
            
        Raises:
            ValueError: If required columns are missing from data
            
        Example:
            >>> data = pd.DataFrame({
            ...     'keyword': ['AI', 'ML', 'AI'],
            ...     'year': [2020, 2020, 2021],
            ...     'count': [10, 5, 15]
            ... })
            >>> config = TrendsConfig(
            ...     entity_col='keyword',
            ...     time_col='year',
            ...     value_col='count'
            ... )
            >>> visualizer = TrendsVisualizer()
            >>> fig = visualizer.create_plot(data, config)
        """
        self._validate_input(data, config)
        
        time_series_df = self._prepare_time_series(data, config)
        
        if time_series_df.empty:
            return self._create_empty_figure(config)
        
        selected_df = self._select_top_entities(time_series_df, config)
        
        if config.chart_type == 'line':
            return self._create_line_chart(selected_df, config)
        elif config.chart_type == 'area':
            return self._create_area_chart(selected_df, config)
        elif config.chart_type == 'stacked_area':
            return self._create_stacked_area_chart(selected_df, config)
        else:
            raise ValueError(f"Unsupported chart type: {config.chart_type}")
    
    def _validate_input(self, data: pd.DataFrame, config: TrendsConfig) -> None:
        """Validate that required columns exist in the data"""
        required_cols = [config.entity_col, config.time_col, config.value_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            available = list(data.columns)
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {available}"
            )
    
    def _prepare_time_series(self, data: pd.DataFrame, config: TrendsConfig) -> pd.DataFrame:
        """
        Aggregate data by entity and time
        
        This method transforms the input data into a standardized time series format
        with entity, time, and value columns.
        """
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
        """
        Select top N entities based on ranking metric
        
        Optionally groups remaining entities as 'Others' for cleaner visualization.
        """
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
        """Create line chart for individual entity tracking"""
        fig = go.Figure()
        
        times = sorted(data['time'].unique())
        entities = data['entity'].unique()
        
        pivot_data = data.pivot_table(
            index='time',
            columns='entity',
            values='value',
            aggfunc='sum',
            fill_value=0
        )
        
        time_index = pd.Index(times)
        pivot_data = pivot_data.reindex(time_index, fill_value=0)
        
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
        """Create area chart (similar to line but with fill)"""
        fig = self._create_line_chart(data, config)
        
        for trace in fig.data:
            trace.fill = 'tozeroy'
            trace.mode = 'lines'
        
        return fig
    
    def _create_stacked_area_chart(self, data: pd.DataFrame, config: TrendsConfig) -> go.Figure:
        """Create stacked area chart for aggregate tracking"""
        pivot_data = data.pivot(
            index='time',
            columns='entity',
            values='value'
        ).fillna(0)
        
        entity_totals = pivot_data.sum().sort_values(ascending=False)
        pivot_data = pivot_data[entity_totals.index]
        
        fig = go.Figure()
        
        for i, entity in enumerate(pivot_data.columns):
            values = pivot_data[entity].values
            
            if config.use_cumulative:
                values = np.cumsum(values)
            
            if entity == 'Others':
                color = '#cccccc'
            else:
                color = config.color_palette[i % len(config.color_palette)]
            
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
        """Create empty figure when no data is available"""
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
    """
    Helper class for preparing data from different sources
    
    This class provides convenience methods to transform various data structures
    into the format expected by TrendsVisualizer.
    """
    
    @staticmethod
    def from_keyword_grants(
        keywords_df: pd.DataFrame,
        grants_df: pd.DataFrame,
        selected_keywords: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Prepare keyword trend data from keywords and grants DataFrames
        
        Args:
            keywords_df: DataFrame with 'name' and 'grants' columns
            grants_df: DataFrame with 'id' and 'start_year' columns
            selected_keywords: Optional list of keywords to include
            
        Returns:
            DataFrame with columns: keyword, year, grant_count
        """
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
        else:
            return pd.DataFrame(columns=['keyword', 'year', 'grant_count'])
    
    @staticmethod
    def from_category_grants(
        categories_df: pd.DataFrame,
        grants_df: pd.DataFrame,
        selected_categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Prepare category trend data from categories and grants DataFrames
        
        Categories contain keywords, and keywords contain grants. This method:
        1. Gets keywords for each category
        2. Looks up grants for those keywords
        3. Creates trend data by category and year
        
        Args:
            categories_df: DataFrame with category data (has 'keywords' column)
            grants_df: DataFrame with grant data
            selected_categories: Optional list of categories to include
            
        Returns:
            DataFrame with columns: category, year, grant_count (and optionally funding)
        """
        # Load keywords data to get keyword-to-grants mapping
        from config import Keywords
        keywords_df = Keywords.load()
        
        if keywords_df is None or keywords_df.empty:
            return pd.DataFrame(columns=['category', 'year', 'grant_count', 'total_funding'])
        
        # Build keyword-to-grants lookup
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
            
            # Collect all grants for this category's keywords
            category_grant_ids = set()
            for keyword in keywords:
                if keyword in keyword_grants_lookup:
                    category_grant_ids.update(keyword_grants_lookup[keyword])
            
            # Create trend data for each grant
            for grant_id in category_grant_ids:
                if grant_id in grant_years and pd.notna(grant_years[grant_id]):
                    year = int(grant_years[grant_id])
                    funding = grant_funding.get(grant_id, 0)
                    viz_data.append((category_name, year, 1, funding))
        
        if viz_data:
            return pd.DataFrame(viz_data, columns=['category', 'year', 'grant_count', 'total_funding'])
        else:
            return pd.DataFrame(columns=['category', 'year', 'grant_count', 'total_funding'])
    
    @staticmethod
    def from_grants_by_attribute(
        grants_df: pd.DataFrame,
        attribute_col: str
    ) -> pd.DataFrame:
        """
        Prepare grant trend data grouped by an attribute (e.g., funder, source)
        
        Args:
            grants_df: DataFrame with grant data
            attribute_col: Column name to group by (e.g., 'funder', 'source')
            
        Returns:
            DataFrame with columns: {attribute_col}, year, grant_count
        """
        if 'start_year' not in grants_df.columns:
            raise ValueError("grants_df must have 'start_year' column")
        
        if attribute_col not in grants_df.columns:
            raise ValueError(f"grants_df must have '{attribute_col}' column")
        
        filtered_df = grants_df[[attribute_col, 'start_year']].dropna()
        filtered_df['grant_count'] = 1
        
        return filtered_df.rename(columns={'start_year': 'year'})
