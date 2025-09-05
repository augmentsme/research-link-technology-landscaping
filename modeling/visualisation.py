"""
Research Landscape Visualizations

This module contains visualization functions for the research link technology 
landscaping analysis, including treemap visualizations of research categories 
and keywords, and keyword trends over time.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

import config
from models import FORCode


def create_keyword_trends_data(keywords_df: pd.DataFrame, grants_df: pd.DataFrame, min_count: int = 10, 
                              funder_filter: Optional[List[str]] = None, 
                              source_filter: Optional[List[str]] = None,
                              keyword_type_filter: Optional[List[str]] = None,
                              for_code_filter: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Create keyword trends data from keywords and grants dataframes
    
    Args:
        keywords_df: DataFrame with keywords data
        grants_df: DataFrame with grants data (should have 'start_year' column)
        min_count: Minimum count threshold for keywords to include
        funder_filter: Optional list of funders to filter by
        source_filter: Optional list of sources to filter by
        keyword_type_filter: Optional list of keyword types to filter by
        for_code_filter: Optional list of FOR codes to filter by
        
    Returns:
        pandas.DataFrame: DataFrame with keyword trends over time
    """
    # Filter keywords by minimum count
    # Only set index if it's not already set to avoid KeyError
    if 'id' in grants_df.columns:
        grants_df = grants_df.set_index('id')
    
    # Apply funder and source filters to grants
    filtered_grants_df = grants_df.copy()
    if funder_filter:
        filtered_grants_df = filtered_grants_df[filtered_grants_df['funder'].isin(funder_filter)]
    if source_filter:
        filtered_grants_df = filtered_grants_df[filtered_grants_df['source'].isin(source_filter)]
    
    # Apply FOR code filter to grants
    if for_code_filter:
        # Function to check if any of the filter codes are in the grant's FOR codes
        def has_for_code(for_codes_str, for_primary_val):
            if pd.isna(for_codes_str) and pd.isna(for_primary_val):
                return False
            
            # Check primary FOR code
            if not pd.isna(for_primary_val):
                # Convert float to string for comparison (e.g., 406.0 -> "406")
                primary_str = str(int(for_primary_val))
                if any(code in primary_str for code in for_code_filter):
                    return True
            
            # Check secondary FOR codes (comma-separated string)
            if not pd.isna(for_codes_str):
                for_codes_list = str(for_codes_str).split(',')
                for filter_code in for_code_filter:
                    if any(filter_code in for_code.strip() for for_code in for_codes_list):
                        return True
            
            return False
        
        mask = filtered_grants_df.apply(
            lambda row: has_for_code(row.get('for'), row.get('for_primary')), axis=1
        )
        filtered_grants_df = filtered_grants_df[mask]
    
    # Apply keyword type filter to keywords
    filtered_keywords_df = keywords_df.copy()
    if keyword_type_filter:
        filtered_keywords_df = filtered_keywords_df[filtered_keywords_df['type'].isin(keyword_type_filter)]
    
    # Filter keywords to only include those with grants in the filtered set
    filtered_keywords_df['filtered_grants'] = filtered_keywords_df['grants'].apply(
        lambda x: [grant_id for grant_id in x if grant_id in filtered_grants_df.index]
    )
    filtered_keywords_df['filtered_count'] = filtered_keywords_df['filtered_grants'].map(len)
    
    filtered_keywords = filtered_keywords_df[filtered_keywords_df['filtered_count'] > min_count].copy()
    
    # Create start_year mapping for each keyword using filtered grants
    if 'start_year' not in filtered_keywords.columns:
        filtered_keywords['start_year'] = filtered_keywords['filtered_grants'].map(
            lambda x: [filtered_grants_df.loc[i, "start_year"] for i in x if i in filtered_grants_df.index and not np.isnan(filtered_grants_df.loc[i, "start_year"])]
        )
    
    # Explode the keywords to create term-year pairs
    kw_years_list = []
    for idx, row in filtered_keywords.iterrows():
        term = row['term'] if 'term' in row else row.name
        for year in row['start_year']:
            kw_years_list.append({'term': term, 'start_year': int(year)})
    
    return pd.DataFrame(kw_years_list)


def create_keyword_trends_visualization(
    keywords_df: Optional[pd.DataFrame] = config.Keywords.load(),
    grants_df: Optional[pd.DataFrame] = config.Grants.load(),
    min_count: int = 10,
    top_n: int = 20,
    title: str = "Cumulative keyword occurrences over time — Top keywords",
    height: int = 600,
    show_average: bool = True,
    show_error_bars: bool = True,
    average_from_population: bool = False,
    show_baseline: bool = False,
    custom_keywords: Optional[List[str]] = None,
    funder_filter: Optional[List[str]] = None,
    source_filter: Optional[List[str]] = None,
    keyword_type_filter: Optional[List[str]] = None,
    for_code_filter: Optional[List[str]] = None,
    use_cumulative: bool = True
) -> Optional[go.Figure]:
    """
    Create keyword trends visualization with cumulative or yearly occurrences over time
    
    Args:
        keywords_df: DataFrame with keywords data
        grants_df: DataFrame with grants data (should have 'start_year' column)
        min_count: Minimum count threshold for keywords to include
        top_n: Number of top keywords to show as individual lines (0 to show only average)
        title: Title for the visualization
        height: Height of the visualization in pixels
        show_average: Whether to show average trend line
        show_error_bars: Whether to show error bars around average
        average_from_population: Whether to calculate average from all keywords or just top N
        show_baseline: Whether to show baseline trend (keywords per grant per year)
        custom_keywords: List of specific keywords to display
        funder_filter: Filter by specific funders
        source_filter: Filter by specific sources  
        keyword_type_filter: Filter by keyword types
        for_code_filter: Filter by FOR codes
        use_cumulative: If True, show cumulative counts; if False, show yearly raw counts
    
    Returns:
        Plotly figure object or None if no data available
    """
    
    # Create keyword trends data
    kw_years = create_keyword_trends_data(keywords_df, grants_df, min_count, funder_filter, source_filter, keyword_type_filter, for_code_filter)
    
    if kw_years.empty:
        return None
    
    # Create full population data for average calculation (if needed)
    if average_from_population and show_average:
        # Use all terms that meet min_count for average calculation
        df_population = (
            kw_years
            .groupby(['term', 'start_year'])
            .size()
            .reset_index(name='occurrences')
            .sort_values(['term', 'start_year'])
        )
    
    # Find terms for individual line display
    if custom_keywords:
        # Use user-specified custom keywords
        # Filter to only include keywords that exist in the data and meet min_count
        available_keywords = kw_years['term'].value_counts()
        valid_custom_keywords = [kw for kw in custom_keywords if kw in available_keywords and available_keywords[kw] >= min_count]
        
        if valid_custom_keywords:
            df_top = (
                kw_years[kw_years['term'].isin(valid_custom_keywords)]
                .groupby(['term', 'start_year'])
                .size()
                .reset_index(name='occurrences')
                .sort_values(['term', 'start_year'])
            )
        else:
            df_top = pd.DataFrame()
    elif top_n > 0:
        # Use top N keywords by occurrence count
        top_terms = kw_years['term'].value_counts().head(top_n).index.tolist()
        
        # Build df_top (occurrences per term-year) from kw_years and top_terms
        df_top = (
            kw_years[kw_years['term'].isin(top_terms)]
            .groupby(['term', 'start_year'])
            .size()
            .reset_index(name='occurrences')
            .sort_values(['term', 'start_year'])
        )
    else:
        # If top_n is 0 and no custom keywords, no individual lines will be shown
        df_top = pd.DataFrame()
    
    # Choose data source for average calculation
    if show_average:
        if average_from_population:
            # Use all terms that meet min_count for average calculation
            df_population = (
                kw_years
                .groupby(['term', 'start_year'])
                .size()
                .reset_index(name='occurrences')
                .sort_values(['term', 'start_year'])
            )
            df_avg = df_population
        else:
            # Use sample (top_n) for average calculation
            if df_top.empty:
                # If no sample data, fall back to population
                df_avg = (
                    kw_years
                    .groupby(['term', 'start_year'])
                    .size()
                    .reset_index(name='occurrences')
                    .sort_values(['term', 'start_year'])
                )
            else:
                df_avg = df_top
    
    if show_average and df_avg.empty:
        return None
    
    # Create matrices for visualization
    # Matrix for individual lines (custom keywords or top_n keywords)
    if (custom_keywords or top_n > 0) and not df_top.empty:
        years = np.arange(df_top['start_year'].min(), df_top['start_year'].max() + 1)
        occ_matrix_display = (
            df_top
            .groupby(['start_year', 'term'])['occurrences']
            .sum()
            .unstack(fill_value=0)
            .reindex(index=years, fill_value=0)
        )
        if use_cumulative:
            occ_matrix_display = occ_matrix_display.cumsum()
    else:
        occ_matrix_display = pd.DataFrame()
    
    # Calculate baseline: cumulative average keywords per grant per year
    baseline_data = None
    if show_baseline:
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
        
        # Determine year range for baseline
        if show_average and not df_avg.empty:
            baseline_min_year = df_avg['start_year'].min()
            baseline_max_year = df_avg['start_year'].max()
        elif (custom_keywords or top_n > 0) and not df_top.empty:
            baseline_min_year = df_top['start_year'].min()
            baseline_max_year = df_top['start_year'].max()
        else:
            baseline_min_year = yearly_baseline['start_year'].min()
            baseline_max_year = yearly_baseline['start_year'].max()
            baseline_max_year = yearly_baseline['start_year'].max()
        
        baseline_years = np.arange(baseline_min_year, baseline_max_year + 1)
        
        # Reindex to match year range and conditionally calculate cumulative sum
        baseline_series = (
            yearly_baseline.set_index('start_year')['keywords_per_grant']
            .reindex(baseline_years, fill_value=0)
        )
        if use_cumulative:
            baseline_series = baseline_series.cumsum()
        baseline_data = baseline_series
    
    # Matrix for average calculation
    if show_average:
        years_avg = np.arange(df_avg['start_year'].min(), df_avg['start_year'].max() + 1)
        occ_matrix_avg = (
            df_avg
            .groupby(['start_year', 'term'])['occurrences']
            .sum()
            .unstack(fill_value=0)
            .reindex(index=years_avg, fill_value=0)
        )
        if use_cumulative:
            occ_matrix_avg = occ_matrix_avg.cumsum()
    
    # Create the figure using graph_objects for more control
    fig_cum = go.Figure()
    
    # Add individual keyword traces only if we have display data (custom keywords or top_n > 0)
    if (custom_keywords or top_n > 0) and not occ_matrix_display.empty:
        for term in occ_matrix_display.columns:
            fig_cum.add_trace(go.Scatter(
                x=occ_matrix_display.index,
                y=occ_matrix_display[term],
                mode='lines+markers',
                name=term,
                opacity=0.6,
                line=dict(width=1.5),
                marker=dict(size=4)
            ))
    
    # Add average trend with error bars if requested
    if show_average:
        # Calculate statistics across keywords for each year
        avg_occurrences = occ_matrix_avg.mean(axis=1)
        std_occurrences = occ_matrix_avg.std(axis=1)
        sem_occurrences = std_occurrences / np.sqrt(occ_matrix_avg.shape[1])  # Standard error of mean
        
        # Add error bars (standard error) first if requested
        if show_error_bars:
            fig_cum.add_trace(go.Scatter(
                x=occ_matrix_avg.index,
                y=avg_occurrences + sem_occurrences,
                mode='lines',
                line=dict(width=0, color='rgba(0,0,0,0)'),
                showlegend=False,
                hoverinfo='skip',
                name='Upper bound'
            ))
            fig_cum.add_trace(go.Scatter(
                x=occ_matrix_avg.index,
                y=avg_occurrences - sem_occurrences,
                mode='lines',
                line=dict(width=0, color='rgba(0,0,0,0)'),
                fill='tonexty',
                fillcolor='rgba(0,0,0,0.1)',
                name='±SE',
                hoverinfo='skip'
            ))
        
        # Add average line on top
        avg_name = 'Average (Population)' if average_from_population else 'Average (Sample)'
        fig_cum.add_trace(go.Scatter(
            x=occ_matrix_avg.index,
            y=avg_occurrences,
            mode='lines+markers',
            name=avg_name,
            line=dict(width=3, color='black'),
            marker=dict(size=6, color='black'),
            opacity=1.0
        ))
    
    # Add baseline (cumulative average keywords per grant per year)
    if show_baseline and baseline_data is not None:
        fig_cum.add_trace(go.Scatter(
            x=baseline_data.index,
            y=baseline_data.values,
            mode='lines+markers',
            name='Baseline (Keywords/Grant/Year)',
            line=dict(width=2, color='red', dash='dash'),
            marker=dict(size=4, color='red'),
            opacity=0.8
        ))
    
    # Update layout
    y_axis_title = 'Cumulative occurrences' if use_cumulative else 'Yearly occurrences'
    
    fig_cum.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title=y_axis_title,
        legend_title='Keyword',
        height=height,
        hovermode='x unified'
    )
    
    return fig_cum


def save_keyword_trends_to_html(
    keywords_df: Optional[pd.DataFrame] = None,
    grants_df: Optional[pd.DataFrame] = None,
    min_count: int = 10,
    top_n: int = 20,
    title: str = "Cumulative keyword occurrences over time — Top keywords",
    output_path: Optional[Path] = None,
    show_average: bool = True,
    show_error_bars: bool = True,
    average_from_population: bool = False,
    show_baseline: bool = False,
    custom_keywords: Optional[List[str]] = None
) -> Optional[Path]:
    """
    Create and save keyword trends visualization to HTML file
    
    Args:
        keywords_df: DataFrame with keywords data
        grants_df: DataFrame with grants data
        min_count: Minimum count threshold for keywords to include
        top_n: Number of top keywords to show as individual lines
        title: Title for the visualization
        output_path: Path to save the HTML file (defaults to figures/keyword_trends.html)
        show_average: Whether to show average trend line
        show_error_bars: Whether to show error bars around average
        average_from_population: Whether to compute average from entire population (True) or sample (False)
        show_baseline: Whether to show baseline (average keywords per grant per year)
        custom_keywords: Optional list of specific keywords to track (overrides top_n if provided)
        
    Returns:
        Path to saved HTML file or None if no data
    """
    if output_path is None:
        output_path = Path("figures/keyword_trends.html")
    
    fig = create_keyword_trends_visualization(
        keywords_df=keywords_df,
        grants_df=grants_df,
        min_count=min_count,
        top_n=top_n,
        title=title,
        show_average=show_average,
        show_error_bars=show_error_bars,
        average_from_population=average_from_population,
        show_baseline=show_baseline,
        custom_keywords=custom_keywords,
        use_cumulative=True  # Default to cumulative for backward compatibility
    )
    
    if fig is None:
        return None
    
    # Create figures directory if it doesn't exist
    output_path.parent.mkdir(exist_ok=True)
    
    # Save to HTML
    fig.write_html(str(output_path))
    
    return output_path


def create_treemap_data(
    categories: List[Dict[str, Any]], 
    max_for_classes: Optional[int] = None,
    max_categories_per_for: Optional[int] = None,
    max_keywords_per_category: Optional[int] = None
) -> pd.DataFrame:
    """
    Create hierarchical data structure for treemap visualization with FOR classes as parents
    
    Args:
        categories: List of category data with keywords
        max_for_classes: Maximum number of FOR classes to include (None for all)
        max_categories_per_for: Maximum number of categories per FOR class (None for all)
        max_keywords_per_category: Maximum number of keywords per category (None for all)
        
    Returns:
        pandas.DataFrame: Treemap data with hierarchical structure (FOR Classes → Categories → Keywords)
    """
    treemap_data = []
    
    # Group categories by FOR division
    for_groups = {}
    for category in categories:
        for_code = category.get('for_code', 'Unknown')
        
        # Handle the new format (string) and fallback for any old format (dict)
        if isinstance(for_code, str):
            # New format: just the string value "46"
            for_code_value = for_code
            for_division_name = FORCode.get_name(for_code_value)
        elif isinstance(for_code, dict):
            # Old format: {"code": "46", "name": "..."}
            for_code_value = for_code.get('code', 'Unknown')
            for_division_name = for_code.get('name', f'FOR {for_code_value}')
        else:
            # Fallback for invalid format
            for_code_value = 'Unknown'
            for_division_name = 'Unknown FOR Division'
        
        if for_code_value not in for_groups:
            for_groups[for_code_value] = {
                'name': for_division_name,
                'categories': []
            }
        for_groups[for_code_value]['categories'].append(category)
    
    # Apply FOR class limit by sorting by total keyword count (descending)
    if max_for_classes is not None:
        for_items = []
        for for_code, for_data in for_groups.items():
            total_keywords = sum(len(cat.get('keywords', [])) for cat in for_data['categories'])
            for_items.append((for_code, for_data, total_keywords))
        
        # Sort by keyword count (descending) and limit
        for_items.sort(key=lambda x: x[2], reverse=True)
        for_groups = {item[0]: item[1] for item in for_items[:max_for_classes]}
    
    # Create treemap data with 3-level hierarchy
    for for_code, for_data in for_groups.items():
        for_name = for_data['name']
        
        # Get categories for this FOR (apply category limit if specified)
        categories_for_this_for = for_data['categories']
        if max_categories_per_for is not None:
            category_items = []
            for category in categories_for_this_for:
                keyword_count = len(category.get('keywords', []))
                category_items.append((category, keyword_count))
            
            # Sort by keyword count (descending) and limit
            category_items.sort(key=lambda x: x[1], reverse=True)
            categories_for_this_for = [item[0] for item in category_items[:max_categories_per_for]]
        
        # Calculate total keywords for this FOR division (after limits applied)
        for_keyword_count = 0
        for category in categories_for_this_for:
            keywords = category.get('keywords', [])
            if max_keywords_per_category is not None:
                keywords = keywords[:max_keywords_per_category]
            for_keyword_count += len(keywords)
        
        for_size = max(1, for_keyword_count)
        
        # Level 1: FOR Classes (top level)
        treemap_data.append({
            'id': f"FOR_{for_code}",
            'parent': '',
            'name': f"FOR {for_code}: {for_name}",
            'value': for_size,
            'level': 'FOR Class',
            'item_type': 'for_class'
        })
        
        # Level 2: Categories under FOR classes  
        for category in categories_for_this_for:
            category_name = category['name']
            keywords = category.get('keywords', [])
            
            # Apply keyword limit (take first N keywords)
            if max_keywords_per_category is not None:
                keywords = keywords[:max_keywords_per_category]
            
            # Size by number of keywords (minimum 1)
            category_size = max(1, len(keywords))
            
            treemap_data.append({
                'id': f"FOR_{for_code} >> {category_name}",
                'parent': f"FOR_{for_code}",
                'name': category_name,
                'value': category_size,
                'level': 'Category',
                'item_type': 'category'
            })
            
            # Level 3: Keywords under categories
            for keyword in keywords:
                treemap_data.append({
                    'id': f"FOR_{for_code} >> {category_name} >> KW: {keyword}",
                    'parent': f"FOR_{for_code} >> {category_name}",
                    'name': keyword,
                    'value': 1,
                    'level': 'Keyword',
                    'item_type': 'keyword'
                })

    return pd.DataFrame(treemap_data)


def create_research_landscape_treemap(
    categories: Optional[List[Dict[str, Any]]] = None,
    classification_results: Optional[List[Dict[str, Any]]] = None,
    title: Optional[str] = None,
    height: int = 800,
    font_size: int = 9,
    color_map: Optional[Dict[str, str]] = None,
    max_for_classes: Optional[int] = None,
    max_categories_per_for: Optional[int] = None,
    max_keywords_per_category: Optional[int] = None
) -> Optional[go.Figure]:
    """
    Create a comprehensive treemap visualization of the research landscape
    
    Args:
        categories: Optional pre-loaded categories data
        classification_results: Optional pre-loaded classification results
        title: Optional custom title for the visualization
        height: Height of the visualization in pixels
        font_size: Font size for labels
        color_map: Custom color mapping for hierarchy levels
        category_path: Path to categories file (if categories not provided)
        classification_path: Path to classification file (if results not provided)
        max_for_classes: Maximum number of FOR classes to include (None for all)
        max_categories_per_for: Maximum number of categories per FOR class (None for all)
        max_keywords_per_category: Maximum number of keywords per category (None for all)
        
    Returns:
        plotly.graph_objects.Figure: Interactive treemap visualization or None if no data
    """
    # Load data if not provided
    if categories is None or classification_results is None:
        categories = config.Categories.load().to_dict('records')
        # classification_results = load_classification_results(classification_path)
        classification_results = []  # Placeholder since function is not defined
    # print(categories, classification_results)
    if not categories:
        return None

    # Create treemap data
    treemap_df = create_treemap_data(
        categories, 
        max_for_classes=max_for_classes,
        max_categories_per_for=max_categories_per_for,
        max_keywords_per_category=max_keywords_per_category
    )
    # print(treemap_df)
    if treemap_df.empty:
        return None
    
    # Define default color mapping for 3-level hierarchy
    if color_map is None:
        color_map = {
            'FOR Class': '#1f77b4',      # Blue
            'Category': '#ff7f0e',       # Orange  
            'Keyword': '#2ca02c',        # Green
        }

    # Create title
    if title is None:
        title = 'Research Landscape: FOR Classes → Categories → Keywords'

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

    fig.update_layout(
        font_size=font_size,
        title_font_size=font_size + 7,
        height=height,
        margin=dict(t=60, l=25, r=25, b=25)
    )

    # Update traces for better text visibility
    fig.update_traces(
        textinfo="label",
        textfont_size=font_size,
        textposition="middle center"
    )

    return fig

