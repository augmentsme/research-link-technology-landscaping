"""
Research Landscape Visualizations

This module contains visualization functions for the research link technology 
landscaping analysis, including treemap visualizations of research categories 
and keywords.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    load_categories,
    load_classification_results,
    load_keywords,
    load_grants
)
from analysis import (
    analyze_keyword_trends,
    extract_grant_id_from_keyword
)
from models import FORCode


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
    category_path: Optional[Path] = None,
    classification_path: Optional[Path] = None,
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
        categories = load_categories(category_path)
        classification_results = load_classification_results(classification_path)
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

# fig, df = create_research_landscape_treemap()
# fig


def create_keyword_trends_visualization(
    trends_analysis: Optional[Dict[str, Any]] = None,
    time_range: Optional[Tuple[int, int]] = None,
    bin_size: int = 5,
    keyword_types: Optional[List[str]] = None,
    top_n_terms: int = 20,
    min_frequency: int = 2,
    show_funding: bool = False,
    height: int = 600,
    title: Optional[str] = None,
    keywords_path: Optional[Path] = None,
    grants_path: Optional[Path] = None,
    enable_type_toggle: bool = True
) -> go.Figure:
    """
    Create visualization of keyword trends over time with optional keyword type toggling
    
    Args:
        trends_analysis: Pre-computed trends analysis. If None, will compute it.
        time_range: (start_year, end_year) tuple
        bin_size: Size of time bins in years
        keyword_types: List of keyword types to include
        top_n_terms: Number of top terms to visualize
        min_frequency: Minimum frequency for inclusion
        show_funding: Whether to show funding information in hover data
        height: Height of the visualization in pixels
        title: Custom title for the visualization
        keywords_path: Path to keywords file
        grants_path: Path to grants file
        enable_type_toggle: Whether to enable interactive keyword type filtering
        
    Returns:
        plotly.graph_objects.Figure: Interactive visualization
        
    Raises:
        ValueError: If no data is available
    """

    # Get trends analysis for all types if toggling is enabled, otherwise use specified types
    analysis_keyword_types = None if enable_type_toggle else keyword_types
    
    if trends_analysis is None:
        trends_analysis = analyze_keyword_trends(
            time_range=time_range,
            bin_size=bin_size,
            keyword_types=analysis_keyword_types,
            top_n_terms=top_n_terms,
            min_frequency=min_frequency,
            keywords_path=keywords_path,
            grants_path=grants_path
        )
    
    trends_df = trends_analysis['trends_data']
    summary_stats = trends_analysis['summary_stats']
    
    # Get available keyword types for toggle functionality
    available_types = list(summary_stats.get('keyword_types', {}).keys())
    
    # Filter data by keyword types if specified and not using toggle
    if keyword_types and not enable_type_toggle:
        trends_df = trends_df[trends_df['type'].isin(keyword_types)]
    
    # Prepare title and subtitle
    if title is None:
        title = f"Keyword Trends Over Time ({summary_stats['time_span']})"
        if keyword_types and not enable_type_toggle:
            title += f" - Types: {', '.join(keyword_types)}"
        elif enable_type_toggle:
            title += " - Interactive Type Filtering"
    
    subtitle = (f"Top {top_n_terms} terms, {bin_size}-year bins, "
               f"{summary_stats['total_keywords']} total keywords from "
               f"{summary_stats['unique_grants']} grants")
    
    # Prepare hover data
    hover_data = ['frequency', 'type']
    if show_funding:
        hover_data.append('total_funding')
    
    # Create the base area chart
    fig = px.area(
        trends_df,
        x='time_bin',
        y='count',
        color='term',
        title=title,
        labels={'count': 'Keyword Frequency', 'time_bin': 'Time Period'},
        hover_data=hover_data
    )

    # Update layout
    fig.update_layout(
        title={
            'text': f"{title}<br><sub>{subtitle}</sub>",
            'x': 0.5,
            'xanchor': 'center'
        },
        height=height,
        hovermode='x unified'
    )
    
    # Add interactive keyword type filtering if enabled
    if enable_type_toggle and available_types:
        # Create dropdown buttons for each keyword type
        buttons = []
        
        # Add "All Types" button
        buttons.append(
            dict(
                label="All Types",
                method="restyle",
                args=[{"visible": [True] * len(fig.data)}]
            )
        )
        
        # Add button for each keyword type
        for kw_type in available_types:
            # Create visibility array - show traces where type matches
            visibility = []
            for trace in fig.data:
                # Check if any data point in this trace matches the keyword type
                trace_data = trends_df[trends_df['term'] == trace.name]
                has_type = kw_type in trace_data['type'].values
                visibility.append(has_type)
            
            buttons.append(
                dict(
                    label=f"{kw_type.title()} ({summary_stats['keyword_types'].get(kw_type, 0)})",
                    method="restyle",
                    args=[{"visible": visibility}]
                )
            )
        
        # Add dropdown menu to layout
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=0.1,
                    y=1.02,
                    xanchor="left",
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1
                )
            ]
        )
        
        # Add annotation for the dropdown
        fig.add_annotation(
            text="Filter by Keyword Type:",
            x=0.02,
            y=1.02,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)"
        )
    
    return fig







