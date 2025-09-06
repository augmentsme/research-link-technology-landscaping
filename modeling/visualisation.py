"""
Research Landscape Visualizations

This module contains visualization functions for the research link technology 
landscaping analysis, including category visualizations and keyword trends over time.
"""
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

import config


def filter_grants_by_attributes(grants_df: pd.DataFrame, 
                               funder_filter: Optional[List[str]] = None,
                               source_filter: Optional[List[str]] = None) -> pd.DataFrame:
    """Filter grants by funder and source attributes."""
    filtered_grants = grants_df.copy()
    
    if funder_filter:
        filtered_grants = filtered_grants[filtered_grants['funder'].isin(funder_filter)]
    
    if source_filter:
        filtered_grants = filtered_grants[filtered_grants['source'].isin(source_filter)]
        
    return filtered_grants


def filter_keywords_by_type(keywords_df: pd.DataFrame,
                           keyword_type_filter: Optional[List[str]] = None) -> pd.DataFrame:
    """Filter keywords by type."""
    if keyword_type_filter:
        return keywords_df[keywords_df['type'].isin(keyword_type_filter)]
    return keywords_df


def create_keyword_year_pairs(keywords_df: pd.DataFrame, grants_df: pd.DataFrame, 
                             min_count: int = 10) -> pd.DataFrame:
    """Create keyword-year pairs from filtered data."""
    # Ensure grants have index set for lookup
    if 'id' in grants_df.columns:
        grants_df = grants_df.set_index('id')
    
    # Filter keywords to only include those with grants in the filtered set
    filtered_keywords = keywords_df.copy()
    filtered_keywords['filtered_grants'] = filtered_keywords['grants'].apply(
        lambda x: [grant_id for grant_id in x if grant_id in grants_df.index]
    )
    filtered_keywords['filtered_count'] = filtered_keywords['filtered_grants'].map(len)
    filtered_keywords = filtered_keywords[filtered_keywords['filtered_count'] > min_count]
    
    # Create start_year mapping for each keyword
    filtered_keywords['start_year'] = filtered_keywords['filtered_grants'].map(
        lambda x: [
            grants_df.loc[i, "start_year"] 
            for i in x 
            if i in grants_df.index and not np.isnan(grants_df.loc[i, "start_year"])
        ]
    )
    
    # Create keyword-year pairs
    kw_years_list = []
    for _, row in filtered_keywords.iterrows():
        # Use 'name' column if it exists, otherwise fall back to index
        if 'name' in row.index:
            term = row['name']
        elif 'term' in row.index:
            term = row['term']
        else:
            term = str(row.name)  # Convert index to string as fallback
        
        for year in row['start_year']:
            kw_years_list.append({'term': term, 'start_year': int(year)})
    
    return pd.DataFrame(kw_years_list)


def create_keyword_trends_data(keywords_df: pd.DataFrame, grants_df: pd.DataFrame, 
                              min_count: int = 10,
                              funder_filter: Optional[List[str]] = None, 
                              source_filter: Optional[List[str]] = None,
                              keyword_type_filter: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Create keyword trends data from keywords and grants dataframes.
    
    Args:
        keywords_df: DataFrame with keywords data
        grants_df: DataFrame with grants data (should have 'start_year' column)
        min_count: Minimum count threshold for keywords to include
        funder_filter: Optional list of funders to filter by
        source_filter: Optional list of sources to filter by
        keyword_type_filter: Optional list of keyword types to filter by
        
    Returns:
        pandas.DataFrame: DataFrame with keyword trends over time
    """
    # Apply filters
    filtered_grants = filter_grants_by_attributes(grants_df, funder_filter, source_filter)
    filtered_keywords = filter_keywords_by_type(keywords_df, keyword_type_filter)
    
    # Create keyword-year pairs
    return create_keyword_year_pairs(filtered_keywords, filtered_grants, min_count)


def prepare_keyword_matrix(kw_years: pd.DataFrame, terms: List[str], 
                          use_cumulative: bool = True) -> pd.DataFrame:
    """Prepare occurrence matrix for selected terms."""
    if kw_years.empty or not terms:
        return pd.DataFrame()
    
    # Filter to selected terms
    filtered_data = kw_years[kw_years['term'].isin(terms)]
    
    # Create matrix
    years = np.arange(filtered_data['start_year'].min(), filtered_data['start_year'].max() + 1)
    occ_matrix = (
        filtered_data
        .groupby(['start_year', 'term'])['start_year']
        .count()
        .unstack(fill_value=0)
        .reindex(index=years, fill_value=0)
    )
    
    # Ensure column names are strings (term names)
    occ_matrix.columns.name = None  # Remove the 'term' name from columns
    
    if use_cumulative:
        occ_matrix = occ_matrix.cumsum()
    
    return occ_matrix


def add_individual_traces(fig: go.Figure, occ_matrix: pd.DataFrame) -> None:
    """Add individual keyword traces to the figure."""
    for term in occ_matrix.columns:
        # Ensure term is treated as string for display
        term_name = str(term)
        fig.add_trace(go.Scatter(
            x=occ_matrix.index,
            y=occ_matrix[term],
            mode='lines+markers',
            name=term_name,
            opacity=0.6,
            line=dict(width=1.5),
            marker=dict(size=4)
        ))


def add_average_trace(fig: go.Figure, occ_matrix: pd.DataFrame, 
                     show_error_bars: bool = True) -> None:
    """Add average trend line with optional error bars."""
    if occ_matrix.empty:
        return
        
    avg_occurrences = occ_matrix.mean(axis=1)
    std_occurrences = occ_matrix.std(axis=1)
    sem_occurrences = std_occurrences / np.sqrt(occ_matrix.shape[1])
    
    # Add error bars if requested
    if show_error_bars:
        fig.add_trace(go.Scatter(
            x=occ_matrix.index,
            y=avg_occurrences + sem_occurrences,
            mode='lines',
            line=dict(width=0, color='rgba(0,0,0,0)'),
            showlegend=False,
            hoverinfo='skip',
            name='Upper bound'
        ))
        fig.add_trace(go.Scatter(
            x=occ_matrix.index,
            y=avg_occurrences - sem_occurrences,
            mode='lines',
            line=dict(width=0, color='rgba(0,0,0,0)'),
            fill='tonexty',
            fillcolor='rgba(0,0,0,0.1)',
            name='±SE',
            hoverinfo='skip'
        ))
    
    # Add average line
    fig.add_trace(go.Scatter(
        x=occ_matrix.index,
        y=avg_occurrences,
        mode='lines+markers',
        name='Average',
        line=dict(width=3, color='black'),
        marker=dict(size=6, color='black'),
        opacity=1.0
    ))


def create_keyword_trends_visualization(
    keywords_df: Optional[pd.DataFrame] = None,
    grants_df: Optional[pd.DataFrame] = None,
    min_count: int = 10,
    top_n: int = 20,
    title: str = "Keyword Trends Over Time",
    height: int = 600,
    show_average: bool = True,
    show_error_bars: bool = True,
    custom_keywords: Optional[List[str]] = None,
    funder_filter: Optional[List[str]] = None,
    source_filter: Optional[List[str]] = None,
    keyword_type_filter: Optional[List[str]] = None,
    use_cumulative: bool = True
) -> Optional[go.Figure]:
    """
    Create keyword trends visualization.
    
    Args:
        keywords_df: DataFrame with keywords data
        grants_df: DataFrame with grants data
        min_count: Minimum count threshold for keywords to include
        top_n: Number of top keywords to show as individual lines
        title: Title for the visualization
        height: Height of the visualization in pixels
        show_average: Whether to show average trend line
        show_error_bars: Whether to show error bars around average
        custom_keywords: List of specific keywords to display
        funder_filter: Filter by specific funders
        source_filter: Filter by specific sources  
        keyword_type_filter: Filter by keyword types
        use_cumulative: If True, show cumulative counts; if False, show yearly counts
    
    Returns:
        Plotly figure object or None if no data available
    """
    # Load default data if not provided
    if keywords_df is None:
        keywords_df = config.Keywords.load()
    if grants_df is None:
        grants_df = config.Grants.load()
    
    # Create keyword trends data
    kw_years = create_keyword_trends_data(
        keywords_df, grants_df, min_count, 
        funder_filter, source_filter, keyword_type_filter
    )
    
    if kw_years.empty:
        return None
    
    # Determine which keywords to display
    if custom_keywords:
        # Use specified keywords that exist in data
        available_keywords = kw_years['term'].value_counts()
        display_terms = [kw for kw in custom_keywords if kw in available_keywords and available_keywords[kw] >= min_count]
    else:
        # Use top N keywords by occurrence count
        display_terms = kw_years['term'].value_counts().head(top_n).index.tolist()
    
    # Create the figure
    fig = go.Figure()
    
    # Add individual keyword traces if we have terms to display
    if display_terms:
        occ_matrix = prepare_keyword_matrix(kw_years, display_terms, use_cumulative)
        add_individual_traces(fig, occ_matrix)
        
        # Add average if requested
        if show_average:
            add_average_trace(fig, occ_matrix, show_error_bars)
    
    # Update layout
    y_axis_title = 'Cumulative occurrences' if use_cumulative else 'Yearly occurrences'
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title=y_axis_title,
        legend_title='Keyword',
        height=height,
        hovermode='x unified'
    )
    
    return fig


def save_keyword_trends_to_html(
    keywords_df: Optional[pd.DataFrame] = None,
    grants_df: Optional[pd.DataFrame] = None,
    min_count: int = 10,
    top_n: int = 20,
    title: str = "Keyword Trends Over Time",
    output_path: Optional[Path] = None,
    show_average: bool = True,
    show_error_bars: bool = True,
    custom_keywords: Optional[List[str]] = None
) -> Optional[Path]:
    """
    Create and save keyword trends visualization to HTML file.
    
    Args:
        keywords_df: DataFrame with keywords data
        grants_df: DataFrame with grants data
        min_count: Minimum count threshold for keywords to include
        top_n: Number of top keywords to show as individual lines
        title: Title for the visualization
        output_path: Path to save the HTML file (defaults to figures/keyword_trends.html)
        show_average: Whether to show average trend line
        show_error_bars: Whether to show error bars around average
        custom_keywords: Optional list of specific keywords to track
        
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
        custom_keywords=custom_keywords,
        use_cumulative=True
    )
    
    if fig is None:
        return None
    
    # Create figures directory if it doesn't exist
    output_path.parent.mkdir(exist_ok=True)
    
    # Save to HTML
    fig.write_html(str(output_path))
    
    return output_path


def create_simple_treemap_data(categories: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create simple hierarchical data structure for treemap visualization.
    
    Args:
        categories: List of category data with keywords
        
    Returns:
        pandas.DataFrame: Treemap data with simple hierarchy (Categories → Keywords)
    """
    treemap_data = []
    
    # Add root
    total_keywords = sum(len(cat.get('keywords', [])) for cat in categories)
    treemap_data.append({
        'id': 'Root',
        'parent': '',
        'name': 'All Categories',
        'value': total_keywords,
        'level': 'Root',
        'item_type': 'root'
    })
    
    # Add categories and their keywords
    for category in categories:
        category_name = category['name']
        keywords = category.get('keywords', [])
        
        # Add category
        treemap_data.append({
            'id': category_name,
            'parent': 'Root',
            'name': category_name,
            'value': len(keywords),
            'level': 'Category',
            'item_type': 'category'
        })
        
        # Add keywords under category
        for keyword in keywords:
            treemap_data.append({
                'id': f"{category_name} >> {keyword}",
                'parent': category_name,
                'name': keyword,
                'value': 1,
                'level': 'Keyword',
                'item_type': 'keyword'
            })

    return pd.DataFrame(treemap_data)


def create_research_landscape_treemap(
    categories: Optional[List[Dict[str, Any]]] = None,
    title: Optional[str] = None,
    height: int = 800,
    font_size: int = 9,
    color_map: Optional[Dict[str, str]] = None
) -> Optional[go.Figure]:
    """
    Create a treemap visualization of the research landscape.
    
    Args:
        categories: Optional pre-loaded categories data
        title: Optional custom title for the visualization
        height: Height of the visualization in pixels
        font_size: Font size for labels
        color_map: Custom color mapping for hierarchy levels
        
    Returns:
        plotly.graph_objects.Figure: Interactive treemap visualization or None if no data
    """
    # Load data if not provided
    if categories is None:
        categories = config.Categories.load().to_dict('records')
    
    if not categories:
        return None

    # Create treemap data
    treemap_df = create_simple_treemap_data(categories)
    
    if treemap_df.empty:
        return None
    
    # Define default color mapping
    if color_map is None:
        color_map = {
            'Root': '#1f77b4',      # Blue
            'Category': '#ff7f0e',   # Orange  
            'Keyword': '#2ca02c',    # Green
        }

    # Create title
    if title is None:
        title = 'Research Landscape: Categories → Keywords'

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


def create_category_hierarchy_visualization(categories_df: pd.DataFrame, 
                                          keyword_to_category_mapping: Dict[str, str],
                                          height: int = 800, font_size: int = 12,
                                          max_keywords_per_category: int = 50) -> go.Figure:
    """
    Create a hierarchical visualization of categories and their keywords.
    
    Args:
        categories_df: DataFrame with category data
        keyword_to_category_mapping: Dictionary mapping keywords to categories
        height: Height of the visualization
        font_size: Font size for labels
        max_keywords_per_category: Maximum number of keywords to show per category
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Calculate keyword counts per category
    category_counts = {}
    for category in keyword_to_category_mapping.values():
        category_counts[category] = category_counts.get(category, 0) + 1
    
    # Create treemap data - categories as root level with keywords as children
    labels = []
    parents = []
    values = []
    
    # Add categories directly as root elements
    for category_name, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        labels.append(category_name)
        parents.append("")  # Empty string means it's a root element
        values.append(count)
    
    # Add keywords as children under their respective categories
    # Group keywords by category and limit the number shown
    keywords_by_category = {}
    for keyword, category in keyword_to_category_mapping.items():
        if category not in keywords_by_category:
            keywords_by_category[category] = []
        keywords_by_category[category].append(keyword)
    
    # Add limited keywords for each category
    total_keywords_shown = 0
    for category, keywords in keywords_by_category.items():
        # Sort keywords alphabetically and take the first max_keywords_per_category
        sorted_keywords = sorted(keywords)[:max_keywords_per_category]
        for keyword in sorted_keywords:
            labels.append(keyword)
            parents.append(category)
            values.append(1)  # Each keyword has a value of 1
            total_keywords_shown += 1
    
    # Create the treemap
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        textinfo="label+value",
        textfont_size=font_size,
        marker_colorscale=px.colors.qualitative.Set3,
        marker_showscale=False,
        hovertemplate="<b>%{label}</b><br>Value: %{value}<extra></extra>",
        maxdepth=3,  # Allow 3 levels: root -> categories -> keywords
        branchvalues="total",  # Use total values for parent nodes
    ))
    
    fig.update_layout(
        title="Research Category Hierarchy",
        title_font_size=font_size + 6,
        height=height,
        margin=dict(t=60, l=25, r=25, b=25)
    )
    
    return fig


def create_category_distribution_chart(keyword_to_category_mapping: Dict[str, str],
                                     chart_type: str = "bar") -> go.Figure:
    """
    Create a distribution chart showing keyword counts per category.
    
    Args:
        keyword_to_category_mapping: Dictionary mapping keywords to categories
        chart_type: Type of chart ("bar", "pie", "horizontal_bar")
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Calculate keyword counts per category
    category_counts = {}
    for category in keyword_to_category_mapping.values():
        category_counts[category] = category_counts.get(category, 0) + 1
    
    # Sort by count
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    category_names = [item[0] for item in sorted_categories]
    counts = [item[1] for item in sorted_categories]
    
    if chart_type == "pie":
        fig = go.Figure(data=[go.Pie(
            labels=category_names,
            values=counts,
            textinfo='label+percent+value',
            hovertemplate="<b>%{label}</b><br>Keywords: %{value}<br>Percentage: %{percent}<extra></extra>"
        )])
        fig.update_layout(title="Distribution of Keywords Across Categories")
        
    elif chart_type == "horizontal_bar":
        fig = go.Figure(data=[go.Bar(
            x=counts,
            y=category_names,
            orientation='h',
            text=counts,
            textposition='auto',
        )])
        fig.update_layout(
            title="Keywords per Category",
            xaxis_title="Number of Keywords",
            yaxis_title="Category",
            height=max(400, len(category_names) * 25)
        )
        
    else:  # default bar chart
        fig = go.Figure(data=[go.Bar(
            x=category_names,
            y=counts,
            text=counts,
            textposition='auto',
        )])
        fig.update_layout(
            title="Keywords per Category",
            xaxis_title="Category",
            yaxis_title="Number of Keywords",
            xaxis_tickangle=-45
        )
    
    return fig

