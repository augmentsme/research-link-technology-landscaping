"""
Example usage of the generalized PopularityTrendsVisualizer

This shows how the Keywords, Grants, and Categories pages can use the same
underlying visualization engine while maintaining their specific requirements.
"""

import pandas as pd
from visualisation import PopularityTrendsVisualizer, EntityTrendsConfig

# Initialize the visualizer
trends_visualizer = PopularityTrendsVisualizer()

# Example 1: Keywords Page Usage
def create_keyword_trends(keywords_data: pd.DataFrame):
    """Example of how Keywords page would use the visualizer"""
    
    # Prepare data: keywords_data should have columns: ['keyword', 'year', 'grant_count']
    config = EntityTrendsConfig(
        entity_column='keyword',
        time_column='year', 
        value_column='grant_count',
        max_entities=10,
        aggregation_method='sum',
        use_cumulative=True,
        chart_type='line',
        title='Cumulative Keyword Trends Over Time',
        x_axis_label='Year',
        y_axis_label='Cumulative Grant Count'
    )
    
    return trends_visualizer.create_trends_visualization(keywords_data, config)

# Example 2: Grants Page Usage  
def create_grant_distribution(grants_data: pd.DataFrame):
    """Example of how Grants page would use the visualizer"""
    
    # Prepare data: grants_data should have columns: ['funder', 'year', 'grant_count'] 
    config = EntityTrendsConfig(
        entity_column='funder',
        time_column='year',
        value_column='grant_count', 
        max_entities=10,
        aggregation_method='sum',
        use_cumulative=False,
        chart_type='area_stacked',
        show_others_group=True,
        title='Grant Distribution by Year and Funder',
        x_axis_label='Year',
        y_axis_label='Number of Grants'
    )
    
    return trends_visualizer.create_trends_visualization(grants_data, config)

# Example 3: Categories Page Usage
def create_category_trends(categories_data: pd.DataFrame, ranking_metric='grant_count', display_metric='total_funding'):
    """Example of how Categories page would use the visualizer"""
    
    # Prepare data: categories_data should have columns: ['category', 'year', 'grant_count', 'total_funding']
    config = EntityTrendsConfig(
        entity_column='category',
        time_column='year',
        value_column=display_metric,  # What to show on y-axis
        ranking_metric=ranking_metric,  # What to use for selecting top categories
        max_entities=8,
        aggregation_method='sum',
        use_cumulative=True,
        chart_type='line',
        title=f'Category Trends Over Time (Ranked by {ranking_metric}, Showing {display_metric})',
        x_axis_label='Year',
        y_axis_label='Cumulative Funding' if display_metric == 'total_funding' else 'Cumulative Grant Count'
    )
    
    return trends_visualizer.create_trends_visualization(categories_data, config)

# Example data transformation functions that each page would implement
def prepare_keyword_data(keywords_df: pd.DataFrame, grants_df: pd.DataFrame) -> pd.DataFrame:
    """Transform keyword and grant data into the format needed for visualization"""
    # This would contain the pandas operations currently in each page
    # Returns DataFrame with columns: ['keyword', 'year', 'grant_count']
    pass

def prepare_grant_data(grants_df: pd.DataFrame) -> pd.DataFrame:
    """Transform grant data into the format needed for visualization"""
    # This would contain the pandas operations currently in grants page
    # Returns DataFrame with columns: ['funder', 'year', 'grant_count']
    pass

def prepare_category_data(categories_df: pd.DataFrame, keywords_df: pd.DataFrame, grants_df: pd.DataFrame) -> pd.DataFrame:
    """Transform category data into the format needed for visualization"""
    # This would contain the pandas operations currently in categories page  
    # Returns DataFrame with columns: ['category', 'year', 'grant_count', 'total_funding']
    pass