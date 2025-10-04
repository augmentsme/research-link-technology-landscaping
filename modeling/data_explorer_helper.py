"""
Data Explorer Helper - Unified module for data exploration preparation

This module provides helper methods to prepare data and configurations for
the DataExplorer component across different entity types (keywords, grants, categories).

Usage:
    from data_explorer_helper import DataExplorerPreparation
    from visualisation import DataExplorer
    
    # Prepare data and config
    display_data, config = DataExplorerPreparation.prepare_keywords_data(filtered_keywords_df)
    
    # Render explorer
    explorer = DataExplorer()
    explorer.render_explorer(display_data, config)
"""

import pandas as pd
from typing import Dict, List, Tuple
from visualisation import DataExplorerConfig


class DataExplorerPreparation:
    """
    Helper class for preparing data and configurations for DataExplorer component
    
    This class provides static methods that handle data preparation and configuration
    creation for different entity types, reducing boilerplate code in page classes.
    """
    
    @staticmethod
    def prepare_keywords_data(filtered_keywords: pd.DataFrame) -> Tuple[pd.DataFrame, DataExplorerConfig]:
        """
        Prepare keywords data and configuration for DataExplorer
        
        Args:
            filtered_keywords: Filtered DataFrame with keyword data
            
        Returns:
            Tuple of (prepared_dataframe, explorer_config)
        """
        display_df = filtered_keywords.copy()
        
        # Ensure 'name' is a column, not the index
        if 'name' not in display_df.columns and display_df.index.name == 'name':
            display_df = display_df.reset_index()
        
        # Add grant count column
        display_df['grant_count'] = display_df['grants'].apply(len)
        
        # Sort by grant count in descending order
        display_df = display_df.sort_values('grant_count', ascending=False)
        
        # Create configuration
        config = DataExplorerConfig(
            title="Keywords Dataset",
            description="Explore the complete keywords dataset with filtering, search capabilities, and distribution analysis.",
            search_columns=['name', 'type'],
            search_placeholder="Search in keyword names or types...",
            search_help="Enter text to search for specific keywords by name or type",
            display_columns=['name', 'type', 'grant_count', 'grants'],
            column_formatters={
                'grants': lambda x: ', '.join(str(g) for g in x[:3]) + ('...' if len(x) > 3 else '')
            },
            column_renames={
                'name': 'Keyword Name',
                'type': 'Type',
                'grant_count': 'Grant Count',
                'grants': 'Associated Grants'
            },
            statistics_columns=['grant_count'],
            max_display_rows=100,
            show_statistics=True,
            show_data_info=True,
            enable_download=True
        )
        
        return display_df, config
    
    @staticmethod
    def prepare_grants_data(grants_df: pd.DataFrame) -> Tuple[pd.DataFrame, DataExplorerConfig]:
        """
        Prepare grants data for DataExplorer
        
        Args:
            grants_df: DataFrame with grant data
            
        Returns:
            Tuple of (prepared_dataframe, explorer_config)
        """
        if grants_df is None or grants_df.empty:
            return pd.DataFrame(), DataExplorerConfig(
                title="Grants Dataset",
                description="No grants data available",
                search_columns=[],
                display_columns=[]
            )
        
        display_df = grants_df.copy()
        
        # Sort by funding amount and year
        display_df = display_df.sort_values(['funding_amount', 'start_year'], ascending=[False, False])
        
        # Create configuration
        config = DataExplorerConfig(
            title="Grants Dataset",
            description="Explore grants data. Search by grant title, funder, or source.",
            search_columns=['title', 'funder', 'source'],
            search_placeholder="Search by grant title, funder, or source...",
            search_help="Enter text to search across grant titles, funders, and sources",
            display_columns=['title', 'funder', 'source', 'start_year', 'funding_amount'],
            column_formatters={
                'funding_amount': lambda x: f"${x:,.0f}" if pd.notna(x) and x > 0 else "N/A",
                'title': lambda x: x[:80] + "..." if len(str(x)) > 80 else str(x)
            },
            column_renames={
                'title': 'Grant Title',
                'funder': 'Funder',
                'source': 'Source',
                'start_year': 'Start Year',
                'funding_amount': 'Funding Amount'
            },
            statistics_columns=['funding_amount', 'start_year'],
            max_display_rows=50,
            show_statistics=True,
            show_data_info=True,
            enable_download=True
        )
        
        return display_df, config
    
    @staticmethod
    def prepare_categories_data(filtered_categories: pd.DataFrame) -> Tuple[pd.DataFrame, DataExplorerConfig]:
        """
        Prepare categories data and configuration for DataExplorer
        
        Args:
            filtered_categories: Filtered DataFrame with category data
            
        Returns:
            Tuple of (prepared_dataframe, explorer_config)
        """
        display_df = filtered_categories[['name', 'field_of_research', 'keyword_count', 'description', 'keywords']].copy()
        display_df = display_df.sort_values('keyword_count', ascending=False)
        
        # Format keywords for display and search
        def format_keywords(keywords):
            if isinstance(keywords, list):
                keywords_str = ", ".join(keywords)
                if len(keywords_str) > 100:
                    return keywords_str[:97] + "..."
                return keywords_str
            else:
                return str(keywords) if keywords else ""
        
        display_df['keywords_formatted'] = display_df['keywords'].apply(format_keywords)
        
        # Create configuration
        config = DataExplorerConfig(
            title="All Categories",
            description="Search and explore research categories with their keywords and descriptions.",
            search_columns=['name', 'description', 'field_of_research', 'keywords_formatted'],
            search_placeholder="Search by category name, description, field of research, or keywords...",
            search_help="Enter text to search across category names, descriptions, fields, and keywords",
            display_columns=['name', 'field_of_research', 'keyword_count', 'keywords_formatted', 'description'],
            column_formatters={
                'description': lambda x: (x[:150] + "...") if len(str(x)) > 150 else str(x),
                'keywords_formatted': lambda x: x
            },
            column_renames={
                'name': 'Category Name',
                'field_of_research': 'Field of Research',
                'keyword_count': 'Keywords Count',
                'keywords_formatted': 'Keywords',
                'description': 'Description'
            },
            statistics_columns=['keyword_count'],
            max_display_rows=50,
            show_statistics=True,
            show_data_info=True,
            enable_download=True
        )
        
        return display_df, config
