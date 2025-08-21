"""
Research Landscape Visualizations

This module contains visualization functions for the research link technology 
landscaping analysis, including treemap visualizations of research categories 
and keywords.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
from config import CATEGORY_PATH, RESULTS_DIR


def load_data():
    """
    Load all required data files for visualization
    
    Returns:
        tuple: (categories, classification_results)
    """
    # Load categories data (with keywords)
    categories_data = []
    if CATEGORY_PATH.exists():
        with open(CATEGORY_PATH, 'r') as f:
            data = json.load(f)

            categories_data = data['categories']

    else:
        print(f"⚠️  Categories file not found: {CATEGORY_PATH}")
        print("    Run the categorise task first to generate categories")

    # Try to load classification results if available
    classification_path = RESULTS_DIR / "classification.json"
    classification_results = []

    if classification_path.exists():
        with open(classification_path, 'r') as f:
            classification_results = json.load(f)
        print(f"Found {len(classification_results)} grant classifications")
    else:
        print("No classification results found - run classify task first")

    print(f"Found {len(categories_data)} categories")
    
    return categories_data, classification_results


def create_data_mappings(categories, classification_results):
    """
    Create data mappings for visualization
    
    Args:
        categories: List of category data with keywords
        classification_results: List of grant classification results
        
    Returns:
        tuple: (category_to_grants, summary_stats)
    """
    # Create mapping for grants based on classification results
    category_to_grants = {}

    for result in classification_results:
        # Map grants to categories
        for category_name in result.get('selected_categories', []):
            if category_name not in category_to_grants:
                category_to_grants[category_name] = []
            category_to_grants[category_name].append({
                'title': result['title'],
                'grant_id': result['grant_id']
            })

    # Calculate summary statistics
    total_keywords = sum(len(cat.get('keywords', [])) for cat in categories)
    total_grants = len(classification_results)
    classified_grants = sum(len(grants) for grants in category_to_grants.values())

    summary_stats = {
        'total_categories': len(categories),
        'total_keywords': total_keywords,
        'total_grants': total_grants,
        'classified_grants': classified_grants,
        'avg_keywords_per_category': total_keywords/len(categories) if categories else 0
    }

    print(f"Total keywords: {total_keywords}")
    print(f"Total grants: {total_grants}")
    print(f"Classified grants: {classified_grants}")
    print(f"Average keywords per category: {summary_stats['avg_keywords_per_category']:.1f}")
    
    return category_to_grants, summary_stats


def create_treemap_data(categories):
    """
    Create hierarchical data structure for treemap visualization with FOR classes as parents
    
    Args:
        categories: List of category data with keywords
        
    Returns:
        pandas.DataFrame: Treemap data with hierarchical structure (FOR Classes → Categories → Keywords)
    """
    treemap_data = []
    
    # Group categories by FOR division
    for_groups = {}
    for category in categories:
        for_code = category.get('for_code', 'Unknown')
        for_division_name = category.get('for_division_name', f'FOR {for_code}')
        
        if for_code not in for_groups:
            for_groups[for_code] = {
                'name': for_division_name,
                'categories': []
            }
        for_groups[for_code]['categories'].append(category)
    
    # Create treemap data with 3-level hierarchy
    for for_code, for_data in for_groups.items():
        for_name = for_data['name']
        
        # Calculate total keywords for this FOR division
        for_keyword_count = sum(len(cat.get('keywords', [])) for cat in for_data['categories'])
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
        for category in for_data['categories']:
            category_name = category['name']
            keywords = category.get('keywords', [])
            
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


def create_research_landscape_treemap(categories=None, classification_results=None):
    """
    Create a comprehensive treemap visualization of the research landscape
    
    Args:
        categories: Optional pre-loaded categories data
        classification_results: Optional pre-loaded classification results
        
    Returns:
        plotly.graph_objects.Figure: Interactive treemap visualization or None if no data
    """
    # Load data if not provided
    if any(x is None for x in [categories, classification_results]):
        categories, classification_results = load_data()
    
    # Check if we have enough data for visualization
    if not categories:
        print("\n❌ Insufficient data for visualization:")
        print("   - Missing categories (run 'make categorise' first)")
        print("\nSkipping visualization...")
        return None
    
    # Create data mappings
    category_to_grants, summary_stats = create_data_mappings(categories, classification_results)
    
    # Create treemap data
    treemap_df = create_treemap_data(categories)
    
    # Define color mapping for 3-level hierarchy
    color_map = {
        'FOR Class': '#1f77b4',      # Blue
        'Category': '#ff7f0e',       # Orange  
        'Keyword': '#2ca02c',        # Green
    }

    # Create title
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
        font_size=9,
        title_font_size=16,
        height=800,
        margin=dict(t=60, l=25, r=25, b=25)
    )

    # Update traces for better text visibility
    fig.update_traces(
        textinfo="label",
        textfont_size=9,
        textposition="middle center"
    )

    # Print summary
    for_class_count = len(treemap_df[treemap_df['level'] == 'FOR Class'])
    category_count = len(treemap_df[treemap_df['level'] == 'Category'])
    keyword_count = len(treemap_df[treemap_df['level'] == 'Keyword'])

    print(f"\nTreemap contains {len(treemap_df)} total elements:")
    print(f"  - {for_class_count} FOR Classes (Blue)")
    print(f"  - {category_count} Categories (Orange)")
    print(f"  - {keyword_count} Keywords (Green)")
    
    # Show FOR class distribution
    if for_class_count > 0:
        for_distribution = treemap_df[treemap_df['level'] == 'FOR Class'][['name', 'value']].sort_values('value', ascending=False)
        print(f"\nFOR Class distribution by keyword count:")
        for _, row in for_distribution.head(5).iterrows():
            print(f"  - {row['name']}: {row['value']} keywords")

    print(f"\nClassification summary:")
    if summary_stats['classified_grants'] > 0:
        print(f"Classified grants: {summary_stats['classified_grants']}")
        
        # Show distribution of grants across categories
        category_dist = {k: len(v) for k, v in category_to_grants.items() if v}
        if category_dist:
            print("  Grant distribution by category:")
            for category, count in sorted(category_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    - {category}: {count} grants")

    return fig


def main():
    """
    Main function to generate and display the research landscape visualization
    """
    print("=" * 70)
    print("🔬 Research Landscape Visualization")
    print("=" * 70)
    
    # Create and display the treemap
    fig = create_research_landscape_treemap()
    
    if fig is not None:
        fig.show()
        print("\n" + "=" * 70)
        print("🎉 Visualization completed!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("⚠️  Visualization skipped due to missing data")
        print("=" * 70)
    
    return fig


if __name__ == "__main__":
    main()
