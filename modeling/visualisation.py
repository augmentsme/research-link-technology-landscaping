"""
Research Landscape Visualizations

This module contains visualization functions for the research link technology 
landscaping analysis, including treemap visualizations of research domains, 
categories, and keywords.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
from config import CATEGORY_PATH, REFINED_CATEGORY_PATH, RESULTS_DIR


def load_data():
    """
    Load all required data files for visualization
    
    Returns:
        tuple: (refined_categories, detailed_categories, classification_results)
    """
    # Try to load refined categories data
    refined_categories = []
    if REFINED_CATEGORY_PATH.exists():
        with open(REFINED_CATEGORY_PATH, 'r') as f:
            refined_categories = json.load(f)
    else:
        print(f"⚠️  Refined categories file not found: {REFINED_CATEGORY_PATH}")
        print("    Run the refine task first to generate refined categories")

    # Load detailed categories data (with keywords)
    detailed_categories = []
    if CATEGORY_PATH.exists():
        with open(CATEGORY_PATH, 'r') as f:
            detailed_categories = json.load(f)
    else:
        print(f"⚠️  Detailed categories file not found: {CATEGORY_PATH}")
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

    print(f"Found {len(refined_categories)} refined categories")
    print(f"Found {len(detailed_categories)} detailed categories")
    
    return refined_categories, detailed_categories, classification_results


def create_data_mappings(refined_categories, detailed_categories, classification_results):
    """
    Create data mappings for visualization
    
    Args:
        refined_categories: List of refined category data
        detailed_categories: List of detailed category data with keywords
        classification_results: List of grant classification results
        
    Returns:
        tuple: (detailed_category_map, strategic_category_to_grants, detailed_category_to_grants, summary_stats)
    """
    # Create mapping from detailed category name to keywords
    detailed_category_map = {cat['name']: cat.get('keywords', []) for cat in detailed_categories}

    # Create mappings for grants based on what's available in the classification results
    strategic_category_to_grants = {}  # Strategic level classifications
    detailed_category_to_grants = {}   # Detailed level classifications

    for result in classification_results:
        # Strategic level mapping (always available)
        for category_name in result.get('selected_categories', []):
            if category_name not in strategic_category_to_grants:
                strategic_category_to_grants[category_name] = []
            strategic_category_to_grants[category_name].append({
                'title': result['title'],
                'grant_id': result['grant_id']
            })
        
        # Detailed level mapping (only if subcategories were classified)
        for subcat_name in result.get('selected_subcategories', []):
            if subcat_name not in detailed_category_to_grants:
                detailed_category_to_grants[subcat_name] = []
            detailed_category_to_grants[subcat_name].append({
                'title': result['title'],
                'grant_id': result['grant_id']
            })

    # Calculate summary statistics
    total_subcategories = sum(len(cat['subcategories']) for cat in refined_categories)
    total_keywords = sum(len(cat.get('keywords', [])) for cat in detailed_categories)
    total_grants = len(classification_results)
    strategic_grants = sum(len(grants) for grants in strategic_category_to_grants.values())
    detailed_grants = sum(len(grants) for grants in detailed_category_to_grants.values())

    summary_stats = {
        'total_subcategories': total_subcategories,
        'total_keywords': total_keywords,
        'total_grants': total_grants,
        'strategic_grants': strategic_grants,
        'detailed_grants': detailed_grants,
        'avg_keywords_per_category': total_keywords/len(detailed_categories) if detailed_categories else 0,
        'has_detailed_classifications': detailed_grants > 0
    }

    print(f"Total keywords: {total_keywords}")
    print(f"Total grants: {total_grants}")
    print(f"Strategic level classifications: {strategic_grants}")
    print(f"Detailed level classifications: {detailed_grants}")
    print(f"Average keywords per detailed category: {summary_stats['avg_keywords_per_category']:.1f}")
    print(f"Visualization mode: {'Detailed level' if summary_stats['has_detailed_classifications'] else 'Strategic level'}")
    
    return detailed_category_map, strategic_category_to_grants, detailed_category_to_grants, summary_stats


def create_treemap_data(refined_categories, detailed_category_map):
    """
    Create hierarchical data structure for treemap visualization
    
    Args:
        refined_categories: List of refined category data
        detailed_category_map: Mapping of category names to keywords
        
    Returns:
        pandas.DataFrame: Treemap data with hierarchical structure
    """
    treemap_data = []

    # Level 1: Strategic domains (refined categories)
    for refined_cat in refined_categories:
        # Calculate total keywords for this strategic domain
        domain_keyword_count = 0
        
        for subcat_name in refined_cat['subcategories']:
            domain_keyword_count += len(detailed_category_map.get(subcat_name, []))
        
        # Size by total keywords
        domain_size = max(1, domain_keyword_count)
        
        treemap_data.append({
            'id': refined_cat['name'],
            'parent': '',
            'name': refined_cat['name'],
            'value': domain_size,
            'level': 'Strategic Domain',
            'item_type': 'domain'
        })
        
        # Level 2: Detailed categories (subcategories)
        for subcat_name in refined_cat['subcategories']:
            keywords = detailed_category_map.get(subcat_name, [])
            
            # Size by keywords only
            category_size = max(1, len(keywords))
            
            treemap_data.append({
                'id': f"{refined_cat['name']} >> {subcat_name}",
                'parent': refined_cat['name'],
                'name': subcat_name,
                'value': category_size,
                'level': 'Detailed Category',
                'item_type': 'category'
            })
            
            # Level 3: Keywords under detailed categories
            for keyword in keywords:
                treemap_data.append({
                    'id': f"{refined_cat['name']} >> {subcat_name} >> KW: {keyword}",
                    'parent': f"{refined_cat['name']} >> {subcat_name}",
                    'name': keyword,
                    'value': 1,
                    'level': 'Keyword',
                    'item_type': 'keyword'
                })

    return pd.DataFrame(treemap_data)


def create_research_landscape_treemap(refined_categories=None, detailed_categories=None, classification_results=None):
    """
    Create a comprehensive treemap visualization of the research landscape
    
    Args:
        refined_categories: Optional pre-loaded refined categories data
        detailed_categories: Optional pre-loaded detailed categories data
        classification_results: Optional pre-loaded classification results
        
    Returns:
        plotly.graph_objects.Figure: Interactive treemap visualization or None if no data
    """
    # Load data if not provided
    if any(x is None for x in [refined_categories, detailed_categories, classification_results]):
        refined_categories, detailed_categories, classification_results = load_data()
    
    # Check if we have enough data for visualization
    if not refined_categories or not detailed_categories:
        print("\n❌ Insufficient data for visualization:")
        if not refined_categories:
            print("   - Missing refined categories (run 'make refine' first)")
        if not detailed_categories:
            print("   - Missing detailed categories (run 'make categorise' first)")
        print("\nSkipping visualization...")
        return None
    
    # Create data mappings
    detailed_category_map, strategic_category_to_grants, detailed_category_to_grants, summary_stats = create_data_mappings(
        refined_categories, detailed_categories, classification_results
    )
    
    # Create treemap data
    treemap_df = create_treemap_data(refined_categories, detailed_category_map)
    
    # Define color mapping
    color_map = {
        'Strategic Domain': '#1f77b4',    # Blue
        'Detailed Category': '#ff7f0e',   # Orange  
        'Keyword': '#2ca02c',             # Green
    }

    # Create title
    title = 'Research Landscape: Domains → Categories → Keywords'

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
    keyword_count = len(treemap_df[treemap_df['level'] == 'Keyword'])

    print(f"\nTreemap contains {len(treemap_df)} total elements:")
    print(f"  - {len(refined_categories)} Strategic Domains (Blue)")  
    print(f"  - {summary_stats['total_subcategories']} Detailed Categories (Orange)")
    print(f"  - {keyword_count} Keywords (Green)")

    print(f"\nClassification summary:")
    if summary_stats['strategic_grants'] > 0:
        print(f"Strategic level grants: {summary_stats['strategic_grants']}")
    if summary_stats['detailed_grants'] > 0:
        print(f"Detailed level grants: {summary_stats['detailed_grants']}")

    if summary_stats['strategic_grants'] > 0 or summary_stats['detailed_grants'] > 0:
        strategic_dist = {k: len(v) for k, v in strategic_category_to_grants.items() if v}
        detailed_dist = {k: len(v) for k, v in detailed_category_to_grants.items() if v}
        
        if strategic_dist:
            print("  Strategic level distribution:")
            for domain, count in sorted(strategic_dist.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"    - {domain}: {count} grants")
        
        if detailed_dist:
            print("  Detailed level distribution:")
            for category, count in sorted(detailed_dist.items(), key=lambda x: x[1], reverse=True)[:3]:
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
