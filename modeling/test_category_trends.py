#!/usr/bin/env python3
"""
Test script for Category Trends functionality
Demonstrates how categories are mapped to grants over time
"""

import sys
from pathlib import Path

# Add web directory to path
web_dir = str(Path(__file__).parent / "web")
if web_dir not in sys.path:
    sys.path.insert(0, web_dir)

from shared_utils import load_data
import config
import pandas as pd

def test_category_trends():
    """Test the category trends mapping functionality"""
    print("🔬 Testing Category Trends Functionality")
    print("=" * 50)
    
    # Load data
    print("📊 Loading data...")
    keywords_df, grants_df, categories_df = load_data()
    
    if categories_df is None or categories_df.empty:
        print("❌ No categories data found. Loading from config...")
        categories_df = config.Categories.load()
    
    if any(df is None or df.empty for df in [keywords_df, grants_df, categories_df]):
        print("❌ Required data not available. Please ensure you have:")
        print("  - Keywords data in results/keywords/keywords.jsonl")
        print("  - Grants data in results/grants/grants.jsonl") 
        print("  - Categories data in results/category/category_proposal.jsonl")
        return
    
    print(f"✅ Data loaded successfully!")
    print(f"   📋 Categories: {len(categories_df)}")
    print(f"   🔑 Keywords: {len(keywords_df)}")
    print(f"   💰 Grants: {len(grants_df)}")
    
    # Test category-to-grants mapping
    print("\n🔗 Testing Category-to-Grants Mapping")
    print("-" * 40)
    
    # Take first few categories as examples
    sample_categories = categories_df.head(3)
    
    for idx, category in sample_categories.iterrows():
        category_name = category['name']
        category_keywords = category.get('keywords', [])
        
        print(f"\n📂 Category: {category_name}")
        print(f"   🏷️  Keywords: {len(category_keywords)} keywords")
        
        if not isinstance(category_keywords, list):
            print("   ⚠️  Keywords not in list format")
            continue
            
        # Find grants through keywords
        associated_grants = set()
        found_keywords = 0
        
        for keyword_name in category_keywords[:5]:  # Check first 5 keywords
            keyword_rows = keywords_df[keywords_df['name'] == keyword_name]
            if not keyword_rows.empty:
                found_keywords += 1
                grants_list = keyword_rows.iloc[0].get('grants', [])
                if isinstance(grants_list, list):
                    associated_grants.update(grants_list)
        
        print(f"   🎯 Found keywords in data: {found_keywords}/{min(5, len(category_keywords))}")
        print(f"   💰 Associated grants: {len(associated_grants)}")
        
        # Get temporal distribution
        if associated_grants:
            grant_years = []
            for grant_id in list(associated_grants)[:10]:  # Check first 10 grants
                grant_rows = grants_df[grants_df['id'] == grant_id]
                if not grant_rows.empty:
                    start_year = grant_rows.iloc[0].get('start_year')
                    if pd.notna(start_year):
                        grant_years.append(int(start_year))
            
            if grant_years:
                print(f"   📅 Year range: {min(grant_years)} - {max(grant_years)}")
                year_counts = pd.Series(grant_years).value_counts().sort_index()
                top_years = year_counts.head(3)
                print(f"   📈 Top years: {dict(top_years)}")
    
    print("\n✅ Category Trends functionality test completed!")
    print("\n💡 To see the full visualization, run:")
    print("   uv run streamlit run web/pages/4_Categories.py")
    print("   Then navigate to the 'Category Trends' tab")

if __name__ == "__main__":
    test_category_trends()
