#!/usr/bin/env python3
"""
Performance test for the optimized Category Trends functionality
"""

import sys
import time
from pathlib import Path
import pandas as pd

# Add web directory to path
web_dir = str(Path(__file__).parent / "web")
if web_dir not in sys.path:
    sys.path.insert(0, web_dir)

from shared_utils import load_data
import config

def test_optimized_performance():
    """Test the performance of the optimized category trends mapping"""
    print("🚀 Testing Optimized Category Trends Performance")
    print("=" * 55)
    
    # Load data
    print("📊 Loading data...")
    keywords_df, grants_df, categories_df = load_data()
    
    if categories_df is None or categories_df.empty:
        categories_df = config.Categories.load()
    
    if any(df is None or df.empty for df in [keywords_df, grants_df, categories_df]):
        print("❌ Required data not available.")
        return
    
    print(f"✅ Data loaded: {len(categories_df)} categories, {len(keywords_df)} keywords, {len(grants_df)} grants")
    
    # Test with small sample first
    sample_size = 100
    sample_categories = categories_df.head(sample_size)
    
    print(f"\n🔬 Testing with {sample_size} categories...")
    
    # Method 1: Optimized approach
    start_time = time.time()
    
    # Build keyword lookup
    keyword_to_grants = {}
    for _, keyword_row in keywords_df.iterrows():
        keyword_name = keyword_row['name']
        grants_list = keyword_row.get('grants', [])
        if isinstance(grants_list, list):
            keyword_to_grants[keyword_name] = grants_list
    
    # Build grants lookup  
    grants_lookup = {}
    for _, grant_row in grants_df.iterrows():
        grant_id = grant_row['id']
        start_year = grant_row.get('start_year')
        if pd.notna(start_year):
            grants_lookup[grant_id] = int(start_year)
    
    # Process categories
    category_results = {}
    for _, category in sample_categories.iterrows():
        category_name = category['name']
        category_keywords = category.get('keywords', [])
        
        if isinstance(category_keywords, list):
            associated_grants = set()
            for keyword_name in category_keywords:
                if keyword_name in keyword_to_grants:
                    associated_grants.update(keyword_to_grants[keyword_name])
            
            category_results[category_name] = len(associated_grants)
    
    optimized_time = time.time() - start_time
    
    print(f"⚡ Optimized approach: {optimized_time:.2f} seconds")
    print(f"   📈 Processed {len(category_results)} categories")
    print(f"   💰 Found grants for {sum(1 for count in category_results.values() if count > 0)} categories")
    
    if category_results:
        max_grants = max(category_results.values())
        avg_grants = sum(category_results.values()) / len(category_results)
        print(f"   📊 Max grants per category: {max_grants}")
        print(f"   📊 Avg grants per category: {avg_grants:.1f}")
    
    # Performance estimate for full dataset
    full_time_estimate = (optimized_time / sample_size) * len(categories_df)
    print(f"\n📈 Estimated time for full dataset ({len(categories_df)} categories): {full_time_estimate:.1f} seconds")
    
    if full_time_estimate > 30:
        print("⚠️  Full dataset processing may be slow. Consider using smaller batches.")
    else:
        print("✅ Full dataset processing should be reasonably fast.")
    
    print(f"\n🎯 Performance improvements implemented:")
    print(f"   • Pre-built keyword lookup dictionary")
    print(f"   • Pre-built grants temporal lookup")
    print(f"   • Eliminated nested dataframe filtering")
    print(f"   • Added progress tracking")
    print(f"   • Reduced default display count to 5 categories")

if __name__ == "__main__":
    test_optimized_performance()
