#!/usr/bin/env python3
"""
Test the keywords display functionality in Category Explorer
"""

import pandas as pd

def test_keywords_formatting():
    """Test the keywords formatting function"""
    print("🧪 Testing Keywords Formatting in Category Explorer")
    print("=" * 55)
    
    # Test function
    def format_keywords(keywords):
        if isinstance(keywords, list):
            # Join keywords with commas, but truncate if too long
            keywords_str = ", ".join(keywords)
            if len(keywords_str) > 100:  # Truncate long keyword lists
                return keywords_str[:97] + "..."
            return keywords_str
        else:
            return str(keywords) if keywords else ""
    
    # Test cases
    test_cases = [
        # Short list
        ["keyword1", "keyword2", "keyword3"],
        
        # Long list that should be truncated
        [f"keyword_{i}" for i in range(20)],
        
        # Empty list
        [],
        
        # Non-list input
        "not a list",
        
        # None input
        None
    ]
    
    print("Test Results:")
    print("-" * 30)
    
    for i, test_case in enumerate(test_cases, 1):
        result = format_keywords(test_case)
        print(f"Test {i}: {type(test_case).__name__}")
        print(f"  Input: {str(test_case)[:50]}{'...' if len(str(test_case)) > 50 else ''}")
        print(f"  Output: {result}")
        print(f"  Length: {len(result)}")
        print()
    
    # Test with realistic category data
    print("📊 Testing with Realistic Data:")
    print("-" * 35)
    
    sample_data = {
        'name': ['Category A', 'Category B', 'Category C'],
        'keywords': [
            ['spin coherence', 'quantum magnetism', 'spintronics'],
            [f'keyword_{i}' for i in range(25)],  # Long list
            ['single keyword']
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df['keywords_formatted'] = df['keywords'].apply(format_keywords)
    
    for _, row in df.iterrows():
        print(f"Category: {row['name']}")
        print(f"  Original keywords count: {len(row['keywords'])}")
        print(f"  Formatted: {row['keywords_formatted']}")
        print()
    
    print("✅ Keywords formatting test completed!")

if __name__ == "__main__":
    test_keywords_formatting()
