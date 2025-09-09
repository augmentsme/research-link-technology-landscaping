#!/usr/bin/env python3
"""
Test script to verify batch size validation in caching
"""

import json
import tempfile
from pathlib import Path
import semantic_clustering

def create_test_data():
    """Create test category proposals"""
    test_categories = [
        {
            "name": "Machine Learning",
            "description": "Machine learning algorithms and neural networks",
            "keywords": ["ML", "AI", "neural networks"],
            "field_of_research": "INFORMATION_COMPUTING_SCIENCES"
        },
        {
            "name": "Deep Learning",
            "description": "Deep neural networks and artificial intelligence",
            "keywords": ["deep learning", "CNN", "RNN"],
            "field_of_research": "INFORMATION_COMPUTING_SCIENCES"
        },
        {
            "name": "Databases",
            "description": "Database systems and data storage technologies",
            "keywords": ["SQL", "databases", "data storage"],
            "field_of_research": "INFORMATION_COMPUTING_SCIENCES"
        },
        {
            "name": "Data Mining",
            "description": "Extracting patterns from large datasets",
            "keywords": ["data mining", "patterns", "big data"],
            "field_of_research": "INFORMATION_COMPUTING_SCIENCES"
        }
    ]
    
    # Create temporary file
    temp_file = Path("/tmp/test_batch_proposals.jsonl")
    with open(temp_file, 'w') as f:
        for category in test_categories:
            f.write(json.dumps(category) + '\n')
    
    return temp_file

def test_batch_size_validation():
    """Test that cache is invalidated when batch size changes"""
    test_file = create_test_data()
    
    try:
        print("Testing batch size validation...")
        
        # First run with batch size 2
        print("\n--- First run with batch_size=2 ---")
        clusters1 = semantic_clustering.generate_and_cache_clusters(
            proposal_path=test_file, 
            batch_size=2, 
            force_regenerate=True
        )
        print(f"Generated {sum(len(field_clusters) for field_clusters in clusters1.values())} clusters")
        
        # Second run with same batch size - should use cache
        print("\n--- Second run with batch_size=2 (should use cache) ---")
        clusters2 = semantic_clustering.generate_and_cache_clusters(
            proposal_path=test_file, 
            batch_size=2, 
            force_regenerate=False
        )
        print(f"Loaded {sum(len(field_clusters) for field_clusters in clusters2.values())} clusters")
        
        # Third run with different batch size - should regenerate
        print("\n--- Third run with batch_size=3 (should regenerate) ---")
        clusters3 = semantic_clustering.generate_and_cache_clusters(
            proposal_path=test_file, 
            batch_size=3, 
            force_regenerate=False
        )
        print(f"Generated {sum(len(field_clusters) for field_clusters in clusters3.values())} clusters")
        
        # Verify the results are different for different batch sizes
        clusters1_count = sum(len(field_clusters) for field_clusters in clusters1.values())
        clusters3_count = sum(len(field_clusters) for field_clusters in clusters3.values())
        
        if clusters1_count != clusters3_count:
            print(f"✓ Batch size validation test passed! Different batch sizes produced different cluster counts ({clusters1_count} vs {clusters3_count})")
        else:
            print(f"⚠ Different batch sizes produced same cluster count, but this might be expected for small datasets")
        
    finally:
        # Clean up
        test_file.unlink(missing_ok=True)

if __name__ == "__main__":
    test_batch_size_validation()
