"""
Merge Taxonomy Script

This script merges the results from categorise, refine, and coarsen tasks to create
a comprehensive taxonomy that can be used for downstream classification tasks.

The merged taxonomy provides multiple levels of granularity:
- Coarsened categories (high-level strategic view)
- Base categories (medium-level operational view)
- Refined categories (detailed technical view)
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from config import (
    CATEGORY_PATH, 
    REFINED_CATEGORY_PATH, 
    COARSENED_CATEGORY_PATH,
    COMPREHENSIVE_TAXONOMY_PATH,
    RESULTS_DIR
)

@dataclass
class TaxonomyLevel:
    """Represents a taxonomy level with its metadata."""
    level: str  # 'coarsened', 'base', 'refined'
    total_categories: int
    description: str

@dataclass
class CategoryNode:
    """Represents a category node in the taxonomy."""
    id: str
    name: str
    description: str
    level: str  # 'coarsened', 'base', 'refined'
    parent_id: Optional[str] = None
    parent_name: Optional[str] = None
    children_ids: List[str] = None
    children_names: List[str] = None
    key_technologies: List[str] = None
    research_applications: List[str] = None
    subcategories: List[str] = None  # For coarsened categories
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.children_names is None:
            self.children_names = []
        if self.key_technologies is None:
            self.key_technologies = []
        if self.research_applications is None:
            self.research_applications = []
        if self.subcategories is None:
            self.subcategories = []

@dataclass
class ComprehensiveTaxonomy:
    """Complete taxonomy structure with all levels."""
    metadata: Dict[str, Any]
    taxonomy_levels: List[TaxonomyLevel]
    categories: List[CategoryNode]
    
    def get_categories_by_level(self, level: str) -> List[CategoryNode]:
        """Get all categories for a specific level."""
        return [cat for cat in self.categories if cat.level == level]
    
    def get_category_by_id(self, category_id: str) -> Optional[CategoryNode]:
        """Get a category by its ID."""
        for cat in self.categories:
            if cat.id == category_id:
                return cat
        return None
    
    def get_children(self, parent_id: str) -> List[CategoryNode]:
        """Get all children of a parent category."""
        return [cat for cat in self.categories if cat.parent_id == parent_id]
    
    def get_hierarchy_path(self, category_id: str) -> List[str]:
        """Get the full hierarchy path for a category."""
        path = []
        current = self.get_category_by_id(category_id)
        
        while current:
            path.insert(0, current.name)
            if current.parent_id:
                current = self.get_category_by_id(current.parent_id)
            else:
                break
        
        return path

class TaxonomyMerger:
    """Handles the merging of different taxonomy levels."""
    
    def __init__(self):
        self.base_categories: List[Dict] = []
        self.refined_categories: List[Dict] = []
        self.coarsened_categories: List[Dict] = []
    
    def load_data(self) -> None:
        """Load data from all taxonomy files."""
        try:
            # Load base categories
            if CATEGORY_PATH.exists():
                with open(CATEGORY_PATH, 'r', encoding='utf-8') as f:
                    base_data = json.load(f)
                    if isinstance(base_data, dict) and 'categories' in base_data:
                        self.base_categories = base_data['categories']
                    elif isinstance(base_data, list):
                        self.base_categories = base_data
                    else:
                        print(f"Warning: Unexpected format in {CATEGORY_PATH}")
                        self.base_categories = []
            else:
                print(f"Warning: Base categories file not found at {CATEGORY_PATH}")
            
            # Load refined categories
            if REFINED_CATEGORY_PATH.exists():
                with open(REFINED_CATEGORY_PATH, 'r', encoding='utf-8') as f:
                    refined_data = json.load(f)
                    if isinstance(refined_data, dict) and 'categories' in refined_data:
                        self.refined_categories = refined_data['categories']
                    elif isinstance(refined_data, list):
                        self.refined_categories = refined_data
                    else:
                        print(f"Warning: Unexpected format in {REFINED_CATEGORY_PATH}")
                        self.refined_categories = []
            else:
                print(f"Warning: Refined categories file not found at {REFINED_CATEGORY_PATH}")
            
            # Load coarsened categories
            if COARSENED_CATEGORY_PATH.exists():
                with open(COARSENED_CATEGORY_PATH, 'r', encoding='utf-8') as f:
                    coarsened_data = json.load(f)
                    if isinstance(coarsened_data, dict) and 'categories' in coarsened_data:
                        self.coarsened_categories = coarsened_data['categories']
                    elif isinstance(coarsened_data, list):
                        self.coarsened_categories = coarsened_data
                    else:
                        print(f"Warning: Unexpected format in {COARSENED_CATEGORY_PATH}")
                        self.coarsened_categories = []
            else:
                print(f"Warning: Coarsened categories file not found at {COARSENED_CATEGORY_PATH}")
                
        except Exception as e:
            print(f"Error loading taxonomy data: {e}")
            raise
    
    def create_category_mappings(self) -> Dict[str, Dict]:
        """Create mappings between category levels."""
        mappings = {
            'name_to_base': {cat['name']: cat for cat in self.base_categories},
            'refined_to_base': {},
            'base_to_coarsened': {}
        }
        
        # Map refined categories to base categories
        for refined in self.refined_categories:
            parent_name = refined.get('parent_category', '')
            if parent_name:
                mappings['refined_to_base'][refined['name']] = parent_name
        
        # Map base categories to coarsened categories
        for coarsened in self.coarsened_categories:
            subcategories = coarsened.get('subcategories', [])
            for subcat in subcategories:
                mappings['base_to_coarsened'][subcat] = coarsened['name']
        
        return mappings
    
    def generate_unique_id(self, name: str, level: str) -> str:
        """Generate a unique ID for a category."""
        # Clean the name for ID generation
        clean_name = name.lower().replace(' ', '_').replace('-', '_')
        clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '_')
        return f"{level}_{clean_name}"
    
    def merge_taxonomy(self) -> ComprehensiveTaxonomy:
        """Merge all taxonomy levels into a comprehensive structure."""
        mappings = self.create_category_mappings()
        categories = []
        
        # Process coarsened categories (top level)
        coarsened_ids = {}
        for coarsened in self.coarsened_categories:
            coarsened_id = self.generate_unique_id(coarsened['name'], 'coarsened')
            coarsened_ids[coarsened['name']] = coarsened_id
            
            category_node = CategoryNode(
                id=coarsened_id,
                name=coarsened['name'],
                description=coarsened['description'],
                level='coarsened',
                subcategories=coarsened.get('subcategories', [])
            )
            categories.append(category_node)
        
        # Process base categories (middle level)
        base_ids = {}
        for base in self.base_categories:
            base_id = self.generate_unique_id(base['name'], 'base')
            base_ids[base['name']] = base_id
            
            # Find parent coarsened category
            parent_coarsened = None
            parent_coarsened_id = None
            for coarsened_name, subcats in [(c['name'], c.get('subcategories', [])) for c in self.coarsened_categories]:
                if base['name'] in subcats:
                    parent_coarsened = coarsened_name
                    parent_coarsened_id = coarsened_ids.get(coarsened_name)
                    break
            
            category_node = CategoryNode(
                id=base_id,
                name=base['name'],
                description=base['description'],
                level='base',
                parent_id=parent_coarsened_id,
                parent_name=parent_coarsened
            )
            categories.append(category_node)
        
        # Process refined categories (bottom level)
        for refined in self.refined_categories:
            refined_id = self.generate_unique_id(refined['name'], 'refined')
            
            # Find parent base category
            parent_base = refined.get('parent_category', '')
            parent_base_id = base_ids.get(parent_base)
            
            category_node = CategoryNode(
                id=refined_id,
                name=refined['name'],
                description=refined['description'],
                level='refined',
                parent_id=parent_base_id,
                parent_name=parent_base,
                key_technologies=refined.get('key_technologies', []),
                research_applications=refined.get('research_applications', [])
            )
            categories.append(category_node)
        
        # Update children relationships
        for category in categories:
            children = [c for c in categories if c.parent_id == category.id]
            category.children_ids = [c.id for c in children]
            category.children_names = [c.name for c in children]
        
        # Create taxonomy levels metadata
        taxonomy_levels = [
            TaxonomyLevel(
                level='coarsened',
                total_categories=len(self.coarsened_categories),
                description='High-level strategic research domains for overview and planning'
            ),
            TaxonomyLevel(
                level='base',
                total_categories=len(self.base_categories),
                description='Medium-level operational research categories for general classification'
            ),
            TaxonomyLevel(
                level='refined',
                total_categories=len(self.refined_categories),
                description='Detailed technical research areas for specialized analysis'
            )
        ]
        
        # Create metadata
        metadata = {
            'version': '1.0',
            'created_from': ['categorise', 'refine', 'coarsen'],
            'total_categories': len(categories),
            'levels': {level.level: level.total_categories for level in taxonomy_levels},
            'description': 'Comprehensive research taxonomy with multiple granularity levels'
        }
        
        return ComprehensiveTaxonomy(
            metadata=metadata,
            taxonomy_levels=taxonomy_levels,
            categories=categories
        )
    


def main():
    """Main function to create and export merged taxonomy."""
    print("Creating comprehensive taxonomy...")
    
    merger = TaxonomyMerger()
    
    # Load data
    print("Loading taxonomy data...")
    merger.load_data()
    
    print(f"Loaded {len(merger.base_categories)} base categories")
    print(f"Loaded {len(merger.refined_categories)} refined categories")  
    print(f"Loaded {len(merger.coarsened_categories)} coarsened categories")
    
    # Merge taxonomy
    print("Merging taxonomy levels...")
    comprehensive_taxonomy = merger.merge_taxonomy()
    
    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Export comprehensive taxonomy
    with open(COMPREHENSIVE_TAXONOMY_PATH, 'w', encoding='utf-8') as f:
        # Convert dataclasses to dictionaries for JSON serialization
        taxonomy_dict = {
            'metadata': comprehensive_taxonomy.metadata,
            'taxonomy_levels': [asdict(level) for level in comprehensive_taxonomy.taxonomy_levels],
            'categories': [asdict(category) for category in comprehensive_taxonomy.categories]
        }
        json.dump(taxonomy_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Comprehensive taxonomy saved to: {COMPREHENSIVE_TAXONOMY_PATH}")
    
    # Print summary
    print("\n" + "="*50)
    print("TAXONOMY MERGE SUMMARY")
    print("="*50)
    print(f"Total categories: {comprehensive_taxonomy.metadata['total_categories']}")
    
    for level_info in comprehensive_taxonomy.taxonomy_levels:
        print(f"{level_info.level.capitalize()}: {level_info.total_categories} categories")
    
    print(f"\nFile created: comprehensive_taxonomy.json")
    
    return comprehensive_taxonomy

if __name__ == "__main__":
    taxonomy = main()
