#!/usr/bin/env python3
"""
Research Link Technology Landscaping - Unified CLI Driver

This is the main command-line interface that orchestrates the entire research analysis pipeline.
It provides a unified entry point for all operations and reads configuration from config.yaml.

The pipeline consists of:
1. Data Management: Loading and managing grant data
2. Keyword Extraction: Extracting keywords from grant summaries  
3. Embedding Generation: Creating vector emb    # Step 1: Extract keywords  
    model_config = {}
    if vllm_url:
        model_config['base_url'] = vllm_url
    
    if not run_inspect_task("extract", "extract", "Extracting keywords from grants", model_config):
        console.print("❌ Keyword extraction failed", style="red")
        return
        
    # Step 2: Generate embeddings
    from database import generate_embeddings_command
    kwargs = {}
    if max_keywords:
        kwargs['max_keywords'] = max_keywords
    if vllm_url:
        kwargs['vllm_url'] = vllm_url
    
    if not run_management_function("database", "generate_embeddings_command", "Generating embeddings", **kwargs):
        console.print("❌ Embedding generation failed", style="red")
        return
        
    console.print("✅ Extraction and embedding pipeline completed!", style="bold green")rds
4. Clustering: Harmonizing and grouping related keywords
5. Categorization: Organizing keywords into research categories
6. Classification: Final classification and taxonomy building
7. Visualization: Creating research landscape visualizations

Usage:
    python cli.py --help                    # Show all available commands
    python cli.py data --help              # Show data management commands
    python cli.py pipeline --help          # Show pipeline commands
    python cli.py config --help            # Show configuration commands
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import typer
import yaml
from dotenv import dotenv_values
# Import inspect_ai for native task evaluation
from inspect_ai import eval
from inspect_ai.model import get_model
from rich import print as rich_print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Import the centralized configuration
from config import Config, CONFIG

# Create Rich console for beautiful output
console = Console()

# Create main Typer app
app = typer.Typer(
    name="cli",
    help="🔬 Research Link Technology Landscaping - Unified CLI Driver",
    rich_markup_mode="rich",
    add_completion=False
)



def run_inspect_task(task_module: str, task_name: str, description: str, model_config: dict = None, **task_kwargs) -> bool:
    """Run an inspect_ai task natively with optional model configuration."""
    console.print(f"🔄 {description}...")
    try:
        # Import the task module dynamically
        import importlib
        module = importlib.import_module(task_module)
        task_func = getattr(module, task_name)
        
        # Get the task with optional parameters
        if task_kwargs:
            task = task_func(**task_kwargs)
        else:
            task = task_func()
        
        # Set up model configuration if provided
        model_args = []
        if model_config:
            vllm_config = CONFIG.get_vllm_config()
            base_url = model_config.get('base_url', vllm_config['base_url'])
            model_name = model_config.get('model', vllm_config.get('generation_model', 'default'))
            model_args = [f"openai/{model_name}", "--model-base-url", base_url]
        
        # Run the task using inspect_ai eval
        if model_args:
            result = eval(task, model=model_args[0], model_base_url=model_args[2])
        else:
            result = eval(task)
        
        console.print("✅ Task completed successfully")
        return True
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        return False


# =============================================================================
# MAIN COMMANDS
# =============================================================================

@app.command()
def status():
    """📊 Show overall pipeline status and health check."""
    console.print(Panel.fit("🔬 Research Link Technology Landscaping - Status", style="bold blue"))
    
    # Check configuration
    console.print("⚙️  Configuration Status:", style="bold")
    try:
        vllm_config = CONFIG.get_vllm_config()
        keywords_config = CONFIG.get_keywords_config() 
        grants_config = CONFIG.get_grants_config()
        console.print(f"✅ Config loaded from: {CONFIG.root_dir / 'config.yaml'}")
        console.print(f"🔌 vLLM server: {vllm_config['base_url']}")
        console.print(f"🏷️  Keywords batch size: {keywords_config['batch_size']}")
    except Exception as e:
        console.print(f"❌ Configuration error: {e}", style="red")
        
    # Check data files
    console.print("\n📁 Data Files Status:", style="bold")
    
    files_to_check = [
        ("Grants data", CONFIG.grants_file),
        ("Extracted keywords", CONFIG.extracted_keywords_path), 
        ("Keywords", CONFIG.keywords_path),
    ]
    
    for name, path in files_to_check:
        if Path(path).exists():
            size = Path(path).stat().st_size
            console.print(f"✅ {name}: {path} ({size:,} bytes)")
        else:
            console.print(f"❌ {name}: {path} (missing)", style="red")
    
    # Check ChromaDB collections
    console.print("\n🗄️  ChromaDB Status:", style="bold")
    try:
        # Check collections using unified database manager
        from database import get_db_manager
        db_manager = get_db_manager()
        
        keywords_count = db_manager.count_documents("keywords")
        console.print(f"✅ Keywords collection: {keywords_count:,} documents")
        
        grants_count = db_manager.count_documents("grants")
        console.print(f"✅ Grants collection: {grants_count:,} grants")
    except Exception as e:
        console.print(f"❌ Grants collection error: {e}", style="red")
    
    # Check vLLM server
    console.print("\n🤖 vLLM Server Status:", style="bold") 
    try:
        import requests
        vllm_config = CONFIG.get_vllm_config()
        url = vllm_config['base_url'].rstrip('/v1') + '/health'
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            console.print(f"✅ vLLM server accessible at {vllm_config['base_url']}")
        else:
            console.print(f"⚠️  vLLM server status: {response.status_code}")
    except Exception as e:
        console.print(f"❌ vLLM server not accessible: {e}", style="red")


# =============================================================================
# CREATE SUBAPPS FOR ORGANIZED COMMANDS
# =============================================================================

# Create subcommand apps
data_app = typer.Typer(help="📁 Data management commands (loading, clearing, info)")
embed_app = typer.Typer(help="🧠 Embedding generation and management commands")
cluster_app = typer.Typer(help="🔗 Clustering and harmonization commands")
pipeline_app = typer.Typer(help="🚀 High-level pipeline orchestration commands")
visualize_app = typer.Typer(help="📊 Visualization and reporting commands")

# Add subcommand apps to main app
app.add_typer(data_app, name="data")
app.add_typer(embed_app, name="embed")
app.add_typer(cluster_app, name="cluster")
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(visualize_app, name="visualize")


# =============================================================================
# DATA MANAGEMENT COMMANDS  
# =============================================================================

@data_app.command()
def load_grants(
    vllm_url: Optional[str] = typer.Option(
        None, 
        "--vllm-url", 
        help="vLLM server URL (defaults to YAML config)"
    )
):
    """📥 Load grants from JSON file into ChromaDB."""
    from grants import load_and_store_grants_in_chromadb
    
    try:
        console.print("🔄 Loading grants into ChromaDB...")
        # Use YAML config as default if not specified
        vllm_config = CONFIG.get_vllm_config()
        actual_vllm_url = vllm_url or vllm_config["base_url"]
        db_manager = load_and_store_grants_in_chromadb(actual_vllm_url)
        count = db_manager.count_documents("grants")
        console.print(f"✅ Total grants in ChromaDB: {count}")
    except Exception as e:
        console.print(f"❌ Error loading grants: {e}", style="red")
        raise typer.Exit(1)


@data_app.command() 
def grants_info(
    vllm_url: Optional[str] = typer.Option(
        None,
        "--vllm-url", 
        help="vLLM server URL (defaults to YAML config)"
    )
):
    """ℹ️  Show information about grants in ChromaDB."""
    from database import get_db_manager
    
    try:
        # Use YAML config as default if not specified
        vllm_config = CONFIG.get_vllm_config()
        actual_vllm_url = vllm_url or vllm_config["base_url"]
        db_manager = get_db_manager()
        count = db_manager.count_documents("grants")
        console.print("📊 ChromaDB Stats:")
        console.print(f"  Database path: {db_manager.db_path}")
        console.print(f"  Collection name: grants")
        console.print(f"  Total grants: {count}")
        
        if count > 0:
            grant_ids = db_manager.get_all_grant_ids()
            sample_ids = grant_ids[:5]
            more_indicator = "..." if len(grant_ids) > 5 else ""
            console.print(f"  Sample grant IDs: {sample_ids}{more_indicator}")
    except Exception as e:
        console.print(f"❌ Error getting grants info: {e}", style="red")
        raise typer.Exit(1)


@data_app.command()
def clear_grants(
    force: bool = typer.Option(
        False,
        "--force", 
        help="Force deletion without confirmation"
    ),
    vllm_url: Optional[str] = typer.Option(
        None,
        "--vllm-url",
        help="vLLM server URL (defaults to YAML config)" 
    )
):
    """🗑️  Clear all grants from ChromaDB."""
    from database import get_db_manager
    
    
    try:
        # Use YAML config defaults if not provided
        vllm_config = CONFIG.get_vllm_config()
        actual_vllm_url = vllm_url or vllm_config['base_url']
        
        if not force:
            confirm = typer.confirm("⚠️ This will delete all grants from ChromaDB. Continue?")
            if not confirm:
                console.print("❌ Operation cancelled")
                raise typer.Exit(1)
        
        db_manager = get_db_manager()
        # Clear grants collection
        try:
            db_manager.client.delete_collection("grants")
            console.print("✅ Cleared all grants from ChromaDB")
        except Exception as e:
            console.print(f"✅ Grants collection cleared (may not have existed): {e}")
        
    except Exception as e:
        console.print(f"❌ Error clearing grants: {e}", style="red")
        raise typer.Exit(1)


@data_app.command()
def extract_keywords(
    max_grants: Optional[int] = typer.Option(
        None,
        "--max-grants", "-m",
        help="Maximum number of grants to process"
    ),
    vllm_url: Optional[str] = typer.Option(
        None,
        "--vllm-url", 
        help="vLLM server URL (defaults to YAML config)"
    )
):
    """🔍 Extract keywords from grant summaries using simple NLP."""
    from grants import load_and_store_grants_in_chromadb
    from database import get_db_manager
    
    
    try:
        # Use YAML config defaults if not provided
        vllm_config = CONFIG.get_vllm_config()
        actual_vllm_url = vllm_url or vllm_config['base_url']
        
        console.print("🔍 Starting simple keyword extraction...")
        db_manager = get_db_manager()
        
        # Ensure grants are loaded
        count = db_manager.count_grants()
        if count == 0:
            console.print("No grants found in ChromaDB. Loading from file...")
            db_manager = load_and_store_grants_in_chromadb(vllm_url)
        
        # Import and download NLTK data
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        from nltk.tokenize import word_tokenize
        
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True) 
            nltk.download('stopwords', quiet=True)
        except Exception:
            console.print("⚠️ Warning: Could not download NLTK data")
        
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        
        all_grant_ids = db_manager.get_all_grant_ids()
        
        # Limit grants if specified
        if max_grants and max_grants < len(all_grant_ids):
            all_grant_ids = all_grant_ids[:max_grants]
            console.print(f"📊 Limited to first {max_grants} grants for testing")
        
        console.print(f"Processing {len(all_grant_ids)} grants...")
        
        all_keywords = set()
        for i, grant_id in enumerate(all_grant_ids):
            if i % 100 == 0:
                console.print(f"  Processed {i}/{len(all_grant_ids)} grants...")
                
            grant_data = db_manager.get_grant(grant_id)
            if grant_data and grant_data.get('grant_summary'):
                # Simple keyword extraction
                text = grant_data['grant_summary'].lower()
                tokens = word_tokenize(text)
                keywords = [
                    stemmer.stem(token) for token in tokens 
                    if token.isalpha() and token not in stop_words and len(token) > 2
                ]
                all_keywords.update(keywords)
        
        console.print(f"✅ Extracted {len(all_keywords)} unique keywords")
        sample_keywords = list(all_keywords)[:10]
        console.print(f"📝 Sample keywords: {sample_keywords}")
        
    except Exception as e:
        console.print(f"❌ Error extracting keywords: {e}", style="red")
        raise typer.Exit(1)


# =============================================================================
# EMBEDDING COMMANDS
# =============================================================================

@embed_app.command()
def generate(
    collection: str = typer.Option(
        "keywords",
        "--collection", "-c",
        help="Collection type: keywords, grants, categories, keyword_embeddings"
    ),
    data_file: Optional[str] = typer.Option(
        None,
        "--data-file", "-f",
        help="Path to data JSON file (auto-detected from collection if not provided)"
    ),
    max_items: Optional[int] = typer.Option(
        None, 
        "--max-items", "-m",
        help="Maximum items to process"
    ),
    vllm_url: Optional[str] = typer.Option(
        None,
        "--vllm-url",
        help="vLLM server URL (defaults to YAML config)"
    ),
    force_recreate: bool = typer.Option(
        False,
        "--force-recreate", 
        help="Force recreation of ChromaDB collection"
    )
):
    """🧠 Generate embeddings for any collection type (keywords, grants, categories)."""
    from database import get_db_manager, load_keywords_from_file
    
    # Validate collection type
    valid_collections = ["keywords", "grants", "categories", "keyword_embeddings"]
    if collection not in valid_collections:
        console.print(f"❌ Invalid collection type: {collection}", style="red")
        console.print(f"   Valid options: {', '.join(valid_collections)}")
        raise typer.Exit(1)
    
    try:
        # Use defaults from config if not provided
        vllm_config = CONFIG.get_vllm_config()
        actual_vllm_url = vllm_url or vllm_config['base_url']
        
        # Auto-detect data file if not provided
        if not data_file:
            if collection == "keywords":
                data_file = str(CONFIG.keywords_path)
            elif collection == "grants":
                data_file = "data/grants_cleaned.json"
            elif collection == "categories":
                data_file = "results/categories.json"
            elif collection == "keyword_embeddings":
                data_file = str(CONFIG.keywords_path)
        
        console.print(f"🧠 Generating embeddings for {collection} collection...")
        console.print(f"📄 Data file: {data_file}")
        
        # Get database manager
        db_manager = get_db_manager(actual_vllm_url)
        
        if collection in ["keywords", "keyword_embeddings"]:
            # Handle keyword collections
            raw_keywords = load_keywords_from_file(data_file)
            keyword_info = db_manager.extract_keywords_info_from_data(raw_keywords)
            
            count = db_manager.create_embeddings_for_keywords(
                keyword_info=keyword_info,
                collection_name=collection,
                max_keywords=max_items,
                force_recreate=force_recreate
            )
            console.print(f"✅ Generated {count} keyword embeddings")
            
        elif collection == "grants":
            # Handle grants collection (typically no embeddings)
            import json
            with open(data_file, 'r') as f:
                grants_data = json.load(f)
            
            if max_items:
                grants_data = grants_data[:max_items]
            
            # Store grants in database
            for grant in grants_data:
                db_manager.store_grant(grant)
            
            count = len(grants_data)
            console.print(f"✅ Stored {count} grants (note: grants collection typically doesn't use embeddings)")
            
        elif collection == "categories":
            # Handle categories collection
            import json
            with open(data_file, 'r') as f:
                categories_data = json.load(f)
            
            db_manager.store_categories(categories_data)
            count = len(categories_data.get("categories", []))
            console.print(f"✅ Stored {count} categories with embeddings")
        
        console.print("✅ Embedding generation completed successfully")
        
    except FileNotFoundError:
        console.print(f"❌ Data file not found: {data_file}", style="red")
        console.print("   Use --data-file to specify the correct path")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Error generating embeddings: {e}", style="red")
        raise typer.Exit(1)


@embed_app.command()
def info(
    collection: Optional[str] = typer.Option(
        None,
        "--collection", "-c",
        help="Specific collection to show info for (shows all if not specified)"
    ),
    vllm_url: Optional[str] = typer.Option(
        None,
        "--vllm-url",
        help="vLLM server URL (defaults to YAML config)"
    )
):
    """ℹ️  Show embedding collection information for all or specific collections."""
    from database import get_db_manager
    
    try:
        # Use config defaults if not provided
        vllm_config = CONFIG.get_vllm_config()
        actual_vllm_url = vllm_url or vllm_config['base_url']
        
        # Get database manager
        db_manager = get_db_manager(actual_vllm_url)
        
        if collection:
            # Show info for specific collection
            if collection not in db_manager.COLLECTION_CONFIG:
                console.print(f"❌ Unknown collection: {collection}", style="red")
                raise typer.Exit(1)
                
            count = db_manager.count_documents(collection)
            uses_embeddings = db_manager.COLLECTION_CONFIG[collection]["use_embeddings"]
            
            console.print(f"📊 Collection: {collection}")
            console.print(f"  Total documents: {count:,}")
            console.print(f"  Uses embeddings: {'Yes' if uses_embeddings else 'No'}")
            console.print(f"  Database path: {db_manager.db_path}")
            console.print(f"  vLLM server: {actual_vllm_url}")
        else:
            # Show info for all collections
            stats = db_manager.get_collection_stats()
            console.print("📊 All Collections Overview:")
            console.print(f"  Database path: {db_manager.db_path}")
            console.print(f"  vLLM server: {actual_vllm_url}")
            console.print()
            
            for collection_name, count in stats.items():
                uses_embeddings = db_manager.COLLECTION_CONFIG.get(collection_name, {}).get("use_embeddings", False)
                embedding_status = "🧠 with embeddings" if uses_embeddings else "📄 without embeddings"
                console.print(f"  {collection_name}: {count:,} documents {embedding_status}")
        
    except Exception as e:
        console.print(f"❌ Error getting collection info: {e}", style="red")
        raise typer.Exit(1)


@embed_app.command()
def clear(
    collection: str = typer.Option(
        ...,
        "--collection", "-c",
        help="Collection to clear: keywords, grants, categories, keyword_embeddings"
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Force clearing without confirmation"
    )
):
    """🗑️  Clear a specific collection."""
    from database import get_db_manager
    
    try:
        db_manager = get_db_manager()
        
        # Validate collection
        if collection not in db_manager.COLLECTION_CONFIG:
            console.print(f"❌ Unknown collection: {collection}", style="red")
            console.print(f"   Valid options: {', '.join(db_manager.COLLECTION_CONFIG.keys())}")
            raise typer.Exit(1)
        
        # Check current count
        count = db_manager.count_documents(collection)
        if count == 0:
            console.print(f"📊 Collection '{collection}' is already empty")
            return
        
        # Confirm deletion unless force is used
        if not force:
            uses_embeddings = db_manager.COLLECTION_CONFIG[collection]["use_embeddings"]
            embedding_note = " (includes embeddings)" if uses_embeddings else ""
            
            console.print(f"⚠️  About to clear collection '{collection}'{embedding_note}")
            console.print(f"   This will delete {count:,} documents")
            
            confirm = typer.confirm("Are you sure you want to continue?")
            if not confirm:
                console.print("❌ Operation cancelled")
                return
        
        # Clear the collection
        db_manager.clear_collection(collection)
        console.print(f"✅ Cleared collection '{collection}' ({count:,} documents removed)")
        
    except Exception as e:
        console.print(f"❌ Error clearing collection: {e}", style="red")
        raise typer.Exit(1)


@embed_app.command() 
def test(
    text: str = typer.Option(
        "test embedding",
        "--text", "-t",
        help="Text to test embedding generation"
    ),
    vllm_url: Optional[str] = typer.Option(
        None,
        "--vllm-url", 
        help="vLLM server URL (defaults to YAML config)"
    )
):
    """🧪 Test embedding generation with vLLM server."""
    from database import test_embedding_generation
    
    
    try:
        # Use config defaults if not provided
        vllm_config = CONFIG.get_vllm_config()
        actual_vllm_url = vllm_url or vllm_config['base_url']
        
        console.print(f"🧪 Testing embedding generation for: '{text}'...")
        embedding = test_embedding_generation(text, actual_vllm_url)
        console.print(f"✅ Generated embedding with dimension: {len(embedding)}")
        console.print(f"📝 First 5 values: {embedding[:5]}")
        
    except Exception as e:
        console.print(f"❌ Error testing embedding: {e}", style="red")
        raise typer.Exit(1)


# =============================================================================
# CLUSTERING COMMANDS
# =============================================================================

@cluster_app.command() 
def harmonize(
    output_file: Optional[str] = typer.Option(
        None,
        "--output", "-o", 
        help="Output file for clustering results (defaults to config)"
    ),
    similarity_threshold: Optional[float] = typer.Option(
        None,
        "--threshold", "-t",
        help="Similarity threshold (0.0-1.0, defaults to config)"
    ),
    min_cluster_size: Optional[int] = typer.Option(
        None,
        "--min-size", "-s", 
        help="Minimum cluster size (defaults to config)"
    ),
    force_regenerate: bool = typer.Option(
        False,
        "--force",
        help="Force regeneration of clustering"
    )
):
    """🔗 Harmonize keywords using clustering."""
    from clustering import perform_keyword_clustering
    
    
    try:
        # Use config defaults if not provided
        keywords_config = CONFIG.get_keywords_config()
        actual_output = output_file or str(CONFIG.clusters_proposal_path)
        actual_threshold = similarity_threshold or keywords_config.get('similarity_threshold', 0.8)
        actual_min_size = min_cluster_size or keywords_config.get('min_cluster_size', 2)
        
        console.print("🔗 Harmonizing keywords using clustering...")
        perform_keyword_clustering(
            output_file=actual_output,
            similarity_threshold=actual_threshold,
            min_cluster_size=actual_min_size,
            force_regenerate=force_regenerate
        )
        console.print("✅ Keyword clustering completed")
        
    except Exception as e:
        console.print(f"❌ Error harmonizing keywords: {e}", style="red")
        raise typer.Exit(1)


@cluster_app.command()
def map_keywords(
    clusters_file: Optional[str] = typer.Option(
        None,
        "--clusters", "-c",
        help="Path to clustering results (defaults to config)"
    ),
    keywords_file: Optional[str] = typer.Option(
        None, 
        "--keywords", "-k",
        help="Path to original keywords (defaults to config)" 
    ),
    output_file: str = typer.Option(
        "keyword_mapping.json",
        "--output", "-o",
        help="Output file for mapping results"
    )
):
    """🗺️  Map original keywords to harmonized clusters."""
    from clustering import create_keyword_mapping
    
    
    try:
        # Use config defaults if not provided
        actual_clusters = clusters_file or str(CONFIG.clusters_proposal_path)
        actual_keywords = keywords_file or str(CONFIG.keywords_path)
        
        console.print("🗺️  Mapping keywords to clusters...")
        create_keyword_mapping(
            clusters_file=actual_clusters,
            keywords_file=actual_keywords,
            output_file=output_file
        )
        console.print(f"✅ Keyword mapping saved to: {output_file}")
        
    except Exception as e:
        console.print(f"❌ Error mapping keywords: {e}", style="red")
        raise typer.Exit(1)


@cluster_app.command()
def clear_cache():
    """🧹 Clear clustering cache and ChromaDB data."""
    from clustering import clear_clustering_cache
    
    try:
        console.print("🧹 Clearing clustering cache...")
        clear_clustering_cache()
        console.print("✅ Clustering cache cleared")
        
    except Exception as e:
        console.print(f"❌ Error clearing cache: {e}", style="red")
        raise typer.Exit(1)


# =============================================================================
# PIPELINE COMMANDS (High-level workflows)
# =============================================================================

@pipeline_app.command()
def extract_and_embed(
    max_grants: Optional[int] = typer.Option(
        None,
        "--max-grants", "-m", 
        help="Maximum grants to process"
    ),
    max_keywords: Optional[int] = typer.Option(
        None,
        "--max-keywords", "-k",
        help="Maximum keywords to embed"
    ),
    vllm_url: Optional[str] = typer.Option(
        None,
        "--vllm-url",
        help="vLLM server URL (defaults to YAML config)"
    )
    ):
    """🚀 Run complete extraction and embedding pipeline."""
    from database import generate_keywords_embeddings
    
    
    console.print("🔄 Running extraction and embedding pipeline...", style="bold blue")
    
    try:
        # Step 1: Extract keywords using inspect_ai task
        model_config = {}
        if vllm_url:
            model_config['base_url'] = vllm_url
        
        if not run_inspect_task("extract", "extract", "Extracting keywords from grants", model_config):
            console.print("❌ Keyword extraction failed", style="red")
            return
            
        # Step 2: Generate embeddings
        vllm_config = CONFIG.get_vllm_config()
        actual_vllm_url = vllm_url or vllm_config['base_url']
        
        console.print("🧠 Generating embeddings...")
        generate_keywords_embeddings(
            keywords_file=str(CONFIG.keywords_path),
            vllm_url=actual_vllm_url,
            max_keywords=max_keywords
        )
        
        console.print("✅ Extraction and embedding pipeline completed!", style="bold green")
        
    except Exception as e:
        console.print(f"❌ Pipeline failed: {e}", style="red")
        raise typer.Exit(1)


@pipeline_app.command()
def categorize(
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size", "-b",
        help="Batch size for categorization"
    ),
    keywords_type: Optional[str] = typer.Option(
        None,
        "--keywords-type", "-t", 
        help="Type of keywords to categorize"
    ),
    vllm_url: Optional[str] = typer.Option(
        None,
        "--vllm-url",
        help="vLLM server URL (defaults to YAML config)"
    )
):
    """📋 Run keyword categorization."""
    model_config = {}
    if vllm_url:
        model_config['base_url'] = vllm_url
        model_config['model'] = CONFIG.get_vllm_config().get('generation_model', 'Qwen/Qwen3-72B-Instruct')
    
    task_kwargs = {}
    if batch_size:
        task_kwargs['batch_size'] = batch_size
    if keywords_type:
        task_kwargs['keywords_type'] = keywords_type
        
    run_inspect_task("categorise", "categorise", "Categorizing keywords", model_config, **task_kwargs)


@pipeline_app.command()
def classify():
    """🏷️  Run keyword classification."""
    run_inspect_task("classify", "classify", "Classifying keywords")


# =============================================================================
# VISUALIZATION COMMANDS  
# =============================================================================

if __name__ == "__main__":
    app()
