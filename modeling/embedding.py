"""
Embedding generation module for keyword vectorization using ChromaDB with vLLM embeddings.

This module provides functionality to create embeddings for keywords and store them in
ChromaDB vector database. It delegates embedding generation to ChromaDB using a custom
OpenAI-compatible embedding function that connects to vLLM servers.

Key features:
- ChromaDB-native embedding generation using custom OpenAI embedding function
- vLLM embedding server integration through OpenAI API compatibility
- Persistent embedding storage with metadata
- CLI interface for embedding operations
- Batch processing for large datasets
"""

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional

import numpy as np
from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import typer

from config import (
    KEYWORDS_EMBEDDING_DBPATH,
    EXTRACTED_KEYWORDS_PATH
)


class EmbeddingManager:
    """
    Embedding manager for keyword vectorization using ChromaDB with vLLM embeddings.
    
    This class handles the embedding generation workflow by delegating to ChromaDB's
    built-in OpenAI embedding function, configured to use vLLM server endpoints.
    
    Key features:
    1. ChromaDB-native embedding generation with OpenAI API compatibility
    2. vLLM integration through OpenAI-compatible endpoint
    3. Persistent storage with metadata
    4. Keyword preprocessing and normalization
    """

    def __init__(self, vllm_base_url: str = "http://localhost:8000/v1", max_model_len: int = 8192):
        """
        Initialize EmbeddingManager with vLLM server configuration.
        
        Args:
            vllm_base_url: Base URL for the vLLM embedding server
            max_model_len: Maximum token length for text truncation
        """
        self.vllm_base_url = vllm_base_url
        self.max_model_len = max_model_len
        
        # ChromaDB components
        self.chroma_client = None
        self.chroma_collection = None
        self.collection_name = None  # Will be set based on model name
        self.model_name = None
        
        # Embedding function
        self.embedding_function = None
        
        # Keyword storage
        self.raw_keywords = []
        
        # Model name (will be detected from vLLM server)
        self.model_name = None

    def get_vllm_model_name(self) -> str:
        """Get the model name from vLLM server and set collection name."""
        try:
            # Create a temporary client to get model info
            temp_client = OpenAI(api_key="EMPTY", base_url=self.vllm_base_url)
            models = temp_client.models.list()
            if not models.data:
                raise Exception("No models available on vLLM server")
            
            self.model_name = models.data[0].id
            
            # Create collection name based on model name
            # Replace special characters and make it a valid collection name
            safe_model_name = self.model_name.replace("/", "_").replace("-", "_").replace(".", "_")
            self.collection_name = f"embeddings_{safe_model_name}"
            
            print(f"✅ Detected vLLM model: {self.model_name}")
            print(f"📦 Using collection name: {self.collection_name}")
            return self.model_name
        except Exception as e:
            raise Exception(f"Failed to get model name from vLLM server: {e}")

    def initialize_chromadb(self) -> None:
        """
        Initialize ChromaDB client with OpenAI embedding function configured for vLLM.
        """
        try:
            # Create ChromaDB client with persistent storage
            db_path = KEYWORDS_EMBEDDING_DBPATH / "chromadb"
            db_path.mkdir(parents=True, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(path=str(db_path))
            
            # Get model name from vLLM server
            model_name = self.get_vllm_model_name()
            
            # Create OpenAI embedding function configured for vLLM
            self.embedding_function = OpenAIEmbeddingFunction(
                api_key="EMPTY",  # vLLM doesn't require a real API key
                model_name=model_name,
                api_base=self.vllm_base_url
            )
            
            # Create or get collection with vLLM-compatible OpenAI embedding function
            try:
                self.chroma_collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                )
                print(f"✅ Created new ChromaDB collection: {self.collection_name}")
            except Exception as create_error:
                # Collection might already exist, try to get it instead
                try:
                    self.chroma_collection = self.chroma_client.get_collection(
                        name=self.collection_name,
                        embedding_function=self.embedding_function
                    )
                    print(f"✅ Using existing ChromaDB collection: {self.collection_name}")
                except Exception as get_error:
                    raise Exception(f"Failed to create or get collection: create_error={create_error}, get_error={get_error}")
                
        except Exception as e:
            raise Exception(f"Failed to initialize ChromaDB: {e}")

    def load_keywords(self, keywords_file: Union[str, Path]) -> None:
        """
        Load keywords from the extract task output file.
        Expects flattened structure: array of keyword objects at top level.
        
        Args:
            keywords_file: Path to keywords JSON file.
        """
        keywords_path = Path(keywords_file)
        with keywords_path.open('r', encoding='utf-8') as f:
            self.raw_keywords = json.load(f)
        
        print(f"✅ Loaded {len(self.raw_keywords)} keywords from {keywords_file}")

    def normalize_keyword(self, keyword: str) -> str:
        """
        Normalize a keyword through preprocessing steps.
        
        Args:
            keyword: Raw keyword string
            
        Returns:
            Normalized keyword string
        """
        # Convert to lowercase
        normalized = keyword.lower().strip()
        
        # Remove special characters but keep hyphens and spaces
        normalized = re.sub(r'[^\w\s\-]', '', normalized)
        
        # Replace multiple spaces with single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove leading/trailing spaces
        normalized = normalized.strip()
        
        return normalized

    def extract_all_keywords(self) -> Dict[str, Dict]:
        """
        Extract all keywords from loaded flat data with enhanced information.
        
        Returns:
            Dictionary mapping keyword terms to their information including frequency, keyword objects, etc.
        """
        keyword_info = defaultdict(lambda: {'frequency': 0, 'keyword_objects': []})
        
        for keyword in self.raw_keywords:
            term = keyword.get('term', '')
            
            normalized_term = self.normalize_keyword(term)
            keyword_info[normalized_term]['frequency'] += 1
            keyword_info[normalized_term]['keyword_objects'].append(keyword)
        
        return keyword_info

    def trim_text_to_max_length(self, text: str) -> str:
        """
        Trim text to fit within the model's maximum token length.
        Simple approach: just take the first max_model_len characters.
        
        Args:
            text: Input text to trim
            
        Returns:
            Trimmed text that should fit within model limits
        """
        if len(text) <= self.max_model_len:
            return text
        return text[:self.max_model_len]

    def create_chromadb_collection(
        self, 
        keyword_info: Dict[str, Dict], 
        max_keywords: Optional[int] = None,
        force_recreate: bool = False
    ) -> int:
        """
        Create ChromaDB collection with keywords using built-in embedding generation.
        
        Args:
            keyword_info: Dictionary of keywords with enhanced information
            max_keywords: Maximum number of keywords to process. If None, use all keywords.
            force_recreate: Force recreation of the collection even if it exists
            
        Returns:
            Number of embeddings in the collection after processing
        """
        # Initialize ChromaDB
        self.initialize_chromadb()
        
        # Initialize sorted_keywords variable
        sorted_keywords = None
        
        # Check if we can use existing collection (unless forcing recreation)
        if not force_recreate:
            try:
                count = self.chroma_collection.count()
                if count > 0:
                    print(f"📦 Found existing ChromaDB collection with {count} embeddings")
                    print("🔍 Checking which keywords are already embedded...")
                    
                    # Get existing keywords from the collection
                    existing_results = self.chroma_collection.get(include=['metadatas'])
                    existing_keywords = set()
                    if existing_results['metadatas']:
                        existing_keywords = {meta['keyword'] for meta in existing_results['metadatas'] if meta and 'keyword' in meta}
                    
                    print(f"📊 Found {len(existing_keywords)} existing keywords in collection")
                    
                    # Filter out keywords that already exist
                    sorted_all = sorted(
                        keyword_info.items(), 
                        key=lambda x: x[1]['frequency'], 
                        reverse=True
                    )
                    if max_keywords is None:
                        sorted_keywords = sorted_all
                    else:
                        sorted_keywords = sorted_all[:max_keywords]
                    
                    new_keywords = [(kw, info) for kw, info in sorted_keywords if kw not in existing_keywords]
                    
                    if len(new_keywords) == 0:
                        print("✅ All requested keywords already exist in collection")
                        final_count = self.chroma_collection.count()
                        print(f"✅ Collection has {final_count} total embeddings")
                        return final_count
                    else:
                        print(f"➕ Found {len(new_keywords)} new keywords to add to collection")
                        # Continue with adding only new keywords
                        sorted_keywords = new_keywords
                        
            except Exception as e:
                print(f"⚠️ Failed to check existing ChromaDB collection: {e}")
                print("Creating new collection...")
                sorted_keywords = None  # Will be set below
        else:
            sorted_keywords = None  # Will be set below
        
        # Clear existing collection if forcing recreation
        if force_recreate:
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
                print("🗑️ Deleted existing collection for recreation")
                # Recreate the collection
                self.chroma_collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                print(f"⚠️ Error during collection recreation: {e}")
        
        # Prepare keywords for embedding
        print("🔄 Preparing keywords for embedding...")
        
        # If sorted_keywords wasn't set above (force_recreate or error), calculate it now
        if sorted_keywords is None:
            sorted_all = sorted(
                keyword_info.items(), 
                key=lambda x: x[1]['frequency'], 
                reverse=True
            )
            if max_keywords is None:
                sorted_keywords = sorted_all
            else:
                sorted_keywords = sorted_all[:max_keywords]
        
        keywords = [kw for kw, info in sorted_keywords]
        
        if len(keywords) < 1:
            # If we have existing collection and no new keywords, return existing data
            if not force_recreate:
                try:
                    final_count = self.chroma_collection.count()
                    print(f"✅ No new keywords to add. Collection has {final_count} existing embeddings")
                    return final_count
                except Exception:
                    pass
            raise ValueError("No keywords to process")
        
        print(f"📊 Processing {len(keywords)} keywords for embedding")
        
        # Prepare documents for ChromaDB (combine keyword terms with descriptions)
        documents = []
        metadatas = []
        ids = []
        
        for i, keyword in enumerate(keywords):
            kw_info = keyword_info[keyword]
            keyword_objects = kw_info.get('keyword_objects', [])
            
            # Create document text from descriptions
            descriptions = []
            for kw_obj in keyword_objects:
                desc = kw_obj.get('description', '')
                if desc:
                    descriptions.append(desc)
            
            # Combine keyword with descriptions for richer embeddings
            document_text = f"{keyword}: " + " ".join(descriptions)
            
            # Trim text to fit within model limits
            trimmed_text = self.trim_text_to_max_length(document_text)
            
            documents.append(trimmed_text)
            metadatas.append({
                "keyword": keyword,
                "frequency": kw_info['frequency'],
                "num_objects": len(keyword_objects)
            })
            # Use keyword-based ID to avoid conflicts
            safe_keyword = keyword.replace(" ", "_").replace("/", "_").replace("\\", "_")
            ids.append(f"keyword_{safe_keyword}_{hash(keyword) % 10000}")
        
        # Add documents to ChromaDB - embeddings will be generated automatically
        print(f"🚀 Adding {len(documents)} documents to ChromaDB (embeddings will be generated automatically)...")
        
        # Add in batches to handle size limits
        self._add_documents_in_batches(documents, metadatas, ids)
        
        print("✅ Documents and embeddings stored in ChromaDB collection")
        
        # Get final count instead of loading all embeddings
        final_count = self.chroma_collection.count()
        print(f"✅ Collection now contains {final_count} total embeddings")
        
        return final_count

    def _add_documents_in_batches(
        self, 
        documents: List[str], 
        metadatas: List[Dict], 
        ids: List[str]
    ) -> None:
        """
        Add documents to ChromaDB collection in batches to handle size limits.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of document IDs
        """
        # Get the maximum batch size directly from ChromaDB client
        max_batch_size = self.chroma_client.get_max_batch_size()
        total_items = len(documents)
        
        print(f"Adding {total_items} documents to ChromaDB collection in batches of up to {max_batch_size}...")
        
        batch_size = max_batch_size
        batch_num = 0
        items_processed = 0
        
        while items_processed < total_items:
            batch_start = items_processed
            batch_end = min(batch_start + batch_size, total_items)
            batch_num += 1
            
            batch_documents = documents[batch_start:batch_end]
            batch_metadatas = metadatas[batch_start:batch_end]
            batch_ids = ids[batch_start:batch_end]
            
            try:
                # ChromaDB will automatically generate embeddings using our embedding function
                self.chroma_collection.add(
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                print(f"  ✅ Added batch {batch_num}: items {batch_start} to {batch_end-1} ({batch_end-batch_start} items)")
                items_processed = batch_end
                
            except Exception as e:
                error_msg = str(e).lower()
                # Handle batch size errors
                if "batch size" in error_msg and "greater than max batch size" in error_msg:
                    # Try to extract the max batch size from error message
                    import re
                    match = re.search(r'max batch size of (\d+)', error_msg)
                    if match:
                        suggested_max = int(match.group(1))
                        new_batch_size = max(1, int(suggested_max * 0.8))
                        print(f"  ⚠️ Batch size too large. Reducing from {batch_size} to {new_batch_size}")
                        batch_size = new_batch_size
                        continue
                
                # If batch size is still too large, try smaller
                if batch_size > 10:
                    batch_size = max(10, batch_size // 2)
                    print(f"  ⚠️ Batch failed, trying smaller size: {batch_size}")
                    continue
                else:
                    print(f"  ❌ Failed to add batch even with small size: {e}")
                    raise e
        
        print(f"✅ Successfully added {total_items} documents to ChromaDB collection")

    def clear_embedding_cache(self) -> int:
        """
        Clear ChromaDB collection and embeddings cache.
        
        Returns:
            Number of items removed
        """
        chromadb_path = KEYWORDS_EMBEDDING_DBPATH / "chromadb"
        
        removed_count = 0
        
        # Remove ChromaDB directory
        if chromadb_path.exists():
            try:
                import shutil
                shutil.rmtree(chromadb_path)
                removed_count += 1
                print(f"🗑️ Removed ChromaDB directory: {chromadb_path}")
            except Exception as e:
                print(f"⚠️ Failed to remove ChromaDB directory: {e}")
        
        # Clear instance variables
        self.chroma_collection = None
        self.chroma_client = None
        self.embedding_function = None
        
        return removed_count

    def estimate_embedding_size(self, num_keywords: int) -> Dict[str, Union[int, str]]:
        """
        Estimate the storage size of embeddings before generating them.
        
        Args:
            num_keywords: Number of keywords to estimate for
            
        Returns:
            Dictionary with size estimates
        """
        # Initialize embedding function to get dimensions
        if self.embedding_function is None:
            # Create a temporary OpenAI embedding function
            model_name = self.get_vllm_model_name()
            temp_embedding_func = OpenAIEmbeddingFunction(
                api_key="EMPTY",
                model_name=model_name,
                api_base=self.vllm_base_url
            )
        else:
            temp_embedding_func = self.embedding_function
        
        # Get embedding dimensions by testing with a small sample
        test_embeddings = temp_embedding_func(["test"])
        embedding_dim = len(test_embeddings[0])
        
        # Calculate size estimates
        # Each embedding is float32 = 4 bytes per dimension
        bytes_per_embedding = embedding_dim * 4  # 4 bytes for float32
        total_embedding_bytes = num_keywords * bytes_per_embedding
        
        # Convert to human-readable units
        def format_bytes(bytes_val):
            for unit in ['B', 'KB', 'MB', 'GB']:
                if bytes_val < 1024.0:
                    return f"{bytes_val:.1f} {unit}"
                bytes_val /= 1024.0
            return f"{bytes_val:.1f} TB"
        
        # Estimate additional storage for ChromaDB (roughly 1.5x for metadata and indexing)
        chromadb_overhead = total_embedding_bytes * 0.5
        total_storage_estimate = total_embedding_bytes + chromadb_overhead
        
        return {
            'num_keywords': num_keywords,
            'embedding_dim': embedding_dim,
            'raw_embeddings_bytes': total_embedding_bytes,
            'chromadb_overhead_bytes': chromadb_overhead,
            'total_storage_bytes': total_storage_estimate,
            'raw_embeddings_formatted': format_bytes(total_embedding_bytes),
            'chromadb_overhead_formatted': format_bytes(chromadb_overhead),
            'total_storage_formatted': format_bytes(total_storage_estimate)
        }


# Create Typer CLI app
app = typer.Typer(help="Keyword embedding generation tool using vLLM and ChromaDB")


@app.command()
def embed(
    keywords_file: str = typer.Option(
        str(EXTRACTED_KEYWORDS_PATH),
        "--keywords-file", "-k",
        help="Path to keywords JSON file"
    ),
    max_keywords: Optional[int] = typer.Option(
        None,
        "--max-keywords", "-m",
        help="Maximum number of keywords to process (for performance testing)"
    ),
    vllm_url: str = typer.Option(
        "http://localhost:8000/v1",
        "--vllm-url",
        help="Base URL for vLLM embedding server"
    ),
    force_recreate: bool = typer.Option(
        False,
        "--force-recreate",
        help="Force recreation of ChromaDB collection even if it exists"
    ),
    show_estimation: bool = typer.Option(
        True,
        "--show-estimation/--no-estimation",
        help="Show embedding size estimation before processing"
    )
):
    """Generate embeddings for keywords using vLLM server through ChromaDB."""
    try:
        typer.echo(f"🔄 Initializing EmbeddingManager with vLLM at {vllm_url}...")
        embedding_manager = EmbeddingManager(vllm_base_url=vllm_url)
        
        typer.echo(f"📄 Loading keywords from {keywords_file}...")
        embedding_manager.load_keywords(keywords_file)
        
        typer.echo("🔍 Extracting keyword information...")
        keyword_info = embedding_manager.extract_all_keywords()
        
        # Determine how many keywords will be processed
        if max_keywords is None:
            num_keywords_to_process = len(keyword_info)
        else:
            num_keywords_to_process = min(max_keywords, len(keyword_info))
        
        # Show size estimation if requested
        if show_estimation:
            typer.echo("📊 Estimating embedding size...")
            size_info = embedding_manager.estimate_embedding_size(num_keywords_to_process)
            
            typer.echo("📊 Embedding size estimation:")
            typer.echo(f"   • Keywords to process: {size_info['num_keywords']:,}")
            typer.echo(f"   • Embedding dimensions: {size_info['embedding_dim']}")
            typer.echo(f"   • Raw embeddings size: {size_info['raw_embeddings_formatted']}")
            typer.echo(f"   • ChromaDB overhead: {size_info['chromadb_overhead_formatted']}")
            typer.echo(f"   • Total storage estimate: {size_info['total_storage_formatted']}")
            
            # Warn if the size is very large
            if size_info['total_storage_bytes'] > 1024 * 1024 * 1024:  # > 1GB
                typer.echo("⚠️  Large storage size detected (>1GB). Consider using --max-keywords to limit processing.")
        
        # Ask for confirmation if processing a large number of keywords
        if num_keywords_to_process > 10000:
            if not typer.confirm(f"⚠️  Processing {num_keywords_to_process:,} keywords. This may take a while. Continue?"):
                typer.echo("❌ Operation cancelled by user.")
                raise typer.Exit(1)
        
        typer.echo("🚀 Creating ChromaDB collection with embeddings...")
        final_count = embedding_manager.create_chromadb_collection(
            keyword_info, 
            max_keywords, 
            force_recreate=force_recreate
        )
        
        # Report results without loading all embeddings
        typer.echo(f"✅ Successfully processed {final_count} embeddings in collection")
        typer.echo(f"💾 Embeddings stored in ChromaDB at {KEYWORDS_EMBEDDING_DBPATH / 'chromadb'}")
        
    except Exception as e:
        typer.echo(f"❌ Error during embedding generation: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def info(
    vllm_url: str = typer.Option(
        "http://localhost:8000/v1",
        "--vllm-url",
        help="Base URL for vLLM embedding server"
    )
):
    """Show information about the ChromaDB collection and vLLM server."""
    try:
        embedding_manager = EmbeddingManager(vllm_base_url=vllm_url)
        
        # Initialize ChromaDB to check collection
        typer.echo("📦 Checking ChromaDB collection...")
        try:
            embedding_manager.initialize_chromadb()
            count = embedding_manager.chroma_collection.count()
            if count > 0:
                typer.echo(f"✅ ChromaDB collection exists with {count} embeddings")
                
                # Get some sample data to show dimensions
                sample = embedding_manager.chroma_collection.peek(limit=1)
                if sample['embeddings'] is not None and len(sample['embeddings']) > 0:
                    embedding_dim = len(sample['embeddings'][0])
                    typer.echo(f"📊 Embedding dimensions: {embedding_dim}")
            else:
                typer.echo("ℹ️ ChromaDB collection is empty")
        except Exception as e:
            typer.echo(f"⚠️ ChromaDB collection not accessible: {e}")
        
        # Check vLLM server
        typer.echo(f"🔌 Checking vLLM server at {vllm_url}...")
        try:
            # Create a temporary OpenAI client to test vLLM server
            temp_client = OpenAI(api_key="EMPTY", base_url=vllm_url)
            models = temp_client.models.list()
            if models.data:
                model_name = models.data[0].id
                typer.echo(f"✅ vLLM server is accessible")
                typer.echo(f"📊 Available model: {model_name}")
            else:
                typer.echo("⚠️ vLLM server accessible but no models found")
        except Exception as e:
            typer.echo(f"❌ vLLM server not accessible: {e}")
        
    except Exception as e:
        typer.echo(f"❌ Error getting information: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def clear_cache():
    """Clear ChromaDB embeddings cache."""
    try:
        embedding_manager = EmbeddingManager()
        removed_count = embedding_manager.clear_embedding_cache()
        
        if removed_count == 0:
            typer.echo("ℹ️ No cache files found to remove")
        else:
            typer.echo(f"✅ Cache cleared: {removed_count} items removed")
            
    except Exception as e:
        typer.echo(f"❌ Error clearing cache: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def test_embedding(
    text: str = typer.Option(
        "test embedding",
        "--text", "-t",
        help="Text to test embedding generation with"
    ),
    vllm_url: str = typer.Option(
        "http://localhost:8000/v1",
        "--vllm-url",
        help="Base URL for vLLM embedding server"
    )
):
    """Test embedding generation with vLLM server."""
    try:
        typer.echo(f"🧪 Testing embedding generation for: '{text}'")
        typer.echo(f"🔌 Using vLLM server at: {vllm_url}")
        
        # Get model name from vLLM server
        temp_client = OpenAI(api_key="EMPTY", base_url=vllm_url)
        models = temp_client.models.list()
        if not models.data:
            raise Exception("No models available on vLLM server")
        
        model_name = models.data[0].id
        typer.echo(f"📊 Using model: {model_name}")
        
        # Create and test embedding function
        embedding_func = OpenAIEmbeddingFunction(
            api_key="EMPTY",
            model_name=model_name,
            api_base=vllm_url
        )
        embeddings = embedding_func([text])
        
        typer.echo(f"✅ Successfully generated embedding")
        typer.echo(f"📊 Embedding dimensions: {len(embeddings[0])}")
        typer.echo(f"📄 First 10 dimensions: {embeddings[0][:10]}")
        
    except Exception as e:
        typer.echo(f"❌ Error testing embedding: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
