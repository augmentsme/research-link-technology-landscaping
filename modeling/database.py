"""
Unified ChromaDB Manager for Research Link Technology Landscaping

This module provides a single, unified ChromaDB manager that handles all data collections
in separate collections within the same database instance. It provides general-purpose
methods that work across different data types (grants, keywords, categories, etc.).

Key features:
- Unified collection management
- Configurable embedding support per collection
- General-purpose storage and retrieval methods
- Batch processing for large datasets
- vLLM integration for embeddings
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.utils.batch_utils import create_batches
from openai import OpenAI

from config import CONFIG


class UnifiedChromaDBManager:
    """
    Unified ChromaDB manager for all research data collections.
    
    This class manages multiple collections within a single ChromaDB instance:
    - grants: Research grant data
    - keywords: Extracted keywords (was extracted_keywords)
    - categories: Categorization results
    - And extensible for future collections
    """
    
    _instance = None
    
    # Collection configuration - defines which collections use embeddings
    COLLECTION_CONFIG = {
        "grants": {"use_embeddings": False},  # From config
        "keywords": {"use_embeddings": True},
        "categories": {"use_embeddings": True},
        "keyword_embeddings": {"use_embeddings": True},  # For embedding-specific keywords
    }
    
    def __new__(cls, vllm_base_url: str = None):
        """Singleton pattern to ensure only one DB manager instance."""
        if cls._instance is None:
            cls._instance = super(UnifiedChromaDBManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, vllm_base_url: str = None):
        # Skip initialization if already initialized
        if hasattr(self, '_initialized'):
            return
            
        # Use YAML config as default if not specified
        vllm_config = CONFIG.get_vllm_config()
        grants_config = CONFIG.get_grants_config()
        
        self.vllm_base_url = vllm_base_url or vllm_config["base_url"]
        self.embedding_model = vllm_config["embedding_model"]
        
        # Update grants embedding config from YAML
        self.COLLECTION_CONFIG["grants"]["use_embeddings"] = grants_config["use_embeddings"]
        
        self.db_path = CONFIG.chromadb_path
        self.client = None
        self.collections = {}
        self._initialized = True
        
        # Initialize the database
        self.initialize_chromadb()
    
    def initialize_chromadb(self):
        """Initialize ChromaDB client and prepare for collection access."""
        try:
            self.db_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.db_path))
            
            # Get maximum batch size for optimal performance
            self.max_batch_size = self.client.get_max_batch_size()
            print(f"✅ Initialized unified ChromaDB at {self.db_path}")
            print(f"📊 Maximum batch size: {self.max_batch_size}")
        except Exception as e:
            raise Exception(f"Failed to initialize ChromaDB: {e}")
    
    def _get_embedding_function(self):
        """Get OpenAI embedding function configured for vLLM, fallback to ChromaDB default."""
        try:
            # Try to use vLLM embedding function
            embedding_function = OpenAIEmbeddingFunction(
                api_key="dummy-key",
                api_base=self.vllm_base_url,
                model_name=self.embedding_model
            )
            
            # Test the connection by trying to get model info
            temp_client = OpenAI(api_key="EMPTY", base_url=self.vllm_base_url)
            models = temp_client.models.list()
            if not models.data:
                raise Exception("No models available on vLLM server")
            
            print(f"✅ Using vLLM embedding function with model: {models.data[0].id}")
            return embedding_function
            
        except Exception as e:
            print(f"⚠️  vLLM not available ({e}), falling back to ChromaDB default embedding function (all-MiniLM-L6-v2)")
            # Return None - ChromaDB will automatically use DefaultEmbeddingFunction (all-MiniLM-L6-v2)
            return None
    
    def get_vllm_model_name(self) -> str:
        """Get the model name from vLLM server, fallback to default if not available."""
        try:
            temp_client = OpenAI(api_key="EMPTY", base_url=self.vllm_base_url)
            models = temp_client.models.list()
            if not models.data:
                raise Exception("No models available on vLLM server")
            
            model_name = models.data[0].id
            print(f"✅ Detected vLLM model: {model_name}")
            return model_name
        except Exception as e:
            print(f"⚠️  vLLM server not available ({e}), using ChromaDB default embedding (all-MiniLM-L6-v2)")
            return "all-MiniLM-L6-v2"  # ChromaDB default model name
    
    # ====================
    # CORE COLLECTION METHODS
    # ====================
    
    def get_or_create_collection(self, collection_name: str, use_embeddings: Optional[bool] = None):
        """
        Get or create a collection, caching it for future use.
        
        Args:
            collection_name: Name of the collection
            use_embeddings: Override embedding setting. If None, uses COLLECTION_CONFIG
        """
        if collection_name in self.collections:
            return self.collections[collection_name]
        
        if not self.client:
            self.initialize_chromadb()
        
        # Determine if this collection should use embeddings
        if use_embeddings is None:
            config = self.COLLECTION_CONFIG.get(collection_name, {"use_embeddings": True})
            use_embeddings = config["use_embeddings"]
        
        embedding_function = self._get_embedding_function() if use_embeddings else None
        
        try:
            # Try to get existing collection first
            if use_embeddings:
                embedding_function = self._get_embedding_function()
                if embedding_function:
                    # Use vLLM embedding function
                    collection = self.client.get_collection(
                        name=collection_name,
                        embedding_function=embedding_function
                    )
                else:
                    # Use ChromaDB default (no embedding function specified)
                    collection = self.client.get_collection(name=collection_name)
            else:
                collection = self.client.get_collection(name=collection_name)
            print(f"✅ Using existing ChromaDB collection: {collection_name}")
        except Exception:
            # Collection doesn't exist, create it
            try:
                if use_embeddings:
                    embedding_function = self._get_embedding_function()
                    if embedding_function:
                        # Use vLLM embedding function
                        collection = self.client.create_collection(
                            name=collection_name,
                            embedding_function=embedding_function,
                            metadata={"hnsw:space": "cosine"}
                        )
                        embed_status = "with vLLM embeddings"
                    else:
                        # Use ChromaDB default embedding function (all-MiniLM-L6-v2)
                        collection = self.client.create_collection(
                            name=collection_name,
                            metadata={"hnsw:space": "cosine"}
                        )
                        embed_status = "with default ChromaDB embeddings (all-MiniLM-L6-v2)"
                else:
                    collection = self.client.create_collection(name=collection_name)
                    embed_status = "without embeddings"
                
                print(f"✅ Created new ChromaDB collection: {collection_name} {embed_status}")
            except Exception as create_error:
                raise Exception(f"Failed to create collection {collection_name}: {create_error}")
        
        self.collections[collection_name] = collection
        return collection
    
    def clear_collection(self, collection_name: str):
        """Clear all data from a collection."""
        try:
            if collection_name in self.collections:
                del self.collections[collection_name]
            
            # Delete and recreate the collection
            try:
                self.client.delete_collection(collection_name)
                print(f"✅ Cleared collection: {collection_name}")
            except Exception:
                # Collection might not exist, which is fine
                pass
                
            # Recreate empty collection
            self.get_or_create_collection(collection_name)
            
        except Exception as e:
            raise Exception(f"Failed to clear collection {collection_name}: {e}")
    
    def count_documents(self, collection_name: str) -> int:
        """Get the total number of documents in a collection."""
        try:
            collection = self.get_or_create_collection(collection_name)
            return collection.count()
        except Exception as e:
            print(f"Error counting documents in {collection_name}: {e}")
            return 0
    
    def get_optimal_batch_size(self) -> int:
        """Get the optimal batch size for this ChromaDB client."""
        return getattr(self, 'max_batch_size', 1000)
    
    # ====================
    # GENERAL DATA METHODS
    # ====================
    
    def store_documents(
        self, 
        collection_name: str, 
        documents: List[str], 
        metadatas: List[Dict[str, Any]], 
        ids: List[str],
        batch_size: Optional[int] = None
    ):
        """
        Store multiple documents in a collection with optimal batch processing.
        
        Args:
            collection_name: Name of the collection
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of document IDs
            batch_size: Override batch size (uses ChromaDB max if None)
        """
        collection = self.get_or_create_collection(collection_name)
        
        if len(documents) != len(metadatas) or len(documents) != len(ids):
            raise ValueError("Documents, metadatas, and ids must have the same length")
        
        total_items = len(documents)
        print(f"📄 Storing {total_items} documents in '{collection_name}' collection...")
        
        # Use ChromaDB's maximum batch size or provided override
        effective_batch_size = batch_size or getattr(self, 'max_batch_size', 1000)
        
        # Check if collection uses embeddings to determine batch processing approach
        config = self.COLLECTION_CONFIG.get(collection_name, {"use_embeddings": True})
        use_embeddings = config["use_embeddings"]
        
        if use_embeddings:
            print(f"🔗 Processing {collection_name} collection WITH embeddings (using ChromaDB batch utils)")
            # For embedding collections, let ChromaDB handle batch creation
            # Create empty embeddings list - ChromaDB will generate embeddings
            embeddings = [None] * total_items
            
            try:
                # Use ChromaDB's batch utilities for optimal batching
                batches = create_batches(
                    api=self.client,
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                
                batch_num = 1
                for batch in batches:
                    batch_ids, batch_embeddings, batch_metadatas, batch_documents = batch
                    batch_size_actual = len(batch_ids)
                    
                    try:
                        collection.add(
                            ids=batch_ids,
                            documents=batch_documents,
                            metadatas=batch_metadatas
                            # Let ChromaDB generate embeddings automatically
                        )
                        print(f"  ✅ Added batch {batch_num}: {batch_size_actual} items")
                        batch_num += 1
                    except Exception as e:
                        print(f"  ❌ Error adding batch {batch_num}: {e}")
                        raise
            except Exception as e:
                print(f"❌ Error creating batches for embedding collection: {e}")
                raise
        else:
            print(f"📄 Processing {collection_name} collection WITHOUT embeddings (using manual batching)")
            # For non-embedding collections, use manual batching
            for i in range(0, total_items, effective_batch_size):
                batch_end = min(i + effective_batch_size, total_items)
                batch_documents = documents[i:batch_end]
                batch_metadatas = metadatas[i:batch_end]
                batch_ids = ids[i:batch_end]
                
                try:
                    collection.add(
                        documents=batch_documents,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                    print(f"  ✅ Added batch: {i+1} to {batch_end} ({batch_end-i} items)")
                except Exception as e:
                    print(f"  ❌ Error adding batch {i+1} to {batch_end}: {e}")
                    raise
        
        print(f"✅ Successfully stored {total_items} documents in '{collection_name}'")
        print(f"📊 Used batch size: {effective_batch_size}")
    
    def get_documents(
        self, 
        collection_name: str, 
        ids: Optional[List[str]] = None,
        where: Optional[Dict] = None,
        limit: Optional[int] = None,
        include: List[str] = ["documents", "metadatas"]
    ) -> Dict:
        """
        Retrieve documents from a collection.
        
        Args:
            collection_name: Name of the collection
            ids: Specific document IDs to retrieve
            where: Filter conditions
            limit: Maximum number of documents to return
            include: What to include in results
            
        Returns:
            Query results from ChromaDB
        """
        collection = self.get_or_create_collection(collection_name)
        
        try:
            if ids:
                results = collection.get(ids=ids, include=include)
            else:
                kwargs = {"include": include}
                if where:
                    kwargs["where"] = where
                if limit:
                    kwargs["limit"] = limit
                results = collection.get(**kwargs)
            
            return results
        except Exception as e:
            print(f"Error retrieving documents from {collection_name}: {e}")
            return {"documents": [], "metadatas": [], "ids": []}
    
    def delete_documents(self, collection_name: str, ids: List[str]):
        """Delete specific documents from a collection."""
        collection = self.get_or_create_collection(collection_name)
        
        try:
            collection.delete(ids=ids)
            print(f"✅ Deleted {len(ids)} documents from '{collection_name}'")
        except Exception as e:
            print(f"❌ Error deleting documents from {collection_name}: {e}")
            raise
    
    def query_documents(
        self,
        collection_name: str,
        query_texts: List[str],
        n_results: int = 10,
        where: Optional[Dict] = None,
        include: List[str] = ["documents", "metadatas", "distances"]
    ) -> Dict:
        """
        Query documents using similarity search (requires embeddings).
        
        Args:
            collection_name: Name of the collection
            query_texts: Texts to query for
            n_results: Number of results to return
            where: Filter conditions
            include: What to include in results
            
        Returns:
            Query results from ChromaDB
        """
        collection = self.get_or_create_collection(collection_name)
        
        try:
            kwargs = {
                "query_texts": query_texts,
                "n_results": n_results,
                "include": include
            }
            if where:
                kwargs["where"] = where
                
            results = collection.query(**kwargs)
            return results
        except Exception as e:
            print(f"Error querying {collection_name}: {e}")
            return {"documents": [], "metadatas": [], "distances": []}
    
    # ====================
    # GRANTS-SPECIFIC METHODS
    # ====================
    
    def store_grant(self, grant_data: dict):
        """Store a single grant in the grants collection."""
        grant_id = grant_data["id"]
        
        # Create document text combining title and summary
        document_text = f"Title: {grant_data.get('title', '')}\n\nSummary: {grant_data.get('grant_summary', '')}"
        
        # Prepare the metadata
        metadata = {
            "grant_id": grant_id,
            "title": grant_data.get("title", ""),
            "funder": grant_data.get("funder", ""),
            "funding_amount": grant_data.get("funding_amount", 0),
            "start_year": grant_data.get("start_year", 0),
            "end_year": grant_data.get("end_year", 0),
        }
        
        # Store using general method with optimal batch size
        self.store_documents(
            collection_name="grants",
            documents=[document_text],
            metadatas=[metadata],
            ids=[grant_id],
            batch_size=self.get_optimal_batch_size()
        )
    
    def get_grant(self, grant_id: str) -> Optional[dict]:
        """Retrieve a specific grant by ID."""
        results = self.get_documents("grants", ids=[grant_id])
        
        if results['documents'] and len(results['documents']) > 0:
            metadata = results['metadatas'][0]
            document = results['documents'][0]
            
            # Extract summary from document
            summary_start = document.find("Summary: ") + len("Summary: ")
            grant_summary = document[summary_start:].strip() if summary_start > 8 else ""
            
            return {
                "id": metadata["grant_id"],
                "title": metadata["title"],
                "grant_summary": grant_summary,
                "funder": metadata["funder"],
                "funding_amount": metadata["funding_amount"],
                "start_year": metadata["start_year"],
                "end_year": metadata["end_year"]
            }
        return None
    
    def get_all_grant_ids(self) -> List[str]:
        """Get all grant IDs."""
        results = self.get_documents("grants", include=["metadatas"])
        if results['metadatas']:
            return [meta['grant_id'] for meta in results['metadatas'] if meta and 'grant_id' in meta]
        return []
    
    # ====================
    # KEYWORDS-SPECIFIC METHODS  
    # ====================
    
    def store_keyword(self, keyword_data: dict, keyword_id: str):
        """Store a single keyword in the keywords collection."""
        document_text = f"Term: {keyword_data.get('term', '')}\nDescription: {keyword_data.get('description', '')}\nType: {keyword_data.get('type', 'general')}"
        
        metadata = {
            "term": keyword_data.get("term", ""),
            "description": keyword_data.get("description", ""),
            "type": keyword_data.get("type", "general"),
            "grants": keyword_data.get("grants", [])
        }
        
        # Store using general method
        self.store_documents(
            collection_name="keywords",
            documents=[document_text],
            metadatas=[metadata],
            ids=[keyword_id]
        )
    
    def store_keywords_from_grant(self, grant_id: str, keywords_data: dict):
        """Store extracted keywords for a grant."""
        keywords = keywords_data.get("keywords", [])
        
        documents = []
        metadatas = []
        ids = []
        
        for i, keyword in enumerate(keywords):
            if isinstance(keyword, dict):
                keyword_id = f"{grant_id}_kw_{i}"
                
                if "grants" not in keyword:
                    keyword["grants"] = [grant_id]
                
                document_text = f"Term: {keyword.get('term', '')}\nDescription: {keyword.get('description', '')}\nType: {keyword.get('type', 'general')}"
                
                metadata = {
                    "grant_id": grant_id,
                    "term": keyword.get("term", ""),
                    "description": keyword.get("description", ""),
                    "type": keyword.get("type", "general"),
                    "grants": keyword.get("grants", [grant_id])
                }
                
                documents.append(document_text)
                metadatas.append(metadata)
                ids.append(keyword_id)
        
        if documents:
            self.store_documents(
                collection_name="keywords", 
                documents=documents, 
                metadatas=metadatas, 
                ids=ids,
                batch_size=self.get_optimal_batch_size()
            )
    
    def get_all_keywords(self) -> List[dict]:
        """Retrieve all keywords in the format expected by categorize."""
        results = self.get_documents("keywords", include=["metadatas"])
        
        keywords = []
        if results['metadatas']:
            for meta in results['metadatas']:
                if meta:
                    keywords.append({
                        "term": meta.get("term", ""),
                        "description": meta.get("description", ""),
                        "type": meta.get("type", "general"),
                        "grants": meta.get("grants", [])
                    })
        return keywords
    
    def get_keywords_by_grant(self, grant_id: str) -> List[dict]:
        """Get all keywords associated with a specific grant."""
        results = self.get_documents("keywords", where={"grant_id": grant_id}, include=["metadatas"])
        
        keywords = []
        if results['metadatas']:
            for meta in results['metadatas']:
                if meta:
                    keywords.append({
                        "term": meta.get("term", ""),
                        "description": meta.get("description", ""),
                        "type": meta.get("type", "general"),
                        "grants": meta.get("grants", [])
                    })
        return keywords
    
    # ====================
    # TEXT PROCESSING UTILITIES
    # ====================
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text through preprocessing steps.
        
        Args:
            text: Raw text string
            
        Returns:
            Normalized text string
        """
        # Convert to lowercase
        normalized = text.lower().strip()
        
        # Remove special characters but keep hyphens and spaces
        normalized = re.sub(r'[^\w\s\-]', '', normalized)
        
        # Replace multiple spaces with single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove leading/trailing spaces
        normalized = normalized.strip()
        
        return normalized
    
    def trim_text_to_max_length(self, text: str, max_length: int = 8192) -> str:
        """
        Trim text to fit within the model's maximum token length.
        
        Args:
            text: Input text to trim
            max_length: Maximum character length
            
        Returns:
            Trimmed text that should fit within model limits
        """
        if len(text) <= max_length:
            return text
        return text[:max_length]
    
    def extract_keywords_info_from_data(self, raw_keywords: List[dict]) -> Dict[str, Dict]:
        """
        Extract keyword information from loaded data with enhanced information.
        
        Args:
            raw_keywords: List of keyword dictionaries
            
        Returns:
            Dictionary mapping keyword terms to their information
        """
        keyword_info = defaultdict(lambda: {'frequency': 0, 'keyword_objects': []})
        
        for keyword in raw_keywords:
            term = keyword.get('term', '')
            
            normalized_term = self.normalize_text(term)
            keyword_info[normalized_term]['frequency'] += 1
            keyword_info[normalized_term]['keyword_objects'].append(keyword)
        
        return keyword_info
    
    # ====================
    # EMBEDDING METHODS
    # ====================
    
    def create_embeddings_for_keywords(
        self,
        keyword_info: Dict[str, Dict],
        collection_name: str = "keyword_embeddings",
        max_keywords: Optional[int] = None,
        force_recreate: bool = False,
        max_model_len: int = 8192
    ) -> int:
        """
        Create ChromaDB collection with keyword embeddings.
        
        Args:
            keyword_info: Dictionary of keywords with enhanced information
            collection_name: Name of the ChromaDB collection
            max_keywords: Maximum number of keywords to process
            force_recreate: Force recreation of the collection
            max_model_len: Maximum token length for text truncation
            
        Returns:
            Number of embeddings in the collection
        """
        # Check if we can use existing collection
        if not force_recreate:
            try:
                existing_count = self.count_documents(collection_name)
                if existing_count > 0:
                    print(f"✅ Using existing {collection_name} collection with {existing_count} embeddings")
                    return existing_count
            except Exception as e:
                print(f"⚠️ Failed to check existing collection: {e}")
        
        # Clear existing collection if forcing recreation
        if force_recreate:
            self.clear_collection(collection_name)
        
        # Prepare keywords for embedding
        print("🔄 Preparing keywords for embedding...")
        
        # Sort keywords by frequency and limit if specified
        sorted_all = sorted(keyword_info.items(), key=lambda x: x[1]['frequency'], reverse=True)
        if max_keywords is not None:
            sorted_keywords = sorted_all[:max_keywords]
        else:
            sorted_keywords = sorted_all
        
        keywords = [kw for kw, info in sorted_keywords]
        
        if len(keywords) < 1:
            raise ValueError("No keywords to process")
        
        print(f"📊 Processing {len(keywords)} keywords for embedding")
        
        # Prepare documents for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for keyword in keywords:
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
            trimmed_text = self.trim_text_to_max_length(document_text, max_model_len)
            
            documents.append(trimmed_text)
            metadatas.append({
                "keyword": keyword,
                "frequency": kw_info['frequency'],
                "num_objects": len(keyword_objects)
            })
            # Create safe ID
            safe_keyword = keyword.replace(" ", "_").replace("/", "_").replace("\\", "_")
            ids.append(f"keyword_{safe_keyword}_{hash(keyword) % 10000}")
        
        # Store embeddings using general method with optimal batch size
        self.store_documents(
            collection_name=collection_name, 
            documents=documents, 
            metadatas=metadatas, 
            ids=ids,
            batch_size=self.get_optimal_batch_size()
        )
        
        final_count = self.count_documents(collection_name)
        print(f"✅ Collection now contains {final_count} total embeddings")
        
        return final_count
    
    def estimate_embedding_size(self, num_keywords: int) -> Dict[str, Union[int, str]]:
        """
        Estimate the storage size of embeddings before generating them.
        
        Args:
            num_keywords: Number of keywords to estimate for
            
        Returns:
            Dictionary with size estimates
        """
        # Get embedding dimensions by testing with a small sample
        embedding_function = self._get_embedding_function()
        test_embeddings = embedding_function(["test"])
        embedding_dim = len(test_embeddings[0])
        
        # Calculate size estimates
        bytes_per_embedding = embedding_dim * 4  # 4 bytes for float32
        total_embedding_bytes = num_keywords * bytes_per_embedding
        
        # Convert to human-readable units
        def format_bytes(bytes_val):
            for unit in ['B', 'KB', 'MB', 'GB']:
                if bytes_val < 1024.0:
                    return f"{bytes_val:.1f} {unit}"
                bytes_val /= 1024.0
            return f"{bytes_val:.1f} TB"
        
        # Estimate additional storage for ChromaDB
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
    
    # ====================
    # CATEGORIES METHODS
    # ====================
    
    def store_categories(self, categories_data: dict):
        """Store categorization results in ChromaDB."""
        categories = categories_data.get("categories", [])
        missing_keywords = categories_data.get("missing_keywords", [])
        unknown_keywords = categories_data.get("unknown_keywords", [])
        
        # Clear existing categories
        self.clear_collection("categories")
        
        documents = []
        metadatas = []
        ids = []
        
        # Process main categories
        for i, category in enumerate(categories):
            if isinstance(category, dict):
                category_id = f"category_{i}"
                
                keywords_text = ", ".join(category.get("keywords", []))
                document_text = f"Name: {category.get('name', '')}\nDescription: {category.get('description', '')}\nKeywords: {keywords_text}"
                
                metadata = {
                    "category_name": category.get("name", ""),
                    "description": category.get("description", ""),
                    "keywords": category.get("keywords", []),
                    "category_type": "main"
                }
                
                documents.append(document_text)
                metadatas.append(metadata)
                ids.append(category_id)
        
        # Add special categories for missing and unknown keywords
        if missing_keywords:
            documents.append(f"Missing keywords: {', '.join(missing_keywords)}")
            metadatas.append({
                "category_name": "Missing Keywords",
                "description": "Keywords that were not categorized",
                "keywords": missing_keywords,
                "category_type": "missing"
            })
            ids.append("missing_keywords")
        
        if unknown_keywords:
            documents.append(f"Unknown keywords: {', '.join(unknown_keywords)}")
            metadatas.append({
                "category_name": "Unknown Keywords",
                "description": "Keywords that could not be categorized",
                "keywords": unknown_keywords,
                "category_type": "unknown"
            })
            ids.append("unknown_keywords")
        
        # Store all categories
        if documents:
            self.store_documents(
                collection_name="categories", 
                documents=documents, 
                metadatas=metadatas, 
                ids=ids,
                batch_size=self.get_optimal_batch_size()
            )
        
        print(f"📦 Stored {len(categories)} categories in ChromaDB")
    
    def get_all_categories(self) -> List[dict]:
        """Retrieve all categories from ChromaDB."""
        results = self.get_documents("categories", include=["metadatas"])
        
        categories = []
        if results['metadatas']:
            for meta in results['metadatas']:
                if meta and meta.get('category_type') == 'main':
                    categories.append({
                        "name": meta.get("category_name", ""),
                        "description": meta.get("description", ""),
                        "keywords": meta.get("keywords", [])
                    })
        return categories
    
    # ====================
    # UTILITY METHODS
    # ====================
    
    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics for all collections."""
        stats = {}
        
        # Check known collections
        known_collections = ["grants", "keywords", "categories", "keyword_embeddings"]
        for collection_name in known_collections:
            stats[collection_name] = self.count_documents(collection_name)
        
        return stats
    
    def list_collections(self) -> List[str]:
        """List all existing collections in the database."""
        try:
            collections = self.client.list_collections()
            return [collection.name for collection in collections]
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []
    
    def reset_database(self, confirm: bool = False):
        """Reset the entire database by deleting all collections."""
        if not confirm:
            raise Exception("Database reset requires explicit confirmation")
        
        collection_names = list(self.collections.keys())
        for collection_name in collection_names:
            try:
                self.client.delete_collection(collection_name)
                print(f"✅ Deleted collection: {collection_name}")
            except Exception as e:
                print(f"⚠️ Could not delete collection {collection_name}: {e}")
        
        self.collections = {}
        print("✅ Database reset completed")


# Global instance - use this throughout the application
_db_manager = None


def get_db_manager(vllm_base_url: str = None) -> UnifiedChromaDBManager:
    """Get the global database manager instance."""
    global _db_manager
    
    if _db_manager is None:
        _db_manager = UnifiedChromaDBManager(vllm_base_url)
    else:
        # Update vLLM URL if different
        if vllm_base_url and vllm_base_url != _db_manager.vllm_base_url:
            _db_manager.vllm_base_url = vllm_base_url
            # Clear collections to reinitialize with new embedding function
            _db_manager.collections = {}
    
    return _db_manager


# ====================
# EMBEDDING COMPATIBILITY FUNCTIONS
# ====================

def load_keywords_from_file(keywords_file: Union[str, Path]) -> List[dict]:
    """Load keywords from JSON file."""
    keywords_path = Path(keywords_file)
    with keywords_path.open('r', encoding='utf-8') as f:
        raw_keywords = json.load(f)
    print(f"✅ Loaded {len(raw_keywords)} keywords from {keywords_file}")
    return raw_keywords


def generate_keywords_embeddings(
    keywords_file: str, 
    vllm_url: str = None, 
    max_keywords: Optional[int] = None, 
    force_recreate: bool = False,
    collection_name: str = "keyword_embeddings"
) -> int:
    """Generate embeddings for keywords."""
    db_manager = get_db_manager(vllm_url)
    raw_keywords = load_keywords_from_file(keywords_file)
    keyword_info = db_manager.extract_keywords_info_from_data(raw_keywords)
    
    return db_manager.create_embeddings_for_keywords(
        keyword_info=keyword_info,
        collection_name=collection_name,
        max_keywords=max_keywords, 
        force_recreate=force_recreate
    )


def generate_embeddings_command(
    keywords_file: str = None,
    vllm_url: str = None,
    max_keywords: Optional[int] = None,
    force_recreate: bool = False,
    collection_name: str = "keyword_embeddings"
) -> bool:
    """CLI command wrapper for generating embeddings."""
    try:
        if not keywords_file:
            keywords_file = "data/grants_enriched.json"
        
        count = generate_keywords_embeddings(
            keywords_file=keywords_file,
            vllm_url=vllm_url,
            max_keywords=max_keywords,
            force_recreate=force_recreate,
            collection_name=collection_name
        )
        print(f"✅ Generated {count} embeddings successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to generate embeddings: {e}")
        return False


def estimate_embedding_storage(
    keywords_file: str, 
    vllm_url: str = None,
    max_keywords: Optional[int] = None
) -> Dict[str, Union[int, str]]:
    """Estimate storage requirements for keyword embeddings."""
    db_manager = get_db_manager(vllm_url)
    raw_keywords = load_keywords_from_file(keywords_file)
    keyword_info = db_manager.extract_keywords_info_from_data(raw_keywords)
    
    total_keywords = len(keyword_info)
    num_keywords = min(max_keywords, total_keywords) if max_keywords else total_keywords
    print(f"📊 Estimating storage for {num_keywords} keywords (out of {total_keywords} total)")
    
    return db_manager.estimate_embedding_size(num_keywords)


def test_embedding_generation(text: str, vllm_url: str = None) -> List[float]:
    """Test embedding generation using UnifiedChromaDBManager (with fallback to default)."""
    db_manager = get_db_manager(vllm_url)
    
    try:
        # Try vLLM first
        model_name = db_manager.get_vllm_model_name()
        
        if model_name != "all-MiniLM-L6-v2":
            # Create and test vLLM embedding function
            embedding_func = OpenAIEmbeddingFunction(
                api_key="EMPTY",
                model_name=model_name,
                api_base=vllm_url or db_manager.vllm_base_url
            )
            embeddings = embedding_func([text])
            print(f"✅ Generated embedding using vLLM model: {model_name}")
            return embeddings[0]
        else:
            raise Exception("vLLM not available")
            
    except Exception as e:
        print(f"⚠️  vLLM embedding failed ({e}), using ChromaDB default")
        # Use ChromaDB default embedding function
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        default_ef = DefaultEmbeddingFunction()
        embeddings = default_ef([text])
        print("✅ Generated embedding using ChromaDB default (all-MiniLM-L6-v2)")
        return embeddings[0]


# Legacy compatibility function
def EmbeddingManager(*args, **kwargs):
    """Legacy compatibility - redirects to UnifiedChromaDBManager."""
    print("⚠️  EmbeddingManager is deprecated. Use UnifiedChromaDBManager from database.py instead.")
    return get_db_manager(*args, **kwargs)
