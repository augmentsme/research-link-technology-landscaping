"""
Semantic clustering system for category proposals and keywords.

This module provides a class-based architecture for generating embeddings and performing
semantic clustering operations on both keywords and category proposals. It supports
balanced batch generation for improved categorization workflows.

## Usage Examples:

### For Keywords (used in categorise.py):

```python
from semantic_clustering import KeywordPipeline

# Run complete pipeline
pipeline = KeywordPipeline()
results = pipeline.run_full_pipeline(batch_size=250)

# Or use individual components
keyword_generator = KeywordEmbeddingGenerator()
embeddings = keyword_generator.generate()
```

### For Category Proposals:

```python
from semantic_clustering import CategoryPipeline

pipeline = CategoryPipeline()
results = pipeline.run_full_pipeline(batch_size=50)
```

## Main Classes:

- `Pipeline`: Generic base class for semantic clustering workflows
- `KeywordPipeline`: Complete workflow for keyword clustering
- `CategoryPipeline`: Complete workflow for category proposal clustering
- `KeywordEmbeddingGenerator`: Generates embeddings for keywords
- `CategoryEmbeddingGenerator`: Generates embeddings for category proposals
- `SemanticClusterManager`: Handles clustering operations
- `BatchGenerator`: Creates balanced batches from clusters
- `DatasetBuilder`: Builds datasets for categorization
"""

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from typing import List, Dict, Any, Optional, Protocol, Union
from abc import ABC, abstractmethod
import config
import utils
import typer
import json
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """Data class for clustering results."""
    total_items: int
    n_clusters: int
    target_batch_size: int
    clusters: Dict[str, List[Dict]]
    metadata: Optional[Dict[str, Any]] = None


class EmbeddingGenerator(ABC):
    """Abstract base class for generating embeddings."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def generate(self, data_path: Path, output_path: Path, force_regenerate: bool = False) -> np.ndarray:
        """Generate embeddings for the data."""
        if self._should_regenerate(data_path, output_path, force_regenerate):
            logger.info(f"Generating embeddings for {data_path}")
            data = self._load_data(data_path)
            texts = self._format_text(data)
            embeddings = self.model.encode(texts, show_progress_bar=True)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, embeddings)
            logger.info(f"Saved embeddings to {output_path}")
            return embeddings
        else:
            logger.info(f"Loading cached embeddings from {output_path}")
            return np.load(output_path)
    
    @abstractmethod
    def _load_data(self, data_path: Path) -> Any:
        """Load data from path."""
        pass
    
    @abstractmethod
    def _format_text(self, data: Any) -> List[str]:
        """Format data into text for embedding generation."""
        pass
    
    def _should_regenerate(self, data_path: Path, output_path: Path, force: bool) -> bool:
        """Check if embeddings should be regenerated."""
        if force or not output_path.exists() or not data_path.exists():
            return True
        return output_path.stat().st_mtime <= data_path.stat().st_mtime


class ClusterManager(ABC):
    """Abstract base class for clustering operations."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def cluster(self, embeddings: np.ndarray, data: Any, batch_size: int) -> ClusteringResult:
        """Perform clustering on embeddings."""
        n_clusters = self._calculate_clusters(len(data), batch_size)
        
        logger.info(f"Clustering {len(data)} items into {n_clusters} clusters (target batch_size={batch_size})")
        
        if len(data) <= batch_size:
            cluster_labels = [0] * len(data)
        else:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=self.random_state, n_init='auto', batch_size=min(1024, len(data)))
            cluster_labels = kmeans.fit_predict(embeddings)
        
        clusters = self._organize_clusters(data, cluster_labels)
        
        return ClusteringResult(
            total_items=len(data),
            n_clusters=len(clusters),
            target_batch_size=batch_size,
            clusters=clusters
        )
    
    def _calculate_clusters(self, data_size: int, batch_size: int) -> int:
        """Calculate optimal number of clusters."""
        n_clusters = max(1, data_size // batch_size)
        if data_size % batch_size > 0:
            n_clusters += 1
        return n_clusters
    
    @abstractmethod
    def _organize_clusters(self, data: Any, labels: np.ndarray) -> Dict[str, List[Dict]]:
        """Organize data into clusters based on labels."""
        pass


class KeywordEmbeddingGenerator(EmbeddingGenerator):
    """Generates embeddings for keywords using the Keywords template."""
    
    def _load_data(self, data_path: Path) -> List[Dict]:
        """Load keywords from JSONL file."""
        return utils.load_jsonl_file(data_path, as_dataframe=True)
    
    def _format_text(self, data) -> List[str]:
        """Format keywords using the Keywords template."""
        texts = []
        for _, record in data.iterrows():
            text = config.Keywords.template(record.to_dict())
            texts.append(text)
        return texts


class CategoryEmbeddingGenerator(EmbeddingGenerator):
    """Generates embeddings for category proposals using the Categories template."""
    
    def _load_data(self, data_path: Path) -> List[Dict]:
        """Load category proposals from JSONL file."""
        return utils.load_jsonl_file(data_path, as_dataframe=True)
    
    def _format_text(self, data) -> List[str]:
        """Format categories using the Categories template."""
        texts = []
        for _, record in data.iterrows():
            text = config.Categories.template(record.to_dict())
            texts.append(text)
        return texts


class SemanticClusterManager(ClusterManager):
    """Manages semantic clustering operations for general data."""
    
    def _organize_clusters(self, data: Any, labels: np.ndarray) -> Dict[str, List[Dict]]:
        """Organize data into clusters based on labels."""
        clusters = {}
        
        # Convert DataFrame to list of dicts if needed
        if hasattr(data, 'to_dict'):
            data_records = data.to_dict(orient='records')
        else:
            data_records = data
        
        for i, (item, cluster_id) in enumerate(zip(data_records, labels)):
            cluster_key = f"cluster_{cluster_id}"
            if cluster_key not in clusters:
                clusters[cluster_key] = []
            
            item_with_meta = item.copy()
            item_with_meta['embedding_index'] = i
            clusters[cluster_key].append(item_with_meta)
        
        # Log cluster statistics
        cluster_sizes = [len(cluster) for cluster in clusters.values()]
        logger.info(f"Created {len(clusters)} clusters with sizes: {cluster_sizes}")
        logger.info(f"Average cluster size: {np.mean(cluster_sizes):.1f}")
        logger.info(f"Min/Max cluster sizes: {min(cluster_sizes)}/{max(cluster_sizes)}")
        
        return clusters


class BatchGenerator:
    """Generates balanced batches from clustering results."""
    
    @staticmethod
    def generate_batches(
        clustering_result: ClusteringResult,
        output_dir: Path,
        batch_size: int,
        file_prefix: str = "batch"
    ) -> List[Path]:
        """Generate balanced batches from clustering results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        batch_files = []
        current_batch = []
        batch_num = 0
        
        for cluster_id, cluster_items in clustering_result.clusters.items():
            for item in cluster_items:
                # Remove embedding_index metadata before saving
                clean_item = {k: v for k, v in item.items() if k != 'embedding_index'}
                current_batch.append(clean_item)
                
                if len(current_batch) >= batch_size:
                    batch_path = output_dir / f"{file_prefix}_{batch_num:03d}.jsonl"
                    utils.save_jsonl_file(current_batch, batch_path)
                    batch_files.append(batch_path)
                    logger.info(f"Saved batch {batch_num} with {len(current_batch)} items to {batch_path}")
                    
                    current_batch = []
                    batch_num += 1
        
        # Save remaining items in final batch
        if current_batch:
            batch_path = output_dir / f"{file_prefix}_{batch_num:03d}.jsonl"
            utils.save_jsonl_file(current_batch, batch_path)
            batch_files.append(batch_path)
            logger.info(f"Saved final batch {batch_num} with {len(current_batch)} items to {batch_path}")
        
        logger.info(f"Generated {len(batch_files)} batches")
        return batch_files


class DatasetBuilder:
    """Builds datasets for categorization from semantic batches."""
    
    @staticmethod
    def load_batches(batch_dir: Path) -> List[List[Dict]]:
        """Load semantic batches for categorization."""
        if not batch_dir.exists():
            raise FileNotFoundError(f"Batch directory does not exist: {batch_dir}")
        
        batch_files = sorted(batch_dir.glob("*batch_*.jsonl"))
        if not batch_files:
            raise FileNotFoundError(f"No batch files found in {batch_dir}")
        
        batches = []
        for batch_file in batch_files:
            batch_data = utils.load_jsonl_file(batch_file, as_dataframe=False)
            batches.append(batch_data)
            logger.info(f"Loaded batch from {batch_file.name} with {len(batch_data)} items")
        
        logger.info(f"Loaded {len(batches)} batches with total {sum(len(b) for b in batches)} items")
        return batches
    
    @staticmethod
    def create_categorization_dataset(batch_dir: Path):
        """Create a dataset from semantic batches for categorization."""
        from inspect_ai.dataset import Sample, MemoryDataset
        
        batches = DatasetBuilder.load_batches(batch_dir)
        
        samples = []
        for idx, keywords in enumerate(batches):
            entries = []
            for kw in keywords:
                entries.append(config.Keywords.template(kw))
            
            sample = Sample(
                id=f"semantic_batch_{idx}",
                input="\n".join(entries),
                metadata={"keywords": [kw['name'] for kw in keywords]}
            )
            samples.append(sample)
        
        return MemoryDataset(samples)


class Pipeline:
    """Generic pipeline for semantic clustering workflows."""
    
    def __init__(self, 
                 data_path: Optional[Path] = None,
                 embeddings_path: Optional[Path] = None,
                 clusters_path: Optional[Path] = None,
                 batch_dir: Optional[Path] = None,
                 embedding_generator: Optional[EmbeddingGenerator] = None,
                 batch_prefix: str = "batch",
                 data_type: str = "items"):
        self.data_path = data_path
        self.embeddings_path = embeddings_path
        self.clusters_path = clusters_path
        self.batch_dir = batch_dir
        self.embedding_generator = embedding_generator
        self.batch_prefix = batch_prefix
        self.data_type = data_type
        
        self.cluster_manager = SemanticClusterManager()
        self.batch_generator = BatchGenerator()
    
    def _should_regenerate_clusters(self) -> bool:
        """Check if clusters should be regenerated based on embeddings modification time."""
        if not self.clusters_path.exists() or not self.embeddings_path.exists():
            return True
        return self.clusters_path.stat().st_mtime <= self.embeddings_path.stat().st_mtime
    
    def _should_regenerate_batches(self) -> bool:
        """Check if batches should be regenerated based on clusters modification time."""
        if not self.clusters_path.exists():
            return True
        
        # Check if any existing batch files are older than clusters
        if self.batch_dir.exists():
            batch_files = list(self.batch_dir.glob(f"{self.batch_prefix}_*.jsonl"))
            if batch_files:
                # Check if any batch file is older than clusters
                clusters_mtime = self.clusters_path.stat().st_mtime
                for batch_file in batch_files:
                    if batch_file.stat().st_mtime <= clusters_mtime:
                        return True
                return False
        return True
    
    def generate_embeddings(self, force_regenerate: bool = False) -> np.ndarray:
        """Generate embeddings for the data."""
        return self.embedding_generator.generate(
            self.data_path, 
            self.embeddings_path, 
            force_regenerate
        )
    
    def cluster_data(self, batch_size: int = 250) -> ClusteringResult:
        """Perform semantic clustering of the data."""
        # Load data and embeddings
        data = utils.load_jsonl_file(self.data_path, as_dataframe=True)
        embeddings = np.load(self.embeddings_path)
        
        if len(data) != embeddings.shape[0]:
            raise ValueError(f"Mismatch between {self.data_type} ({len(data)}) and embeddings ({embeddings.shape[0]})")
        
        # Perform clustering
        result = self.cluster_manager.cluster(embeddings, data, batch_size)
        
        # Save results
        self.clusters_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.clusters_path, 'w') as f:
            json.dump({
                f'total_{self.data_type}': result.total_items,
                'n_clusters': result.n_clusters,
                'target_batch_size': result.target_batch_size,
                'clusters': result.clusters
            }, f, indent=2, default=str)
        
        logger.info(f"{self.data_type.title()} clustering results saved to: {self.clusters_path}")
        return result
    
    def generate_batches(self, batch_size: int = 250) -> List[Path]:
        """Generate balanced batches from clustering results."""
        # Check if clusters need to be regenerated due to newer embeddings
        if self._should_regenerate_clusters():
            logger.info(f"Embeddings are newer than clusters. Regenerating clusters first...")
            self.cluster_data(batch_size)
        
        # Check if batches need to be regenerated due to newer clusters
        elif self._should_regenerate_batches():
            logger.info(f"Clusters are newer than existing batches. Regenerating batches...")
        
        with open(self.clusters_path, 'r') as f:
            cluster_data = json.load(f)
        
        clustering_result = ClusteringResult(
            total_items=cluster_data[f'total_{self.data_type}'],
            n_clusters=cluster_data['n_clusters'],
            target_batch_size=cluster_data['target_batch_size'],
            clusters=cluster_data['clusters']
        )
        
        return self.batch_generator.generate_batches(
            clustering_result, 
            self.batch_dir, 
            batch_size, 
            self.batch_prefix
        )
    
    def run_full_pipeline(self, batch_size: int = 250, force_embeddings: bool = False) -> Dict[str, Any]:
        """Run the complete clustering pipeline."""
        logger.info(f"=== Step 1: Generating {self.data_type} embeddings ===")
        embeddings = self.generate_embeddings(force_embeddings)
        
        logger.info(f"\n=== Step 2: Clustering {self.data_type} semantically ===")
        clustering_result = self.cluster_data(batch_size)
        
        logger.info("\n=== Step 3: Generating balanced batches ===")
        batch_files = self.generate_batches(batch_size)
        
        logger.info(f"\n=== Pipeline complete ===")
        return {
            'embeddings_shape': embeddings.shape,
            f'total_{self.data_type}': clustering_result.total_items,
            'semantic_clusters': clustering_result.n_clusters,
            'generated_batches': len(batch_files),
            'target_batch_size': batch_size,
            'batch_files': batch_files
        }
    
    def create_categorization_dataset(self):
        """Create a dataset for categorization from semantic batches."""
        return DatasetBuilder.create_categorization_dataset(self.batch_dir)


class KeywordPipeline(Pipeline):
    """Complete workflow for keyword semantic clustering."""
    
    def __init__(self, 
                 keywords_path: Optional[Path] = None,
                 embeddings_path: Optional[Path] = None,
                 clusters_path: Optional[Path] = None,
                 batch_dir: Optional[Path] = None):
        super().__init__(
            data_path=keywords_path or config.Keywords.keywords_path,
            embeddings_path=embeddings_path or config.Keywords.keyword_embeddings_path,
            clusters_path=clusters_path or config.Keywords.semantic_clusters_path,
            batch_dir=batch_dir or config.Keywords.batch_dir,
            embedding_generator=KeywordEmbeddingGenerator(),
            batch_prefix="keyword_batch",
            data_type="keywords"
        )
    
    def cluster_keywords(self, batch_size: int = 250) -> ClusteringResult:
        """Perform semantic clustering of keywords (backward compatibility)."""
        return self.cluster_data(batch_size)


class CategoryPipeline(Pipeline):
    """Complete workflow for category proposal clustering."""
    
    def __init__(self,
                  proposal_path: Optional[Path] = None,
                 embeddings_path: Optional[Path] = None,
                 clusters_path: Optional[Path] = None,
                 batch_dir: Optional[Path] = None):
        super().__init__(
            data_path= proposal_path or config.Categories.category_proposal_path,
            embeddings_path=embeddings_path or config.Categories.category_embeddings_path,
            clusters_path=clusters_path or config.Categories.semantic_clusters_path,
            batch_dir=batch_dir or config.Categories.batch_dir,
            embedding_generator=CategoryEmbeddingGenerator(),
            batch_prefix="category_batch",
            data_type="categories"
        )
    
    def cluster_categories(self, batch_size: int = 50) -> ClusteringResult:
        """Perform semantic clustering of category proposals (backward compatibility)."""
        return self.cluster_data(batch_size)


# CLI Interface
app = typer.Typer(help="Semantic clustering utilities for category proposals and keywords")
keywords_app = typer.Typer(help="Keyword semantic clustering operations")
categories_app = typer.Typer(help="Category semantic clustering operations")

app.add_typer(keywords_app, name="keywords")
app.add_typer(categories_app, name="categories")


# Keywords subcommands
@keywords_app.command("embeddings")
def keyword_embeddings(
    force: bool = typer.Option(False, "--force", help="Force regeneration even if cache exists"),
    keywords_path: str = typer.Option(None, "--keywords-path", help="Path to keywords JSONL file"),
    output_path: str = typer.Option(None, "--output-path", help="Path to save embeddings npy file")
):
    """Generate embeddings for keywords."""
    pipeline = KeywordPipeline(
        keywords_path=Path(keywords_path) if keywords_path else None,
        embeddings_path=Path(output_path) if output_path else None
    )
    embeddings_array = pipeline.generate_embeddings(force)
    print(f"Generated keyword embeddings with shape: {embeddings_array.shape}")


@keywords_app.command("cluster")
def cluster_keywords(
    batch_size: int = typer.Option(250, "--batch-size", "-b", help="Target number of keywords per cluster"),
    keywords_path: str = typer.Option(None, "--keywords-path", help="Path to keywords JSONL file"),
    embeddings_path: str = typer.Option(None, "--embeddings-path", help="Path to keyword embeddings npy file"),
    output_path: str = typer.Option(None, "--output-path", help="Path to save clustering results")
):
    """Perform semantic clustering of keywords for balanced categorization batches."""
    pipeline = KeywordPipeline(
        keywords_path=Path(keywords_path) if keywords_path else None,
        embeddings_path=Path(embeddings_path) if embeddings_path else None,
        clusters_path=Path(output_path) if output_path else None
    )
    result = pipeline.cluster_keywords(batch_size)
    
    print(f"\nKeyword clustering summary:")
    print(f"  Total keywords: {result.total_items}")
    print(f"  Number of clusters: {result.n_clusters}")
    print(f"  Target batch size: {batch_size}")


@keywords_app.command("generate-batches")
def generate_keyword_batches(
    batch_size: int = typer.Option(250, "--batch-size", "-b", help="Target size for each batch"),
    keywords_path: str = typer.Option(None, "--keywords-path", help="Path to keywords JSONL file"),
    clusters_path: str = typer.Option(None, "--clusters-path", help="Path to clustering results JSON file"),
    output_dir: str = typer.Option(None, "--output-dir", help="Directory to save batch files")
):
    """Generate balanced keyword batches from semantic clusters."""
    pipeline = KeywordPipeline(
        keywords_path=Path(keywords_path) if keywords_path else None,
        clusters_path=Path(clusters_path) if clusters_path else None,
        batch_dir=Path(output_dir) if output_dir else None
    )
    batch_files = pipeline.generate_batches(batch_size)
    
    print(f"\nGenerated {len(batch_files)} batch files")
    for i, batch_file in enumerate(batch_files):
        print(f"  Batch {i}: {batch_file}")


@keywords_app.command("full-pipeline")
def full_keyword_pipeline(
    batch_size: int = typer.Option(250, "--batch-size", "-b", help="Target size for each batch"),
    force_embeddings: bool = typer.Option(False, "--force-embeddings", help="Force regeneration of embeddings"),
    keywords_path: str = typer.Option(None, "--keywords-path", help="Path to keywords JSONL file")
):
    """Run the complete keyword clustering pipeline: embeddings -> clustering -> batches."""
    pipeline = KeywordPipeline(keywords_path=Path(keywords_path) if keywords_path else None)
    results = pipeline.run_full_pipeline(batch_size, force_embeddings)
    
    print(f"\n=== Pipeline Summary ===")
    print(f"  Total keywords: {results['total_keywords']}")
    print(f"  Semantic clusters: {results['semantic_clusters']}")
    print(f"  Generated batches: {results['generated_batches']}")
    print(f"  Target batch size: {results['target_batch_size']}")


# Categories subcommands
@categories_app.command("embeddings")
def category_embeddings(
    force: bool = typer.Option(False, "--force", help="Force regeneration even if cache exists"),
     proposal_path: str = typer.Option(None, "--proposal-path", help="Path to category proposals JSONL file"),
    output_path: str = typer.Option(None, "--output-path", help="Path to save embeddings npy file")
):
    """Generate embeddings for category proposals."""
    pipeline = CategoryPipeline(
         proposal_path=Path(proposal_path) if  proposal_path else None,
        embeddings_path=Path(output_path) if output_path else None
    )
    embeddings_array = pipeline.generate_embeddings(force)
    print(f"Generated category embeddings with shape: {embeddings_array.shape}")


@categories_app.command("cluster")
def cluster_categories(
    batch_size: int = typer.Option(50, "--batch-size", "-b", help="Target number of categories per cluster"),
     proposal_path: str = typer.Option(None, "--proposal-path", help="Path to category proposals JSONL file"),
    embeddings_path: str = typer.Option(None, "--embeddings-path", help="Path to category embeddings npy file"),
    output_path: str = typer.Option(None, "--output-path", help="Path to save clustering results")
):
    """Perform semantic clustering of category proposals for balanced categorization batches."""
    pipeline = CategoryPipeline(
         proposal_path=Path( proposal_path) if  proposal_path else None,
        embeddings_path=Path(embeddings_path) if embeddings_path else None,
        clusters_path=Path(output_path) if output_path else None
    )
    result = pipeline.cluster_categories(batch_size)
    
    print(f"\nCategory clustering summary:")
    print(f"  Total categories: {result.total_items}")
    print(f"  Number of clusters: {result.n_clusters}")
    print(f"  Target batch size: {batch_size}")


@categories_app.command("generate-batches")
def generate_category_batches(
    batch_size: int = typer.Option(50, "--batch-size", "-b", help="Target size for each batch"),
    proposal_path: str = typer.Option(None, "--proposal-path", help="Path to category proposals JSONL file"),
    clusters_path: str = typer.Option(None, "--clusters-path", help="Path to clustering results JSON file"),
    output_dir: str = typer.Option(None, "--output-dir", help="Directory to save batch files"),
    force_embeddings: bool = typer.Option(False, "--force-embeddings", help="Force regeneration of embeddings and clusters")
):
    """Generate balanced category batches from semantic clusters."""
    pipeline = CategoryPipeline(
        proposal_path=Path(proposal_path) if proposal_path else None,
        clusters_path=Path(clusters_path) if clusters_path else None,
        batch_dir=Path(output_dir) if output_dir else None
    )
    
    if force_embeddings:
        # Run full pipeline to regenerate embeddings, clusters, and batches
        results = pipeline.run_full_pipeline(batch_size, force_embeddings)
        batch_files = results.get('batch_files', [])
        print(f"\n=== Pipeline Summary ===")
        print(f"  Total categories: {results.get('total_categories', 'N/A')}")
        print(f"  Semantic clusters: {results.get('semantic_clusters', 'N/A')}")
        print(f"  Generated batches: {results.get('generated_batches', len(batch_files))}")
    else:
        # Only generate batches from existing clusters
        batch_files = pipeline.generate_batches(batch_size)
    
    print(f"\nGenerated {len(batch_files)} batch files")
    for i, batch_file in enumerate(batch_files):
        print(f"  Batch {i}: {batch_file}")


@categories_app.command("full-pipeline")
def full_category_pipeline(
    batch_size: int = typer.Option(50, "--batch-size", "-b", help="Target size for each batch"),
    force_embeddings: bool = typer.Option(False, "--force-embeddings", help="Force regeneration of embeddings"),
    proposal_path: str = typer.Option(None, "--proposal-path", help="Path to category proposals JSONL file")
):
    """Run the complete category clustering pipeline: embeddings -> clustering -> batches."""
    pipeline = CategoryPipeline(proposal_path=Path(proposal_path) if proposal_path else None)
    results = pipeline.run_full_pipeline(batch_size, force_embeddings)
    
    print(f"\n=== Pipeline Summary ===")
    print(f"  Total categories: {results['total_categories']}")
    print(f"  Semantic clusters: {results['semantic_clusters']}")
    print(f"  Generated batches: {results['generated_batches']}")
    print(f"  Target batch size: {results['target_batch_size']}")


if __name__ == "__main__":
    app()
