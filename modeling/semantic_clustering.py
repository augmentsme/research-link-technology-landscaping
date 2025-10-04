
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


class EmbeddingGenerator:
    """Generic embedding generator for JSONL data."""
    
    def __init__(self, template_func: callable, model_name: str = 'Qwen/Qwen3-Embedding-0.6B'):
        """
        Initialize embedding generator.
        
        Args:
            template_func: Function that takes a dict and returns formatted text string
            model_name: Name of the sentence transformer model to use
        """
        self.template_func = template_func
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
            data = utils.load_jsonl_file(data_path, as_dataframe=True)
            texts = self._format_text(data)
            embeddings = self.model.encode(texts, show_progress_bar=True)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, embeddings)
            logger.info(f"Saved embeddings to {output_path}")
            return embeddings
        else:
            logger.info(f"Loading cached embeddings from {output_path}")
            return np.load(output_path)
    
    def _format_text(self, data) -> List[str]:
        """Format data into text for embedding generation using template function."""
        texts = []
        for _, record in data.iterrows():
            text = self.template_func(record.to_dict())
            texts.append(text)
        return texts
    
    def _should_regenerate(self, data_path: Path, output_path: Path, force: bool) -> bool:
        """Check if embeddings should be regenerated."""
        if force or not output_path.exists() or not data_path.exists():
            return True
        return output_path.stat().st_mtime <= data_path.stat().st_mtime


class ClusterManager:
    """Generic clustering operations manager."""
    
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
        batch_size: int
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
                    batch_path = output_dir / f"{batch_num:03d}.jsonl"
                    utils.save_jsonl_file(current_batch, batch_path)
                    batch_files.append(batch_path)
                    logger.info(f"Saved batch {batch_num} with {len(current_batch)} items to {batch_path}")
                    
                    current_batch = []
                    batch_num += 1
        
        # Save remaining items in final batch
        if current_batch:
            batch_path = output_dir / f"{batch_num}.jsonl"
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
        
        batch_files = sorted(batch_dir.glob("*.jsonl"))
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
                 data_path: Path,
                 embeddings_path: Optional[Path] = None,
                 clusters_path: Optional[Path] = None,
                 batch_dir: Optional[Path] = None,
                 template_func: Optional[callable] = None,
                 model_name: str = 'Qwen/Qwen3-Embedding-0.6B'):
        """
        Initialize pipeline for semantic clustering.
        
        Args:
            data_path: Path to input JSONL file
            embeddings_path: Path to save/load embeddings (defaults to data_path with .npy extension)
            clusters_path: Path to save/load clusters (defaults to data_path with _clusters.json)
            batch_dir: Directory to save batches (defaults to data_path parent / 'batches')
            template_func: Function to format records as text (defaults to "{name}: {description}")
            model_name: Sentence transformer model name
        """
        self.data_path = data_path
        self.embeddings_path = embeddings_path or data_path.with_suffix('.embeddings.npy')
        self.clusters_path = clusters_path or data_path.with_name(f"{data_path.stem}_clusters.json")
        self.batch_dir = batch_dir or (data_path.parent / 'batches')
        
        # Default template function if none provided
        if template_func is None:
            template_func = lambda x: f"{x.get('name', '')}: {x.get('description', '')}"
        
        self.embedding_generator = EmbeddingGenerator(template_func, model_name)
        self.cluster_manager = ClusterManager()
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
            batch_files = list(self.batch_dir.glob("*.jsonl"))
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
    
    def cluster_data(self, batch_size: int = 50) -> ClusteringResult:
        """Perform semantic clustering of the data."""
        # Load data and embeddings
        data = utils.load_jsonl_file(self.data_path, as_dataframe=True)
        embeddings = np.load(self.embeddings_path)
        
        if len(data) != embeddings.shape[0]:
            raise ValueError(f"Mismatch between data items ({len(data)}) and embeddings ({embeddings.shape[0]})")
        
        # Perform clustering
        result = self.cluster_manager.cluster(embeddings, data, batch_size)
        
        # Save results
        self.clusters_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.clusters_path, 'w') as f:
            json.dump({
                'total_items': result.total_items,
                'n_clusters': result.n_clusters,
                'target_batch_size': result.target_batch_size,
                'clusters': result.clusters
            }, f, indent=2, default=str)
        
        logger.info(f"Clustering results saved to: {self.clusters_path}")
        return result
    
    def generate_batches(self, batch_size: int = 50) -> List[Path]:
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
            total_items=cluster_data['total_items'],
            n_clusters=cluster_data['n_clusters'],
            target_batch_size=cluster_data['target_batch_size'],
            clusters=cluster_data['clusters']
        )
        
        return self.batch_generator.generate_batches(
            clustering_result, 
            self.batch_dir, 
            batch_size
        )
    
    def run_full_pipeline(self, batch_size: int = 50, force_embeddings: bool = False) -> Dict[str, Any]:
        """Run the complete clustering pipeline."""
        logger.info("=== Step 1: Generating embeddings ===")
        embeddings = self.generate_embeddings(force_embeddings)
        
        logger.info("\n=== Step 2: Clustering semantically ===")
        clustering_result = self.cluster_data(batch_size)
        
        logger.info("\n=== Step 3: Generating balanced batches ===")
        batch_files = self.generate_batches(batch_size)
        
        logger.info("\n=== Pipeline complete ===")
        return {
            'embeddings_shape': embeddings.shape,
            'total_items': clustering_result.total_items,
            'semantic_clusters': clustering_result.n_clusters,
            'generated_batches': len(batch_files),
            'target_batch_size': batch_size,
            'batch_files': batch_files
        }
    
    def create_categorization_dataset(self):
        """Create a dataset for categorization from semantic batches."""
        return DatasetBuilder.create_categorization_dataset(self.batch_dir)




# CLI Interface
app = typer.Typer(help="Semantic clustering utilities for structured JSONL data")

@app.command("embeddings")
def generate_embeddings_cmd(
    input_path: str = typer.Argument(..., help="Path to input JSONL file"),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Directory to place all generated files"),
    force: bool = typer.Option(False, "--force", "-f", help="Force regeneration even if cache exists"),
    template: str = typer.Option(None, "--template", "-t", help="Template string (use {name} and {description})"),
    model: str = typer.Option('Qwen/Qwen3-Embedding-0.6B', "--model", "-m", help="Sentence transformer model name")
):
    """Generate embeddings for JSONL data."""
    input_p = Path(input_path)
    
    # Handle output_dir to set embeddings_path
    embeddings_p = None
    if output_dir:
        base_dir = Path(output_dir)
        embeddings_p = base_dir / f"{input_p.stem}.embeddings.npy"
    
    # Parse template if provided
    template_func = None
    if template:
        template_func = lambda x: template.format(**x)
    
    pipeline = Pipeline(
        data_path=input_p,
        embeddings_path=embeddings_p,
        template_func=template_func,
        model_name=model
    )
    embeddings_array = pipeline.generate_embeddings(force)
    print(f"Generated embeddings with shape: {embeddings_array.shape}")
    print(f"Saved to: {pipeline.embeddings_path}")


@app.command("cluster")
def cluster_cmd(
    input_path: str = typer.Argument(..., help="Path to input JSONL file"),
    batch_size: int = typer.Option(50, "--batch-size", "-b", help="Target number of items per cluster"),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Directory to place all generated files"),
    template: str = typer.Option(None, "--template", "-t", help="Template string (use {name} and {description})")
):
    """Perform semantic clustering of data."""
    input_p = Path(input_path)
    
    # Handle output_dir to set paths
    embeddings_p = None
    clusters_p = None
    if output_dir:
        base_dir = Path(output_dir)
        embeddings_p = base_dir / f"{input_p.stem}.embeddings.npy"
        clusters_p = base_dir / f"{input_p.stem}_clusters.json"
    
    # Parse template if provided
    template_func = None
    if template:
        template_func = lambda x: template.format(**x)
    
    pipeline = Pipeline(
        data_path=input_p,
        embeddings_path=embeddings_p,
        clusters_path=clusters_p,
        template_func=template_func
    )
    result = pipeline.cluster_data(batch_size)
    
    print(f"\nClustering summary:")
    print(f"  Total items: {result.total_items}")
    print(f"  Number of clusters: {result.n_clusters}")
    print(f"  Target batch size: {batch_size}")
    print(f"  Saved to: {pipeline.clusters_path}")


@app.command("generate-batches")
def generate_batches_cmd(
    input_path: str = typer.Argument(..., help="Path to input JSONL file"),
    batch_size: int = typer.Option(50, "--batch-size", "-b", help="Target size for each batch"),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Directory to place all generated files")
):
    """Generate balanced batches from semantic clusters."""
    input_p = Path(input_path)
    
    # Handle output_dir to set paths
    clusters_p = None
    batch_dir = None
    if output_dir:
        base_dir = Path(output_dir)
        clusters_p = base_dir / f"{input_p.stem}_clusters.json"
        batch_dir = base_dir / "batches"
    
    pipeline = Pipeline(
        data_path=input_p,
        clusters_path=clusters_p,
        batch_dir=batch_dir
    )
    batch_files = pipeline.generate_batches(batch_size)
    
    print(f"\nGenerated {len(batch_files)} batch files in {pipeline.batch_dir}")
    for i, batch_file in enumerate(batch_files):
        print(f"  Batch {i}: {batch_file.name}")


@app.command("pipeline")
def pipeline(
    input_path: str = typer.Argument(..., help="Path to input JSONL file"),
    output_dir: str = typer.Argument(..., help="Directory to place all generated files (embeddings, clusters, batches)"),
    # output_dir: str = typer.Option(None, "--output-dir", "-o", help="Directory to place all generated files (embeddings, clusters, batches)"),
    batch_size: int = typer.Option(50, "--batch-size", "-b", help="Target size for each batch"),
    force_embeddings: bool = typer.Option(False, "--force", "-f", help="Force regeneration of embeddings"),
    template: str = typer.Option(None, "--template", "-t", help="Template string (use {name} and {description})"),
    model: str = typer.Option('Qwen/Qwen3-Embedding-0.6B', "--model", "-m", help="Sentence transformer model name")
):
    """Run the complete clustering pipeline: embeddings -> clustering -> batches."""
    input_p = Path(input_path)
    
    # Handle output_dir to set all paths
    embeddings_p = None
    clusters_p = None
    batch_dir = None
    if output_dir:
        base_dir = Path(output_dir)
        embeddings_p = base_dir / f"{input_p.stem}.embeddings.npy"
        clusters_p = base_dir / f"{input_p.stem}_clusters.json"
        batch_dir = base_dir / "batches"
    
    # Parse template if provided
    template_func = None
    if template:
        template_func = lambda x: template.format(**x)
    
    pipeline = Pipeline(
        data_path=input_p,
        embeddings_path=embeddings_p,
        clusters_path=clusters_p,
        batch_dir=batch_dir,
        template_func=template_func,
        model_name=model
    )
    results = pipeline.run_full_pipeline(batch_size, force_embeddings)
    
    print(f"\n=== Pipeline Summary ===")
    print(f"  Total items: {results['total_items']}")
    print(f"  Semantic clusters: {results['semantic_clusters']}")
    print(f"  Generated batches: {results['generated_batches']}")
    print(f"  Target batch size: {results['target_batch_size']}")
    print(f"  Embeddings: {pipeline.embeddings_path}")
    print(f"  Clusters: {pipeline.clusters_path}")
    print(f"  Batches: {pipeline.batch_dir}")


if __name__ == "__main__":
    app()

