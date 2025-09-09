"""
Simplified embedding generation and clustering for category proposals.
This module generates embeddings using the Categories template and provides clustering within FOR divisions.
"""

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from typing import List, Dict, Any
import config
import utils
import typer
import json


def generate_embeddings(proposal_path: Path = None, embeddings_path: Path = None, force_regenerate: bool = False) -> np.ndarray:
    """
    Generate embeddings for category proposals using the Categories template.
    
    Args:
        proposal_path: Path to the category proposals JSONL file (defaults to config value)
        embeddings_path: Path to save embeddings npy file (defaults to config value)
        force_regenerate: Force regeneration even if cache exists
        
    Returns:
        Numpy array of embeddings
    """
    if proposal_path is None:
        proposal_path = config.Categories.category_proposal_path
        
    if embeddings_path is None:
        embeddings_path = config.Categories.embeddings_cache_path.with_suffix('.npy')
    
    # Check if we need to regenerate
    if not force_regenerate and embeddings_path.exists() and proposal_path.exists():
        if embeddings_path.stat().st_mtime > proposal_path.stat().st_mtime:
            return np.load(embeddings_path)
    
    # Load proposals
    proposals = utils.load_jsonl_file(proposal_path, as_dataframe=True)
    
    # Generate text using Categories template
    category_texts = []
    for _, record in proposals.iterrows():
        category_text = config.Categories.template(record.to_dict())
        category_texts.append(category_text)
    
    # Initialize model and generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(category_texts, show_progress_bar=True)
    
    # Save embeddings
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    
    return embeddings


def load_embeddings(embeddings_path: Path = None) -> np.ndarray:
    """
    Load embeddings from npy file.
    
    Args:
        embeddings_path: Path to embeddings npy file (defaults to config value)
        
    Returns:
        Numpy array of embeddings
    """
    if embeddings_path is None:
        embeddings_path = config.Categories.embeddings_cache_path.with_suffix('.npy')
    
    return np.load(embeddings_path)


def cluster_by_for_division(
    proposal_path: Path = None, 
    embeddings_path: Path = None,
    clusters_output_path: Path = None,
    batch_size: int = 50
) -> Dict[str, Any]:
    """
    Perform clustering within each FOR (Field of Research) division.
    
    Args:
        proposal_path: Path to category proposals JSONL file
        embeddings_path: Path to embeddings npy file
        clusters_output_path: Path to save clustering results
        batch_size: Expected number of proposals per cluster (will calculate n_clusters from this)
        
    Returns:
        Dictionary containing clustering results by FOR division
    """
    if proposal_path is None:
        proposal_path = config.Categories.category_proposal_path
    
    if embeddings_path is None:
        embeddings_path = config.Categories.embeddings_cache_path.with_suffix('.npy')
        
    if clusters_output_path is None:
        clusters_output_path = config.Categories.category_dir / "clusters_by_for.json"
    
    # Load proposals and embeddings
    proposals = utils.load_jsonl_file(proposal_path, as_dataframe=True)
    embeddings = np.load(embeddings_path)
    
    if len(proposals) != embeddings.shape[0]:
        raise ValueError(f"Mismatch between proposals ({len(proposals)}) and embeddings ({embeddings.shape[0]})")
    
    clustering_results = {}
    
    # Group by field_of_research and cluster within each group
    for field_of_research, group_df in proposals.groupby('field_of_research'):
        group_indices = group_df.index.tolist()
        group_embeddings = embeddings[group_indices]
        group_proposals = group_df.to_dict(orient='records')
        
        # Calculate number of clusters based on batch size
        n_clusters = max(1, len(group_proposals) // batch_size)
        if len(group_proposals) % batch_size > 0:
            n_clusters += 1
        
        print(f"Clustering {len(group_proposals)} proposals for FOR: {field_of_research} (batch_size={batch_size}, n_clusters={n_clusters})")
        
        if len(group_proposals) <= batch_size:
            # If we have fewer proposals than batch size, use single cluster
            cluster_labels = [0] * len(group_proposals)
            n_clusters = 1
        else:
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(group_embeddings)
        
        # Organize results by cluster
        clusters = {}
        for i, (proposal, cluster_id) in enumerate(zip(group_proposals, cluster_labels)):
            cluster_key = f"cluster_{cluster_id}"
            if cluster_key not in clusters:
                clusters[cluster_key] = []
            
            # Add proposal with its embedding index for reference
            proposal_with_meta = proposal.copy()
            proposal_with_meta['embedding_index'] = group_indices[i]
            clusters[cluster_key].append(proposal_with_meta)
        
        clustering_results[field_of_research] = {
            'total_proposals': len(group_proposals),
            'n_clusters': len(clusters),
            'batch_size': batch_size,
            'clusters': clusters
        }
        
        # Show cluster sizes
        cluster_sizes = [len(cluster_categories) for cluster_categories in clusters.values()]
        print(f"  Created {len(clusters)} clusters with sizes: {cluster_sizes}")
    
    # Save results
    clusters_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(clusters_output_path, 'w') as f:
        json.dump(clustering_results, f, indent=2, default=str)
    
    print(f"Clustering results saved to: {clusters_output_path}")
    return clustering_results


# CLI setup
app = typer.Typer(help="Semantic clustering utilities for category proposals")


@app.command()
def embeddings(
    force: bool = typer.Option(False, "--force", help="Force regeneration even if cache exists"),
    proposal_path: str = typer.Option(None, "--proposal-path", help="Path to category proposals JSONL file"),
    output_path: str = typer.Option(None, "--output-path", help="Path to save embeddings npy file")
):
    """Generate embeddings for category proposals."""
    proposal_path_obj = Path(proposal_path) if proposal_path else None
    output_path_obj = Path(output_path) if output_path else None
    
    embeddings_array = generate_embeddings(
        proposal_path=proposal_path_obj,
        embeddings_path=output_path_obj,
        force_regenerate=force
    )
    print(f"Generated embeddings with shape: {embeddings_array.shape}")


@app.command()
def cluster(
    batch_size: int = typer.Option(50, "--batch-size", "-b", help="Expected number of proposals per cluster"),
    proposal_path: str = typer.Option(None, "--proposal-path", help="Path to category proposals JSONL file"),
    embeddings_path: str = typer.Option(None, "--embeddings-path", help="Path to embeddings npy file"),
    output_path: str = typer.Option(None, "--output-path", help="Path to save clustering results")
):
    """Perform clustering within each FOR (Field of Research) division based on batch size."""
    proposal_path_obj = Path(proposal_path) if proposal_path else None
    embeddings_path_obj = Path(embeddings_path) if embeddings_path else None
    output_path_obj = Path(output_path) if output_path else None
    
    results = cluster_by_for_division(
        proposal_path=proposal_path_obj,
        embeddings_path=embeddings_path_obj,
        clusters_output_path=output_path_obj,
        batch_size=batch_size
    )
    
    # Print summary
    total_proposals = sum(result['total_proposals'] for result in results.values())
    total_clusters = sum(result['n_clusters'] for result in results.values())
    avg_cluster_size = total_proposals / total_clusters if total_clusters > 0 else 0
    
    print(f"\nSummary:")
    print(f"  FOR divisions: {len(results)}")
    print(f"  Total proposals: {total_proposals}")
    print(f"  Total clusters: {total_clusters}")
    print(f"  Target batch size: {batch_size}")
    print(f"  Average cluster size: {avg_cluster_size:.1f}")


if __name__ == "__main__":
    app()
