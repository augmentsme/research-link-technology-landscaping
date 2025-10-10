from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.cluster import MiniBatchKMeans

import config
import utils

from . import llm_client

logger = logging.getLogger(__name__)


DEFAULT_EMBEDDING_MODEL = getattr(config, "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")


@dataclass
class ClusteringResult:
    total_items: int
    n_clusters: int
    target_batch_size: int
    clusters: Dict[str, List[Dict]]
    metadata: Optional[Dict[str, Any]] = None


class EmbeddingGenerator:
    def __init__(self, template_func: callable, model_name: Optional[str] = None, batch_size: int = 64):
        self.template_func = template_func
        self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
        self.batch_size = max(batch_size, 1)
        self.client = llm_client.sync_client()

    def generate(self, data_path: Path, output_path: Path, force_regenerate: bool = False) -> np.ndarray:
        if self._should_regenerate(data_path, output_path, force_regenerate):
            logger.info("Generating embeddings for %s", data_path)
            data = utils.load_jsonl_file(data_path, as_dataframe=True)
            texts = self._format_text(data)
            vectors = self._embed(texts)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, vectors)
            logger.info("Saved embeddings to %s", output_path)
            return vectors

        logger.info("Loading cached embeddings from %s", output_path)
        return np.load(output_path)

    def _embed(self, texts: List[str]) -> np.ndarray:
        rows: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start:start + self.batch_size]
            if not chunk:
                continue
            rows.extend(
                llm_client.retry_embeddings(
                    chunk,
                    client=self.client,
                    model=self.model_name,
                )
            )
        return np.asarray(rows, dtype=np.float32)

    def _format_text(self, data) -> List[str]:
        texts = []
        for _, record in data.iterrows():
            text = self.template_func(record.to_dict())
            texts.append(text)
        return texts

    def _should_regenerate(self, data_path: Path, output_path: Path, force: bool) -> bool:
        if force or not output_path.exists() or not data_path.exists():
            return True
        return output_path.stat().st_mtime <= data_path.stat().st_mtime


class ClusterManager:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def cluster(self, embeddings: np.ndarray, data: Any, batch_size: int) -> ClusteringResult:
        n_clusters = self._calculate_clusters(len(data), batch_size)
        logger.info("Clustering %s items into %s clusters (target batch_size=%s)", len(data), n_clusters, batch_size)

        if len(data) <= batch_size:
            cluster_labels = [0] * len(data)
        else:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init="auto",
                batch_size=min(1024, len(data)),
            )
            cluster_labels = kmeans.fit_predict(embeddings)

        clusters = self._organize_clusters(data, cluster_labels)

        return ClusteringResult(
            total_items=len(data),
            n_clusters=len(clusters),
            target_batch_size=batch_size,
            clusters=clusters,
        )

    def _calculate_clusters(self, data_size: int, batch_size: int) -> int:
        n_clusters = max(1, data_size // batch_size)
        if data_size % batch_size > 0:
            n_clusters += 1
        return n_clusters

    def _organize_clusters(self, data: Any, labels: np.ndarray) -> Dict[str, List[Dict]]:
        clusters: Dict[str, List[Dict]] = {}
        if hasattr(data, "to_dict"):
            data_records = data.to_dict(orient="records")
        else:
            data_records = data

        for index, (item, cluster_id) in enumerate(zip(data_records, labels)):
            cluster_key = f"cluster_{cluster_id}"
            clusters.setdefault(cluster_key, [])
            item_with_meta = item.copy()
            item_with_meta["embedding_index"] = index
            clusters[cluster_key].append(item_with_meta)

        cluster_sizes = [len(cluster) for cluster in clusters.values()]
        if cluster_sizes:
            logger.info("Created %s clusters with sizes: %s", len(clusters), cluster_sizes)
            logger.info("Average cluster size: %.1f", np.mean(cluster_sizes))
            logger.info("Min/Max cluster sizes: %s/%s", min(cluster_sizes), max(cluster_sizes))
        return clusters


class BatchGenerator:
    @staticmethod
    def generate_batches(clustering_result: ClusteringResult, output_dir: Path, batch_size: int) -> List[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_files: List[Path] = []
        current_batch: List[Dict] = []
        batch_num = 0

        for cluster_items in clustering_result.clusters.values():
            for item in cluster_items:
                clean_item = {k: v for k, v in item.items() if k != "embedding_index"}
                current_batch.append(clean_item)

                if len(current_batch) >= batch_size:
                    batch_path = output_dir / f"{batch_num:03d}.jsonl"
                    utils.save_jsonl_file(current_batch, batch_path)
                    batch_files.append(batch_path)
                    logger.info("Saved batch %s with %s items to %s", batch_num, len(current_batch), batch_path)

                    current_batch = []
                    batch_num += 1

        if current_batch:
            batch_path = output_dir / f"{batch_num}.jsonl"
            utils.save_jsonl_file(current_batch, batch_path)
            batch_files.append(batch_path)
            logger.info("Saved final batch %s with %s items to %s", batch_num, len(current_batch), batch_path)

        logger.info("Generated %s batches", len(batch_files))
        return batch_files


class DatasetBuilder:
    @staticmethod
    def load_batches(batch_dir: Path) -> List[List[Dict]]:
        if not batch_dir.exists():
            raise FileNotFoundError(f"Batch directory does not exist: {batch_dir}")

        batch_files = sorted(batch_dir.glob("*.jsonl"))
        if not batch_files:
            raise FileNotFoundError(f"No batch files found in {batch_dir}")

        batches: List[List[Dict]] = []
        for batch_file in batch_files:
            batch_data = utils.load_jsonl_file(batch_file, as_dataframe=False)
            batches.append(batch_data)
            logger.info("Loaded batch from %s with %s items", batch_file.name, len(batch_data))

        logger.info("Loaded %s batches with total %s items", len(batches), sum(len(b) for b in batches))
        return batches

    @staticmethod
    def create_categorization_dataset(batch_dir: Path):
        from inspect_ai.dataset import MemoryDataset, Sample

        batches = DatasetBuilder.load_batches(batch_dir)

        samples = []
        for idx, keywords in enumerate(batches):
            entries = [config.Keywords.template(kw) for kw in keywords]
            sample = Sample(
                id=f"semantic_batch_{idx}",
                input="\n".join(entries),
                metadata={"keywords": [kw["name"] for kw in keywords]},
            )
            samples.append(sample)

        return MemoryDataset(samples)


class Pipeline:
    def __init__(
        self,
        data_path: Path,
        embeddings_path: Optional[Path] = None,
        clusters_path: Optional[Path] = None,
        batch_dir: Optional[Path] = None,
        template_func: Optional[callable] = None,
        model_name: Optional[str] = None,
    ):
        self.data_path = data_path
        self.embeddings_path = embeddings_path or data_path.with_suffix(".embeddings.npy")
        self.clusters_path = clusters_path or data_path.with_name(f"{data_path.stem}_clusters.json")
        self.batch_dir = batch_dir or (data_path.parent / "batches")

        if template_func is None:
            template_func = lambda x: f"{x.get('name', '')}: {x.get('description', '')}"

        self.embedding_generator = EmbeddingGenerator(template_func, model_name)
        self.cluster_manager = ClusterManager()
        self.batch_generator = BatchGenerator()

    def _should_regenerate_clusters(self) -> bool:
        if not self.clusters_path.exists() or not self.embeddings_path.exists():
            return True
        return self.clusters_path.stat().st_mtime <= self.embeddings_path.stat().st_mtime

    def _should_regenerate_batches(self) -> bool:
        if not self.clusters_path.exists():
            return True

        if self.batch_dir.exists():
            batch_files = list(self.batch_dir.glob("*.jsonl"))
            if batch_files:
                clusters_mtime = self.clusters_path.stat().st_mtime
                for batch_file in batch_files:
                    if batch_file.stat().st_mtime <= clusters_mtime:
                        return True
                return False
        return True

    def generate_embeddings(self, force_regenerate: bool = False) -> np.ndarray:
        return self.embedding_generator.generate(
            self.data_path,
            self.embeddings_path,
            force_regenerate,
        )

    def cluster_data(self, batch_size: int = 50) -> ClusteringResult:
        data = utils.load_jsonl_file(self.data_path, as_dataframe=True)
        embeddings = np.load(self.embeddings_path)

        if len(data) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch between data items ({len(data)}) and embeddings ({embeddings.shape[0]})"
            )

        result = self.cluster_manager.cluster(embeddings, data, batch_size)

        self.clusters_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.clusters_path, "w") as handle:
            json.dump(
                {
                    "total_items": result.total_items,
                    "n_clusters": result.n_clusters,
                    "target_batch_size": result.target_batch_size,
                    "clusters": result.clusters,
                },
                handle,
                indent=2,
                default=str,
            )

        logger.info("Clustering results saved to: %s", self.clusters_path)
        return result

    def generate_batches(self, batch_size: int = 50) -> List[Path]:
        if self._should_regenerate_clusters():
            logger.info("Embeddings are newer than clusters. Regenerating clusters first...")
            self.cluster_data(batch_size)
        elif self._should_regenerate_batches():
            logger.info("Clusters are newer than existing batches. Regenerating batches...")

        with open(self.clusters_path, "r") as handle:
            cluster_data = json.load(handle)

        clustering_result = ClusteringResult(
            total_items=cluster_data["total_items"],
            n_clusters=cluster_data["n_clusters"],
            target_batch_size=cluster_data["target_batch_size"],
            clusters=cluster_data["clusters"],
        )

        return self.batch_generator.generate_batches(
            clustering_result,
            self.batch_dir,
            batch_size,
        )

    def run_full_pipeline(self, batch_size: int = 50, force_embeddings: bool = False) -> Dict[str, Any]:
        logger.info("=== Step 1: Generating embeddings ===")
        embeddings = self.generate_embeddings(force_embeddings)

        logger.info("\n=== Step 2: Clustering semantically ===")
        clustering_result = self.cluster_data(batch_size)

        logger.info("\n=== Step 3: Generating balanced batches ===")
        batch_files = self.generate_batches(batch_size)

        logger.info("\n=== Pipeline complete ===")
        return {
            "embeddings_shape": embeddings.shape,
            "total_items": clustering_result.total_items,
            "semantic_clusters": clustering_result.n_clusters,
            "generated_batches": len(batch_files),
            "target_batch_size": batch_size,
            "batch_files": batch_files,
        }

    def create_categorization_dataset(self):
        return DatasetBuilder.create_categorization_dataset(self.batch_dir)
