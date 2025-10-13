from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class ClusterBatch:
    index: int
    batch_id: str
    records: List[dict[str, Any]]

    @property
    def size(self) -> int:
        return len(self.records)


@dataclass(frozen=True)
class ClusterFile:
    path: Path
    batches: List[ClusterBatch]
    total_items: int
    metadata: Dict[str, Any] | None = None


def load_cluster_file(path: Path | str) -> ClusterFile:
    clusters_path = Path(path)
    with clusters_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    clusters = payload.get("clusters")
    if not isinstance(clusters, dict):
        raise ValueError(f"Cluster file {clusters_path} must contain a 'clusters' dictionary")

    batches: List[ClusterBatch] = []
    for index, batch_id in enumerate(sorted(clusters.keys())):
        items = clusters[batch_id] or []
        normalized = [
            {k: v for k, v in item.items() if k != "embedding_index"}
            for item in items
            if isinstance(item, dict)
        ]
        batches.append(ClusterBatch(index=index, batch_id=batch_id, records=normalized))

    total_items = payload.get("total_items", sum(batch.size for batch in batches))
    metadata = payload.get("metadata")

    return ClusterFile(path=clusters_path, batches=batches, total_items=total_items, metadata=metadata)
