from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

import config
import utils

from . import llm_client

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = getattr(config, "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")


class EmbeddingGenerator:
    def __init__(self, template_func: Callable[[dict], str], model_name: Optional[str] = None, batch_size: int = 64):
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
                llm_client.create_embeddings(
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
