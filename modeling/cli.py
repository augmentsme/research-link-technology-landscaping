from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

import config
import utils
from pipelines import categorise as categorise_pipeline
from pipelines import extract as extract_pipeline
from pipelines import merge as merge_pipeline
import numpy as np
from pipelines.semantic import Pipeline, DEFAULT_EMBEDDING_MODEL

try:  # prefer local vLLM when available
    from vllm import LLM
except ImportError:  # fall back to sentence transformers at runtime
    LLM = None  # type: ignore


app = typer.Typer(help="Unified interface for keyword extraction and categorisation workflows")
semantic_app = typer.Typer(help="Semantic clustering utilities")
app.add_typer(semantic_app, name="semantic")


SENTENCE_TRANSFORMER_FALLBACK = "sentence-transformers/all-MiniLM-L6-v2"


@app.command()
def extract(
    skip_finished: bool = typer.Option(True, help="Skip grants that already have extracted keywords"),
    concurrency: int = typer.Option(10, help="Maximum concurrent requests"),
) -> None:
    processed = extract_pipeline.extract(filter_finished=skip_finished, concurrency=concurrency)
    count = len(processed)
    typer.echo(f"Processed {count} grant{'s' if count != 1 else ''}")


@app.command()
def categorise(
    input_dir: Path = typer.Argument(..., help="Directory containing keyword batch jsonl files"),
    output_dir: Path = typer.Argument(..., help="Directory to write categorisation outputs"),
    concurrency: int = typer.Option(config.CONCURRENCY, help="Maximum concurrent requests"),
) -> None:
    processed = categorise_pipeline.categorise(input_dir, output_dir, concurrency=concurrency)
    count = len(processed)
    typer.echo(f"Processed {count} batch{'es' if count != 1 else ''}")


@app.command()
def merge(
    input_dir: Path = typer.Argument(..., help="Directory with category batch jsonl files"),
    output_dir: Path = typer.Argument(..., help="Directory to write merged outputs"),
    concurrency: int = typer.Option(config.CONCURRENCY, help="Maximum concurrent requests"),
) -> None:
    processed = merge_pipeline.merge(input_dir, output_dir, concurrency=concurrency)
    count = len(processed)
    typer.echo(f"Processed {count} batch{'es' if count != 1 else ''}")


@app.command()
def embed(
    dataset: str = typer.Argument(..., help="Dataset to embed: 'keywords' or 'categories'"),
    input_path: Optional[Path] = typer.Option(None, help="Path to source jsonl; defaults to standard location"),
    output_path: Optional[Path] = typer.Option(None, help="Path to write embeddings npy file"),
    model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="Local Hugging Face model id"),
    batch_size: int = typer.Option(64, help="Number of texts per embedding batch"),
    force: bool = typer.Option(False, help="Regenerate embeddings even if cache exists"),
) -> None:
    choice = dataset.lower()
    if choice == "keywords":
        data_path = input_path or config.Keywords.keywords_path
        data = utils.load_jsonl_file(data_path, as_dataframe=True)
        texts = data.apply(config.Keywords.template, axis=1).tolist()
    elif choice == "categories":
        data_path = input_path or config.Categories.CATEGORIY_PATH
        data = utils.load_jsonl_file(data_path, as_dataframe=True)
        texts = data.apply(config.Categories.template, axis=1).tolist()
    else:
        raise typer.BadParameter("Dataset must be 'keywords' or 'categories'")

    if not texts:
        typer.echo("No records to embed")
        return

    embeddings_path = output_path or data_path.with_suffix(".embeddings.npy")
    if embeddings_path.exists() and not force:
        typer.echo(f"Embeddings already exist at {embeddings_path}; use --force to overwrite")
        return

    if LLM is None:
        from sentence_transformers import SentenceTransformer

        target_model = model
        if target_model == DEFAULT_EMBEDDING_MODEL:
            target_model = SENTENCE_TRANSFORMER_FALLBACK

        encoder = SentenceTransformer(target_model)
        array = encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        ).astype(np.float32)
    else:
        llm = LLM(model=model, task="embed")
        embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            outputs = llm.embed(texts[start : start + batch_size])
            embeddings.extend(np.asarray(result.outputs.embedding, dtype=np.float32) for result in outputs)

        array = np.vstack(embeddings)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, array)
    typer.echo(f"Saved {array.shape[0]} embeddings with dimension {array.shape[1]} to {embeddings_path}")


@semantic_app.command("run")
def run(
    data_path: Path = typer.Argument(..., help="Input dataset jsonl"),
    embeddings_path: Optional[Path] = typer.Option(None, help="Embeddings file path"),
    clusters_path: Optional[Path] = typer.Option(None, help="Clusters json path"),
    batch_dir: Optional[Path] = typer.Option(None, help="Directory for generated batches"),
    batch_size: int = typer.Option(50, help="Target batch size"),
    force_embeddings: bool = typer.Option(False, help="Regenerate embeddings even if cached"),
) -> None:
    pipeline = Pipeline(
        data_path=data_path,
        embeddings_path=embeddings_path,
        clusters_path=clusters_path,
        batch_dir=batch_dir,
    )
    result = pipeline.run_full_pipeline(batch_size=batch_size, force_embeddings=force_embeddings)
    typer.echo(
        f"Embeddings {result['embeddings_shape']}, clusters {result['semantic_clusters']}, batches {result['generated_batches']}"
    )


@semantic_app.command()
def embeddings(
    data_path: Path = typer.Argument(..., help="Input dataset jsonl"),
    embeddings_path: Optional[Path] = typer.Option(None, help="Embeddings file path"),
    force: bool = typer.Option(False, help="Regenerate embeddings even if cached"),
) -> None:
    pipeline = Pipeline(data_path=data_path, embeddings_path=embeddings_path)
    vectors = pipeline.generate_embeddings(force_regenerate=force)
    typer.echo(f"Created embeddings with shape {vectors.shape}")


@semantic_app.command()
def cluster(
    data_path: Path = typer.Argument(..., help="Input dataset jsonl"),
    embeddings_path: Optional[Path] = typer.Option(None, help="Embeddings file path"),
    clusters_path: Optional[Path] = typer.Option(None, help="Clusters json path"),
    batch_size: int = typer.Option(50, help="Target batch size"),
) -> None:
    pipeline = Pipeline(
        data_path=data_path,
        embeddings_path=embeddings_path,
        clusters_path=clusters_path,
    )
    result = pipeline.cluster_data(batch_size=batch_size)
    typer.echo(f"Clustered {result.total_items} items into {result.n_clusters} groups")


@semantic_app.command()
def batches(
    data_path: Path = typer.Argument(..., help="Input dataset jsonl"),
    embeddings_path: Optional[Path] = typer.Option(None, help="Embeddings file path"),
    clusters_path: Optional[Path] = typer.Option(None, help="Clusters json path"),
    batch_dir: Optional[Path] = typer.Option(None, help="Directory for generated batches"),
    batch_size: int = typer.Option(50, help="Target batch size"),
) -> None:
    pipeline = Pipeline(
        data_path=data_path,
        embeddings_path=embeddings_path,
        clusters_path=clusters_path,
        batch_dir=batch_dir,
    )
    files = pipeline.generate_batches(batch_size=batch_size)
    typer.echo(f"Saved {len(files)} batch files")


if __name__ == "__main__":
    app()
