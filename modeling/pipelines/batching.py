from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Iterable, Any
import json

import jsonlines


@dataclass(frozen=True)
class JsonlBatch:
    index: int
    batch_id: str
    path: Path


def iter_jsonl_batches(directory: Path | str) -> Iterator[JsonlBatch]:
    folder = Path(directory)
    files = sorted(folder.glob("*.jsonl"))
    for index, path in enumerate(files):
        yield JsonlBatch(index=index, batch_id=path.stem, path=path)


def stream_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            yield json.loads(text)


@dataclass
class CategoryOutputs:
    categories: jsonlines.Writer
    unknown: jsonlines.Writer
    missing: jsonlines.Writer

    def close(self) -> None:
        self.categories.close()
        self.unknown.close()
        self.missing.close()


def prepare_category_outputs(output_dir: Path | str) -> CategoryOutputs:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return CategoryOutputs(
        categories=jsonlines.open(output_path / "output.jsonl", "w"),
        unknown=jsonlines.open(output_path / "unknown.jsonl", "w"),
        missing=jsonlines.open(output_path / "missing.jsonl", "w"),
    )


def write_categorisation_result(
    records: Iterable[dict[str, Any]],
    result: Any,
    outputs: CategoryOutputs,
) -> None:
    keyword_names = {record.get("name") for record in records if record.get("name")}
    assigned_keywords: set[str] = set()

    for category in result.categories:
        payload = category.model_dump()
        keywords = payload.get("keywords", [])
        assigned_keywords.update(keywords)
        if payload.get("name") == "Unknown":
            for keyword in keywords:
                outputs.unknown.write(keyword)
        else:
            outputs.categories.write(payload)

    missing_keywords = sorted(keyword_names - assigned_keywords)
    for keyword in missing_keywords:
        outputs.missing.write(keyword)


@dataclass
class MergeOutputs:
    categories: jsonlines.Writer
    missing: jsonlines.Writer

    def close(self) -> None:
        self.categories.close()
        self.missing.close()


def prepare_merge_outputs(output_dir: Path | str) -> MergeOutputs:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return MergeOutputs(
        categories=jsonlines.open(output_path / "output.jsonl", "w"),
        missing=jsonlines.open(output_path / "missing.jsonl", "w"),
    )


def write_merge_result(
    records: Iterable[dict[str, Any]],
    result: Any,
    outputs: MergeOutputs,
) -> None:
    category_lookup: Dict[str, list[str]] = {
        record.get("name"): record.get("keywords", [])
        for record in records
        if record.get("name")
    }
    referenced: set[str] = set()

    for merged_category in result.categories:
        payload = merged_category.model_dump()
        source_names = payload.get("source_categories", [])
        referenced.update(source_names)
        merged_keywords: set[str] = set()
        for source_name in source_names:
            merged_keywords.update(category_lookup.get(source_name, []))
        payload["keywords"] = sorted(merged_keywords)
        outputs.categories.write(payload)

    missing = sorted(set(category_lookup.keys()) - referenced)
    for category_name in missing:
        outputs.missing.write(category_name)
