import chromadb
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset, FieldSpec, Sample
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.solver import system_message, generate, user_message
from inspect_ai.util import json_schema

from models import KeywordType, Keyword, KeywordsList
import utils
import config

import json
from typing import List
from enum import Enum
from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks
from inspect_ai.solver import TaskState
from pydantic import BaseModel, Field
from inspect_ai.scorer import scorer, Score, Target, metric, Metric, SampleScore


from metric import total
import re

from nltk.stem import PorterStemmer

    



def normalize_keyword_term(term: str) -> str:
    """
    Normalize keyword terms for deduplication using NLTK stemming and lexical variations.
    
    Args:
        term: The original keyword term
        
    Returns:
        Normalized term for comparison
    """
    # Convert to lowercase
    normalized = term.lower().strip()

    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = normalized.replace('-', ' ').replace('_', ' ')
    
    words = normalized.split()
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_normalized = ' '.join(stemmed_words)
    return stemmed_normalized



@hooks(name="KeywordsExtractionHook", description="Hook to save keywords extraction results as JSON files")
class KeywordsExtractionHook(Hooks):
    """Hook to save keywords extraction results as JSON files."""

    def __init__(self):
        self.keywords = []

    async def on_sample_end(self, data: SampleEnd) -> None:
        """Save keywords extraction results with grant references."""

        output_text = data.sample.output.completion
        result_json = json.loads(output_text)

        grant_id = data.sample.id
        
        for keyword in result_json["keywords"]:
            canonical_term = normalize_keyword_term(keyword['term'])
            if canonical_term not in self.keywords.keys():
                keyword["grants"] = [grant_id]
                self.keywords.append(keyword)
            else:
                self.keywords[canonical_term]["grants"].append(grant_id)

    async def on_task_end(self, data: TaskEnd) -> None:
        utils.save_jsonl_file(self.keywords, config.Keywords.extracted_keywords_path)


def record_to_sample(record: dict) -> Sample:
    return Sample(
        id=record["id"],
        input=f"""**Grant ID**: {record["id"]}
**Title**: {record["title"]}
**Summary**: 
{record["grant_summary"]}
                """,
        metadata={
            "title": record["title"],
            "summary": record["grant_summary"],
            "funding_amount": record["funding_amount"],
            "funder": record["funder"],
            "start_year": record["start_year"],
            "end_year": record["end_year"],
        }
    )


@task
def extract(filter_finished=True) -> Task:
    finished_grants = config.finished_grants()
    dataset = json_dataset(
            str(config.Grants.grants_path),
            record_to_sample
        )
    if filter_finished:
        dataset = dataset.filter(lambda sample: sample.id not in finished_grants)
    return Task(
        dataset=dataset,
        solver=[
            system_message(str(config.PROMPTS_DIR / "extract.txt")),
            generate()
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="keywords_extraction",
                json_schema=json_schema(KeywordsList),
                strict=True
            )
        ),
        hooks=["KeywordsExtractionHook"],
    )
