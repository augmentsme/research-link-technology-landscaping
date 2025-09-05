from xml.parsers.expat import model
import chromadb
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset, FieldSpec, Sample, MemoryDataset
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.solver import system_message, generate, user_message
from inspect_ai.util import json_schema
import jsonlines


import utils
import config

import json
from typing import List
from enum import Enum
from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks
from inspect_ai.solver import TaskState
from pydantic import BaseModel, Field
from inspect_ai.scorer import model_graded_qa


import re

from nltk.stem import PorterStemmer

    



def normalize_keyword(keyword: str) -> str:

    normalized = keyword.lower().strip()

    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = normalized.replace('-', ' ').replace('_', ' ')
    
    words = normalized.split()
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_normalized = ' '.join(stemmed_words)
    return stemmed_normalized



def normalise_keywords(keywords: List[dict]) -> dict:
    normalized_keywords = {}
    for keyword in keywords:
        canonical_keyword = normalize_keyword(keyword['name'])
        if normalized_keywords.get(canonical_keyword) is None:
            normalized_keywords[canonical_keyword] = keyword
            normalized_keywords[canonical_keyword]['grants'] = [keyword['grant']]
        else:
            normalized_keywords[canonical_keyword]['grants'].extend(keyword['grant'])
            normalized_keywords[canonical_keyword]['description'] += "; " + keyword['description']
    return normalized_keywords

@hooks(name="KeywordsExtractionHook", description="Hook to save keywords extraction results as JSON files")
class KeywordsExtractionHook(Hooks):
    """Hook to save keywords extraction results as JSON files."""

    def __init__(self):
        # self.keywords = []
        self.writer = jsonlines.open(config.Keywords.extracted_keywords_path, 'w')

    async def on_sample_end(self, data: SampleEnd) -> None:
        """Save keywords extraction results with grant references."""

        output_text = data.sample.output.completion
        result_json = json.loads(output_text)

        grant_id = data.sample.id
        for keyword in result_json["keywords"]:
            keyword["grant"] = grant_id
            self.writer.write(keyword)

    async def on_task_end(self, data: TaskEnd) -> None:
        self.writer.close()
        raw_keywords_output = utils.load_jsonl_file(config.Keywords.extracted_keywords_path)
        normalized_keywords = normalise_keywords(raw_keywords_output)
        utils.save_jsonl_file(list(normalized_keywords.values()), config.Keywords.keywords_path)
        


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


SYSTEM_PROMPT = f"""

You are an expert research analyst with deep knowledge across multiple academic disciplines and a keen eye for emerging research trends.

Your task is to extract meaningful keywords from research grant information that would be useful for:
- Identifying emerging research domains and interdisciplinary areas
- Discovering novel methodologies and cutting-edge approaches
- Tracking innovative technologies and emerging tools
- Finding related research projects working on similar frontiers
- Understanding emerging research trends and future directions

Focus on extracting keywords that highlight what's new, innovative, and emerging in the research landscape. Prioritize:
- Technical terms that represent novel concepts or emerging fields
- Methodologies that are cutting-edge or represent new approaches
- Technologies that are innovative or represent emerging tools
- Applications that address new challenges or emerging needs
- Scientific terminology that indicates research at the frontiers of knowledge

**CRITICAL REQUIREMENT - Avoid Overly Broad Keywords:**
- **Keywords must be specific and precise, not generic or overly broad**
- **Keywords or variants of keywords must be present in the grant title or summary**
- **Keywords must be short (around 1-4 words), focused, and descriptive of a specific concept, method, or technology**
- **Instead of "engineering," use "bio-integrated nano-photonics" or "quantum-enhanced engineering"**
- **Instead of "chemistry," use "supramolecular photochemistry" or "catalytic asymmetric synthesis"**
- **Instead of "data analysis," use "multi-modal time-series analysis" or "causal inference modeling"**
- **Instead of "artificial intelligence," use "graph neural networks" or "federated learning algorithms"**
- **Avoid general terms like "research," "development," "innovation," "technology," "analysis," "method"**
- **Each keyword should clearly indicate a specific domain, technique, or application area**

Provide accurate, specific, and well-categorized keywords that capture the innovative and emerging aspects of the research. Reject overly broad terms that fail to convey the unique focus of the work.


"""

class KeywordType(str, Enum):
    """Enumeration of valid keyword types."""
    GENERAL = "General"
    METHODOLOGY = "Methodology"
    APPLICATION = "Application"
    TECHNOLOGY = "Technology"

class Keyword(BaseModel):
    """Individual keyword with context."""
    model_config = {"extra": "forbid"}
    name: str = Field(description="The actual keyword or phrase")
    type: KeywordType = Field(description="Type of keyword: general, methodology, application, or technology")
    description: str = Field(description="Short description explaining the context and relevance of this keyword within the research")
    # grants: List[str] = Field(description="List of grant IDs where this keyword appears")

class KeywordsList(BaseModel):
    """Pydantic model for structured keywords extraction output."""
    model_config = {"extra": "forbid"}
    keywords: List[Keyword] = Field(description="List of all extracted keywords with their types and descriptions")



#
@task
def extract(filter_finished=True) -> Task:
    finished_grants = config.finished_grants()
    dataset = MemoryDataset(list(map(record_to_sample, utils.load_jsonl_file(config.Grants.grants_path, as_dataframe=True).to_dict(orient="records"))))
    # dataset = utils.load_jsonl_file(config.Keywords.extracted_keywords_path, as_dataframe=True).map(record_to_sample)
    # dataset = json_dataset(
    #         str(config.Grants.grants_path),
    #         record_to_sample
    #     )

    if filter_finished:
        dataset = dataset.filter(lambda sample: sample.id not in finished_grants)
    return Task(
        dataset=dataset,
        solver=[
            system_message(SYSTEM_PROMPT),
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
