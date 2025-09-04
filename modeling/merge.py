from inspect_ai import Task, task
from itertools import batched
import config

from inspect_ai.hooks import Hooks, TaskEnd, hooks, SampleEnd
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import system_message, generate, TaskState
from inspect_ai.model import GenerateConfig, ResponseSchema, get_model
from inspect_ai.util import json_schema
from inspect_ai.scorer import scorer, Target, Score, mean, Scorer
import json
import jsonlines
from pydantic import BaseModel, Field

from models import Category, CategoryList
import utils
from scorer import keywords_coverage
import re

import utils

@hooks(name="MergeCategoriesHook", description="Hook to combine FOR group merge results into a single categories.json")
class MergeCategoriesHook(Hooks):
    def __init__(self):
        self.results = []
        # Create merge-specific paths for unknown and missing keywords
        merge_unknown_path = config.Categories.category_dir / "merge_unknown_keywords.jsonl"
        merge_missing_path = config.Categories.category_dir / "merge_missing_keywords.jsonl"
        
        self.unknown_write = jsonlines.open(merge_unknown_path, 'w')
        self.missing_write = jsonlines.open(merge_missing_path, 'w')
        
    async def on_sample_end(self, data: SampleEnd):
        _, result = utils.parse_html(data.sample.output)
        self.results.extend(result['categories'])
        
        # Save missing keywords from the keywords_coverage scorer
        if 'keywords_coverage' in data.sample.scores:
            missing_keywords = data.sample.scores['keywords_coverage'].metadata.get('missing', [])
            if missing_keywords:
                self.missing_write.write_all(missing_keywords)
        
        # Save unknown keywords (keywords in "Unknown" category)
        for category in result['categories']:
            if category['name'] == "Unknown":
                self.unknown_write.write_all(category['keywords'])

    async def on_task_end(self, data: TaskEnd) -> None:
        utils.save_jsonl_file(self.results, config.Categories.categories_path)
        self.unknown_write.close()
        self.missing_write.close()

def load_category_files_dataset(proposal_path) -> MemoryDataset:
    proposals = utils.load_jsonl_file(proposal_path, as_dataframe=True)
    proposals = proposals.sort_values(by='for_code')
    samples = []
    for for_code, categories in proposals.groupby('for_code'):
        for batch_index, batch_categories in enumerate(batched(categories.to_dict(orient="records"), n=config.Categories.batch_size)):
            cat_text = "\n".join([config.Categories.template(i) for i in batch_categories])
            sample = Sample(
                id=f"merge_for_{for_code}_batch_{batch_index}",
                input=cat_text,
                metadata={
                    "keywords": [kw for category in batch_categories for kw in category['keywords']]
                }
            )
            samples.append(sample)

    return MemoryDataset(samples)

@task
def merge(proposal_path=config.Categories.category_proposal_path) -> Task:
    
    dataset = load_category_files_dataset(proposal_path)

    return Task(
        dataset=dataset,
        solver=[
            system_message(str(config.PROMPTS_DIR / "merge.txt")),
            generate()
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="MergedCategories",
                json_schema=json_schema(CategoryList),
                strict=True
            )
        ),
        scorer=[keywords_coverage()],
        hooks=["MergeCategoriesHook"]
    )


