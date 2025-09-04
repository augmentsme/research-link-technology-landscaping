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
from pydantic import BaseModel, Field
# from config import CATEGORY_PROPOSAL_PATH, CATEGORY_PATH, PROMPTS_DIR, BATCH_SIZE, RESULTS_DIR

from models import Category, CategoryList
import utils
from scorer import keywords_coverage
import re

import utils

@hooks(name="MergeCategoriesHook", description="Hook to combine FOR group merge results into a single categories.json")
class MergeCategoriesHook(Hooks):
    def __init__(self):
        self.results = []
        
    async def on_sample_end(self, data: SampleEnd):
        _, result = utils.parse_html(data.sample.output)
        self.results.extend(result['categories'])

    async def on_task_end(self, data: TaskEnd) -> None:
        # Write all collected categories to the final file
        utils.save_jsonl_file(self.results, config.Categories.categories_path)

def load_category_files_dataset() -> MemoryDataset:

    proposals = config.Categories.load_proposal()
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
def merge() -> Task:
    dataset = load_category_files_dataset()

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
