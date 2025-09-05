from random import choices
from inspect_ai import Task, task
import shutil
import utils
import jsonlines
import config
from pathlib import Path
from models import Category, CategoryList
from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks
from inspect_ai.dataset import json_dataset, FieldSpec, Sample, MemoryDataset
import json
from inspect_ai.solver import system_message, generate, multiple_choice
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.util import json_schema
from itertools import batched
import json
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from inspect_ai._util.answer import answer_character, answer_index
from inspect_ai.solver._multiple_choice import parse_answers

class Category(BaseModel):
    model_config = {"extra": "forbid"}
    name: str = Field(description="Name of the category")
    description: str = Field(description="A few sentences describing what this category is about, including its scope, focus areas, and the types of research or technologies it encompasses")


class CategoryList(BaseModel):
    model_config = {"extra": "forbid"}
    categories: List[Category] = Field(description="List of research categories")


SYSTEM_PROMPT = f"""
You are an expert Technology Analyst and Innovation Forecaster, specializing in abstracting and condensing research concepts into higher-level thematic categories.

Your task is to analyze lower-level concepts and abstract them into broader, more comprehensive research and technology categories. Each category should represent a unified theme that encompasses multiple related lower-level concepts.

**IMPORTANT: You MUST complete this task immediately and provide the full categorization. Do NOT ask for clarification, approval, or propose alternative approaches.**

**Core Objective:**
Your primary goal is to condense and abstract the provided lower-level concepts into meaningful, higher-level categories. Look for common themes, shared methodologies, overlapping applications, and conceptual relationships that allow you to group specific concepts under broader umbrellas.

---

### **CRITICAL RULES FOR ABSTRACTION:**

**1. Abstraction & Condensation:**
*   **Thematic Grouping:** Identify overarching themes that can encompass multiple lower-level concepts. Look for shared principles, methodologies, or application domains.
*   **Hierarchical Thinking:** Create categories that sit at a higher level of abstraction than the input concepts, representing broader research domains or technological areas.
*   **Conceptual Synthesis:** Combine related concepts into unified categories that capture their common essence while maintaining meaningful distinctions.

**2. Category Formation:**
*   **Emergent Patterns:** Look for patterns across the lower-level concepts that suggest natural higher-level groupings.
*   **Interdisciplinary Recognition:** Identify where concepts from different fields converge into broader interdisciplinary domains.
*   **Innovation Focus:** Prioritize categories that represent emerging or transformative areas of research and technology.

**3. Output Requirements:**
*   **Category Naming:** Each category requires a concise, descriptive `name` (ideally 2-4 words) that captures the essence of the abstracted theme.
*   **Comprehensive Descriptions:** The `description` for each category must explain how the lower-level concepts relate to this higher-level theme and what unifies them under this category.
"""





# class keyword_md_template(template):
    # def __call__(self, record):
        # return f"**Keyword**: {record['name']}\n**Description**: {record['description']}"
    

KEYWORD_MD_TEMPLATE = lambda record: f"**Keyword**: {record['name']}\n**Description**: {record['description']}"
CATEGORY_MD_TEMPLATE = lambda record: f"**Category**: {record['name']}\n**Description**: {record['description']}"

from typing import Callable

def load_dataset(input_path: Path, template: Callable, batch_size: int):
    inputs = utils.load_jsonl_file(input_path)
    samples = []
    for idx, batch in enumerate(batched(inputs, batch_size)):
        sample = Sample(id=f"{idx}", input="\n".join([template(item) for item in batch]))
        samples.append(sample)
    return MemoryDataset(samples)


def categorise(input_path: Path, output_path: Path, system_prompt: str, template: Callable, batch_size: int):


    @hooks(name="CategoriseOutputHook", description="Hook to save categorisation results as JSON files")
    class CategoriseOutputHook(Hooks):
        def __init__(self):
            self.output_write = jsonlines.open(output_path, 'w')

        async def on_sample_end(self, data: SampleEnd) -> None:
            result_json = json.loads(data.sample.output.completion)
            self.output_write.write_all(result_json['categories'])
        async def on_task_end(self, data: TaskEnd) -> None:
            self.output_write.close()


    dataset = load_dataset(input_path=input_path, template=template, batch_size=batch_size)

    return Task(
        dataset=dataset,
        solver=[
            system_message(system_prompt),
            generate(),
        ],
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="Categories",
                json_schema=json_schema(CategoryList),
                strict=True
            ),
        ),
        hooks=["CategoriseOutputHook"]
    )


def get_max_level():
    levels = []
    for file in (config.Categories.category_dir).glob("*.jsonl"):
        if file.stem.isnumeric():
            levels.append(int(file.stem))
    return max(levels)

@task
def abstract():
    if not (config.Categories.category_dir / "0.jsonl").exists():
        shutil.copy(config.Keywords.keywords_path, config.Categories.category_dir / "0.jsonl")
    max_level = get_max_level()
    input_file = config.Categories.category_dir / f"{max_level}.jsonl"
    output_file = config.Categories.category_dir / f"{max_level + 1}.jsonl"
    return categorise(input_path=input_file, output_path=output_file, system_prompt=SYSTEM_PROMPT, template=config.Template(), batch_size=config.Categories.batch_size)

def load_classification_dataset(cats_texts, keywords) -> MemoryDataset:
    keyword_md_template = config.Template(output_format="md", keys=["name", "description"], type="keyword")
    samples = []
    header = f"Classify the following keywords into one of the categories provided. Respond ONLY with the letter corresponding to your chosen category. Do NOT include any additional text, explanations, or formatting.\n\n"
    for keyword in keywords:
        keyword_text = keyword_md_template(keyword)
        samples.append(Sample(input=f"{header}{keyword_text}", choices=cats_texts, metadata={"keyword": keyword['name']}))
    dataset = MemoryDataset(samples)
    return dataset

MCQ_PROMPT = r"""
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes and without dollar sign) where LETTER is one of {letters}.

{question}

{choices}
""".strip()
@task
def classify():
    max_level = get_max_level()
    category_md_template = config.Template(output_format="md", keys=["name", "description"], type="category")
    cats = utils.load_jsonl_file(config.Categories.category_dir / f"{max_level}.jsonl", as_dataframe=True)
    cats_texts = cats.apply(category_md_template, axis=1).tolist()
    keywords = utils.load_jsonl_file(config.Keywords.keywords_path, as_dataframe=False)
    dataset = load_classification_dataset(cats_texts, keywords)

    @hooks(name="ClassificationOutputHook", description="Hook to save classification results as JSON files")
    class ClassificationOutputHook(Hooks):
        
        def __init__(self):
            self.writer = jsonlines.open(config.Categories.category_dir / f"{max_level}.mapping.jsonl", 'w')
            
        async def on_sample_end(self, data: SampleEnd) -> None:
            answer_letter = parse_answers(data.sample, multiple_correct=False)
            choice_index = answer_index(list(answer_letter)[0])
            self.writer.write({data.sample.metadata['keyword']: cats.iloc[choice_index]['name']})
            
            # choice_text = data.sample.choices[choice_index]
            # cat = config.reverse_item(choice_text)
            # keyword = config.reverse_item(data.sample.input)
            # self.writer.write({keyword["name"]: cat["name"]})
            
        async def on_task_end(self, data: TaskEnd) -> None:
            self.writer.close()

    
    return Task(
        dataset=dataset,
        solver=[
            multiple_choice(template=MCQ_PROMPT),
        ],
        hooks=["ClassificationOutputHook"]
    )

