from random import choices
from inspect_ai import Task, task
import shutil
import utils
import jsonlines
import config
from pathlib import Path
# from models import Category, CategoryList
from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks
from inspect_ai.dataset import json_dataset, FieldSpec, Sample, MemoryDataset
from inspect_ai.solver import system_message, generate, multiple_choice
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.util import json_schema
from inspect_ai.scorer import scorer, Score, Target, CORRECT, INCORRECT, accuracy, mean
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


# Define compression ratios for different granularity levels
COMPRESSION_RATIOS = {
    "fine": 0.4,      # 40% of input size (low compression, many categories)
    "medium": 0.2,    # 20% of input size (moderate compression)
    "coarse": 0.1     # 10% of input size (high compression, few categories)
}


@scorer(metrics=[accuracy(), mean()])
def category_count_scorer():
    """
    Scorer to check if the output number of categories matches the expected target.
    
    Returns:
        Score: Accuracy score and deviation metrics
    """
    
    async def score(state, target):
        try:
            # Parse the model output
            completion = state.output.completion
            result_json = json.loads(completion)
            actual_count = len(result_json.get('categories', []))
            
            # Get expected count from sample metadata
            expected_count = state.metadata.get("expected_categories")
            
            if expected_count is None:
                # If no expected count in metadata, we can't score
                return Score(
                    value=INCORRECT,
                    answer=str(actual_count),
                    explanation=f"Generated {actual_count} categories (no expected count in metadata)"
                )
            
            expected_count = int(expected_count)
            
            # Calculate deviation
            deviation = abs(actual_count - expected_count)
            deviation_percentage = (deviation / expected_count) * 100 if expected_count > 0 else 100
            
            # Consider it correct if within 20% of target
            tolerance = max(1, int(expected_count * 0.2))  # At least 1 category tolerance
            is_correct = deviation <= tolerance
            
            explanation = f"Generated {actual_count} categories, expected ~{expected_count} (deviation: {deviation}, {deviation_percentage:.1f}%)"
            
            return Score(
                value=CORRECT if is_correct else INCORRECT,
                answer=str(actual_count),
                explanation=explanation,
                metadata={
                    "actual_count": actual_count,
                    "expected_count": expected_count,
                    "deviation": deviation,
                    "deviation_percentage": deviation_percentage,
                    "tolerance": tolerance
                }
            )
            
        except Exception as e:
            return Score(
                value=INCORRECT,
                answer="error",
                explanation=f"Error parsing categories: {str(e)}"
            )
    
    return score


def create_system_prompt(granularity: str = "medium", batch_size: int = None) -> str:
    """
    Create a system prompt with configurable granularity for categorization.
    
    Args:
        granularity: Level of abstraction for categories
                    - "fine": Low compression ratio (many output categories)
                    - "medium": Moderate compression ratio
                    - "coarse": High compression ratio (few output categories)
        batch_size: Number of input items to be categorized (used to calculate target categories)
    """
    
    if batch_size is None:
        batch_size = config.Categories.batch_size
    
    ratio = COMPRESSION_RATIOS.get(granularity, COMPRESSION_RATIOS["medium"])
    target_categories = max(4, int(batch_size * ratio))  # Minimum of 4 categories
    
    granularity_guidelines = {
        "fine": {
            "scope": "narrow and specific",
            "description": f"Create detailed, specialized categories that capture nuanced distinctions between concepts. Aim for approximately {target_categories} categories to maintain specificity while providing meaningful groupings.",
            "grouping": "Group only closely related concepts together, maintaining clear distinctions between different approaches or domains. Preserve subtle differences between concepts."
        },
        "medium": {
            "scope": "moderate and balanced",
            "description": f"Create well-balanced categories that group related concepts while maintaining meaningful distinctions. Aim for approximately {target_categories} categories to achieve a good balance between detail and abstraction.",
            "grouping": "Group concepts that share common themes, methodologies, or applications, but avoid overly broad generalizations. Find the middle ground between specificity and abstraction."
        },
        "coarse": {
            "scope": "broad and comprehensive", 
            "description": f"Create high-level, comprehensive categories that encompass wide ranges of related concepts. Aim for approximately {target_categories} categories to achieve maximum abstraction and consolidation.",
            "grouping": "Group concepts under broad thematic umbrellas, focusing on fundamental commonalities and overarching principles. Prioritize major conceptual domains over detailed distinctions."
        }
    }
    
    config_info = granularity_guidelines.get(granularity, granularity_guidelines["medium"])
    
    return f"""
You are an expert Technology Analyst and Innovation Forecaster, specializing in abstracting and condensing research concepts into higher-level thematic categories.

Your task is to analyze lower-level concepts and abstract them into broader, more comprehensive research and technology categories. Each category should represent a unified theme that encompasses multiple related lower-level concepts.

**IMPORTANT: You MUST complete this task immediately and provide the full categorization. Do NOT ask for clarification, approval, or propose alternative approaches.**

**GRANULARITY SPECIFICATION:**
- **Input Items:** {batch_size} concepts to categorize
- **Target Categories:** Approximately {target_categories} categories (compression ratio: {ratio:.1%})
- **Scope Level:** {config_info['scope'].title()}
- **Approach:** {config_info['description']}
- **Grouping Strategy:** {config_info['grouping']}

**Core Objective:**
Your primary goal is to condense and abstract the provided {batch_size} lower-level concepts into approximately {target_categories} meaningful, higher-level categories. Look for common themes, shared methodologies, overlapping applications, and conceptual relationships that allow you to group specific concepts under broader umbrellas.

---

### **CRITICAL RULES FOR ABSTRACTION:**

**1. Abstraction & Condensation:**
*   **Thematic Grouping:** Identify overarching themes that can encompass multiple lower-level concepts. Look for shared principles, methodologies, or application domains.
*   **Hierarchical Thinking:** Create categories that sit at a higher level of abstraction than the input concepts, representing broader research domains or technological areas.
*   **Conceptual Synthesis:** Combine related concepts into unified categories that capture their common essence while maintaining meaningful distinctions.
*   **Compression Target:** Aim to reduce {batch_size} input concepts to approximately {target_categories} output categories.

**2. Category Formation:**
*   **Emergent Patterns:** Look for patterns across the lower-level concepts that suggest natural higher-level groupings.
*   **Interdisciplinary Recognition:** Identify where concepts from different fields converge into broader interdisciplinary domains.
*   **Innovation Focus:** Prioritize categories that represent emerging or transformative areas of research and technology.

**3. Output Requirements:**
*   **Category Naming:** Each category requires a concise, descriptive `name` (ideally 2-4 words) that captures the essence of the abstracted theme.
*   **Comprehensive Descriptions:** The `description` for each category must explain how the lower-level concepts relate to this higher-level theme and what unifies them under this category.
"""

# Default system prompt for backward compatibility
SYSTEM_PROMPT = create_system_prompt("medium", config.Categories.batch_size)



from typing import Callable

def load_dataset(input_path: Path, template: Callable, batch_size: int, expected_categories: int):
    inputs = utils.load_jsonl_file(input_path)
    samples = []
    for idx, batch in enumerate(batched(inputs, batch_size)):
        sample = Sample(
            id=f"{idx}", 
            input="\n".join([template(item) for item in batch]),
            metadata={"expected_categories": expected_categories}
        )
        samples.append(sample)
    return MemoryDataset(samples)


def categorise(input_path: Path, output_path: Path, template: Callable, batch_size: int, granularity: str = "fine"):

    # Calculate expected number of categories based on granularity
    ratio = COMPRESSION_RATIOS.get(granularity, COMPRESSION_RATIOS["medium"])
    expected_categories = max(4, int(batch_size * ratio))

    @hooks(name="CategoriseOutputHook", description="Hook to save categorisation results as JSON files")
    class CategoriseOutputHook(Hooks):
        def __init__(self):
            self.output_write = jsonlines.open(output_path, 'w')

        async def on_sample_end(self, data: SampleEnd) -> None:
            result_json = json.loads(data.sample.output.completion)
            self.output_write.write_all(result_json['categories'])
            
        async def on_task_end(self, data: TaskEnd) -> None:
            self.output_write.close()


    dataset = load_dataset(input_path=input_path, template=template, batch_size=batch_size, expected_categories=expected_categories)
    
    # Generate system prompt with specified granularity and batch size
    system_prompt = create_system_prompt(granularity, batch_size)

    return Task(
        dataset=dataset,
        solver=[
            system_message(system_prompt),
            generate(),
        ],
        scorer=category_count_scorer(),
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="Categories",
                json_schema=json_schema(CategoryList),
                strict=True
            ),
        ),
        hooks=["CategoriseOutputHook"]
    )

@task
def abstract(level: int = None, granularity: str = "medium"):
    """
    Create abstracted categories from lower-level concepts.
    
    Args:
        level: The level to abstract from (defaults to max level)
        granularity: Level of abstraction granularity based on compression ratio
                    - "fine": 40% compression (many output categories)
                    - "medium": 20% compression (balanced) 
                    - "coarse": 10% compression (few output categories)
    """
    if not (config.Categories.category_dir / "0.jsonl").exists():
        shutil.copy(config.Keywords.keywords_path, config.Categories.category_dir / "0.jsonl")
    if level is None:
        level = config.Categories.get_max_level()
    input_file = config.Categories.category_dir / f"{level}.jsonl"
    output_file = config.Categories.category_dir / f"{level + 1}.jsonl"
    
    return categorise(
        input_path=input_file, 
        output_path=output_file, 
        template=config.Template(), 
        batch_size=config.Categories.batch_size,
        granularity=granularity
    )

def load_classification_dataset(targets_text, data, data_type, source_level) -> MemoryDataset:
    template = config.Template(output_format="md", keys=["name", "description"], type=data_type)
    samples = []
    item_type = f"level {source_level} items" if source_level > 0 else "keywords"
    header = f"Classify the following {item_type} into one of the categories provided. Respond ONLY with the letter corresponding to your chosen category. Do NOT include any additional text, explanations, or formatting.\n\n"
    for _, item in data.iterrows():  # Use iterrows() to get both index and row
        item_dict = item.to_dict()  # Convert pandas Series to dictionary
        item_text = template(item_dict)
        samples.append(Sample(input=f"{header}{item_text}", choices=targets_text, metadata={data_type: item_dict['name']}))
    dataset = MemoryDataset(samples)
    return dataset

MCQ_PROMPT = r"""
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: {{LETTER}}' (without quotes and always include the literal "ANSWER:" precedign the actual letter answer) where LETTER is one of {letters}.

{question}

{choices}
""".strip()
@task
def classify(target_level: int = None, source_level: int = 0):

    if target_level is None:
        target_level = config.Categories.get_max_level()

    category_md_template = config.Template(output_format="md", keys=["name", "description"], type="category")
    targets = config.Categories.load(target_level)
    targets_text = targets.apply(category_md_template, axis=1).tolist()
    data = config.Categories.load(level=source_level)
    data_type = f"level_{source_level}_item"
    dataset = load_classification_dataset(targets_text, data, data_type, source_level)

    @hooks(name="ClassificationOutputHook", description="Hook to save classification results as JSON files")
    class ClassificationOutputHook(Hooks):
        
        def __init__(self):
            self.writer = jsonlines.open(config.Categories.category_dir / f"{source_level}-{target_level}.mapping.jsonl", 'w')

        async def on_sample_end(self, data: SampleEnd) -> None:
            answer_letter = parse_answers(data.sample, multiple_correct=False)
            choice_index = answer_index(list(answer_letter)[0])
            metadata_key = data_type  # Use the data_type as the metadata key
            self.writer.write({f"level_{source_level}": data.sample.metadata[metadata_key], f"level_{target_level}": targets.iloc[choice_index]['name']})

        async def on_task_end(self, data: TaskEnd) -> None:
            self.writer.close()

    
    return Task(
        dataset=dataset,
        solver=[
            multiple_choice(template=MCQ_PROMPT),
        ],
        hooks=["ClassificationOutputHook"]
    )

