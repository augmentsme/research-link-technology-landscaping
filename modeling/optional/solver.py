from inspect_ai.solver import TaskState, Generate, solver
from inspect_ai import Task, task, Task, eval
from inspect_ai.dataset import json_dataset, Sample
import asyncio
import config
import json
from inspect_ai.model import GenerateConfig, ResponseSchema, ChatMessageUser
from inspect_ai.util import json_schema, StoreModel
from matplotlib.pylab import source
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from inspect_ai.model import get_model, ResponseSchema, ModelOutput
from pydantic import BaseModel, Field
import numpy as np
from typing import List


import llama_cpp

llm = llama_cpp.Llama(model_path="/Users/luhancheng/Library/Caches/llama.cpp/Qwen_Qwen3-Embedding-0.6B-GGUF_Qwen3-Embedding-0.6B-f16.gguf", embedding=True)

# embeddings = llm.create_embedding("Hello, world!")
# embeddings['data'][1]['embedding']

@solver(name="EmbeddingSolver")
def embed():
    # model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    async def solve(state: TaskState, generate: Generate):
        embeddings = llm.create_embedding(input=state.input)
        state.metadata['embeddings'] = embeddings['data'][0]['embedding']
        return state
    return solve

@solver(name="ClusteringSolver")
def cluster():
    async def solve(state: TaskState, generate: Generate):
        embeddings = state.metadata['embeddings']
        cluster = DBSCAN()
        clusters = cluster.fit_predict(embeddings)
        state.metadata['cluster_labels'] = clusters
        return state
    return solve


class Concept(BaseModel):
    name: str = Field(..., description="The name of the proposed concept")
    description: str = Field(..., description="A brief description of the proposed concept")
    sources: "List[Concept]" = Field(..., description="The source concepts")
    
    
def concept_as_input_template(input):
    return f"""<concept><name>{input.name}</name><description>{input.description}</description><sources>{', '.join([s.name for s in input.sources])}</sources></concept>"""


@solver(name="ProposeConceptSolver")
def propose_concept():
    async def solve(state: TaskState, generate: Generate):
        clusters = state.metadata['cluster_labels']
        unique_clusters = np.unique(clusters)

        coroutines = []
        for cluster in unique_clusters:
            concept = Concept(
                name=f"Cluster {cluster}",
                description=f"A concept representing cluster {cluster}",
                sources=[Concept(name=f"Source {i}", description=f"Description for source {i}", sources=[]) for i in range(3)]
            )
            input_template = concept_as_input_template(concept)
            output_coroutine: ModelOutput = generate(
                input=input_template,
                config=GenerateConfig(
                    response_schema=ResponseSchema(
                        name="Concept",
                        description="A concept is a general idea or understanding of something.",
                        json_schema=json_schema(List[Concept]),
                        strict=True
                    )
                )
            )
            coroutines.append(output_coroutine)


        raw_results = await asyncio.gather(*coroutines)
        results = [json.loads(output.completion) for output in raw_results]
        state.messages.append(ChatMessageUser(content=json.dumps(results)))

        return state
    return solve

def keyword_record_to_sample(record):
    return Sample(
        id=record['term'],
        input=config.Keywords.template(record),
    )

@task()
def abstract():
    return Task(
        dataset=json_dataset(str(config.Keywords.keywords_path), keyword_record_to_sample),
        solver=[
            embed(),  
        ],
    )

eval(abstract(), model="openai/gpt-5-mini-2025-08-07", limit=1)
