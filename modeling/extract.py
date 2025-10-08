import jsonlines
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset, FieldSpec, Sample, MemoryDataset
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.solver import system_message, generate, user_message, TaskState
from inspect_ai.model import get_model
from inspect_ai.util import json_schema
from inspect_ai.scorer import scorer, Score, Target, accuracy
from inspect_ai.hooks import Hooks, SampleEnd, TaskEnd, hooks

from models import KeywordType, Keyword, KeywordsList
import utils
import config

import json
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field
import re
import pandas as pd
from pathlib import Path
from process import postprocess_keywords
# Evaluation schema for the quality scorer
class OverallScores(BaseModel):
    """Overall quality scores for keyword evaluation"""
    text_presence: int = Field(..., ge=1, le=10, description="Text presence score 1-10")
    specificity: int = Field(..., ge=1, le=10, description="Technical precision score 1-10")
    uniqueness: int = Field(..., ge=1, le=10, description="Uniqueness score 1-10") 
    technical_precision: int = Field(..., ge=1, le=10, description="Technical precision score 1-10")

class KeywordEvaluation(BaseModel):
    """Individual keyword evaluation"""
    keyword: str = Field(..., description="The keyword being evaluated")
    should_reject: bool = Field(..., description="Whether this keyword should be rejected")
    rejection_reason: Optional[str] = Field(None, description="Reason for rejection if applicable")

class QualityEvaluation(BaseModel):
    """Complete quality evaluation response"""
    overall_scores: OverallScores = Field(..., description="Overall quality scores")
    keyword_evaluations: List[KeywordEvaluation] = Field(..., description="Individual keyword evaluations")
    average_score: float = Field(..., ge=1, le=10, description="Average of the four overall scores")
    explanation: str = Field(..., description="Brief explanation of the evaluation")

@scorer(metrics=[accuracy()])
def keyword_quality_scorer():
    """
    Score the quality of extracted keywords using an LLM evaluator.
    Evaluates specificity, uniqueness, and technical precision of keywords.
    """
    async def score(state: TaskState, target: Target) -> Score:
        # Get the extracted keywords from the model output
        try:
            output_text = state.output.completion
            result_json = json.loads(output_text)
            keywords = result_json.get("keywords", [])
            
            if not keywords:
                return Score(value=0.0, explanation="No keywords extracted")
            
            # Create evaluation prompt
            keywords_text = "\n".join([f"- {kw['name']}: {kw['description']}" for kw in keywords])
            
            evaluation_prompt = f"""
You are an expert research analyst evaluating the quality of extracted keywords from a research grant.

Original Grant Input:
{state.input_text}

Extracted Keywords:
{keywords_text}

**CRITICAL REQUIREMENT - Keywords Must Exist in Grant Text:**
All keywords MUST be directly derivable from or explicitly mentioned in the original grant input text above (including both the grant title and description). 

Evaluate these keywords on four criteria (score 1-10 for each):

1. **TEXT_PRESENCE** (1-10): Do the keywords actually appear in or are directly derivable from the grant text (title and description)?
   - Score 10: Keywords are explicitly mentioned or clearly derivable from specific text in the grant title or description
   - Score 1: Keywords appear to be invented or not found anywhere in the grant title or description

2. **SPECIFICITY** (1-10): Are the keywords technically precise and domain-specific rather than generic?
   - Score 10: Highly specific technical terms (e.g., "quantum-enhanced magnetometry", "CRISPR-Cas13 diagnostics")
   - Score 1: Generic terms (e.g., "research", "innovation", "technology")

3. **UNIQUENESS** (1-10): Would these keywords uniquely identify this grant among thousands of others?
   - Score 10: Keywords could only apply to this specific research or very similar projects
   - Score 1: Keywords could apply to 80%+ of research grants

4. **TECHNICAL_PRECISION** (1-10): Do the keywords capture the core technical essence of the research?
   - Score 10: Keywords precisely capture novel methodologies, specific technologies, or unique approaches
   - Score 1: Keywords are vague or could describe any research project

For each keyword, also identify if it should be REJECTED based on these criteria:
- **Not present in the grant text (title or description) (MANDATORY REJECTION)**
- Too generic (could apply to most grants)
- Administrative/process terms
- Funding-related boilerplate

Provide your evaluation following the required JSON schema.
"""
            
            # Use evaluator model with proper schema
            # evaluator_model = get_model("openai/Qwen/Qwen3-4B-Instruct-2507", base_url="http://localhost:8001/v1")  # Use the same model for evaluation
            evaluator_model = get_model()
            
            # Create evaluation config with proper schema
            eval_config = GenerateConfig(
                response_schema=ResponseSchema(
                    name="quality_evaluation",
                    json_schema=json_schema(QualityEvaluation),
                    strict=True
                )
            )
            
            # Generate evaluation
            eval_response = await evaluator_model.generate(
                evaluation_prompt,
                config=eval_config
            )
            eval_text = eval_response.completion
            
            # Parse evaluation result
            try:
                eval_result = json.loads(eval_text)
                
                # Calculate metrics from structured response
                overall_scores = eval_result.get("overall_scores", {})
                text_presence = overall_scores.get("text_presence", 0)
                specificity = overall_scores.get("specificity", 0)
                uniqueness = overall_scores.get("uniqueness", 0) 
                technical_precision = overall_scores.get("technical_precision", 0)
                
                average_score = eval_result.get("average_score", 0)
                rejected_count = sum(1 for kw_eval in eval_result.get("keyword_evaluations", []) 
                                   if kw_eval.get("should_reject", False))
                acceptance_rate = (len(keywords) - rejected_count) / len(keywords) if keywords else 0
                
                # Normalize score to 0-1 range
                final_score = (average_score / 10.0) * acceptance_rate
                
                explanation = f"Average quality: {average_score:.1f}/10, " \
                            f"Accepted: {len(keywords) - rejected_count}/{len(keywords)} keywords. " \
                            f"{eval_result.get('explanation', '')}"
                
                return Score(
                    value=final_score,
                    explanation=explanation,
                    metadata={
                        "text_presence": text_presence,
                        "specificity": specificity,
                        "uniqueness": uniqueness, 
                        "technical_precision": technical_precision,
                        "total_keywords": len(keywords),
                        "rejected_keywords": rejected_count,
                        "acceptance_rate": acceptance_rate,
                        "keyword_evaluations": eval_result["keyword_evaluations"],
                        "evaluation_result": eval_result  # Store full evaluation for hook
                    }
                )
                
            except json.JSONDecodeError:
                return Score(value=0.0, explanation=f"Failed to parse evaluation response: {eval_text[:200]}...")
                
        except Exception as e:
            return Score(value=0.0, explanation=f"Error in keyword evaluation: {str(e)}")
    
    return score

@hooks(name="KeywordsExtractionHook", description="Hook to save keywords extraction results as JSON files")
class KeywordsExtractionHook(Hooks):
    """Hook to save deduplicated keywords extraction results as JSON files."""

    def __init__(self):
        self.extracted_keywords_writer = jsonlines.open(config.Keywords.extracted_keywords_path, 'a')

    async def on_sample_end(self, data: SampleEnd) -> None:
        """Collect keywords and filter out rejected ones using evaluation results from scorer."""

        output_text = data.sample.output.completion
        result_json = json.loads(output_text)
        grant_id = data.sample.id
        keywords = result_json["keywords"]
        
        if not keywords:
            return
        
        # Get evaluation results from sample scores
        evaluation_result = None
        if data.sample.scores and "keyword_quality_scorer" in data.sample.scores:
            scorer_result = data.sample.scores["keyword_quality_scorer"]
            if scorer_result.metadata:
                evaluation_result = scorer_result.metadata.get("evaluation_result")
        
        if evaluation_result:
            # Use evaluation results from scorer
            keyword_evaluations = evaluation_result.get("keyword_evaluations", [])
            
            # Create a mapping of keyword names to rejection status
            rejection_map = {
                kw_eval["keyword"]: kw_eval.get("should_reject", False) 
                for kw_eval in keyword_evaluations
            }
            
            # Filter and save only non-rejected keywords
            accepted_keywords = []
            for keyword in keywords:
                keyword_name = keyword["name"]
                should_reject = rejection_map.get(keyword_name, False)
                
                if not should_reject:
                    keyword_with_grant = keyword.copy()
                    keyword_with_grant["grant_id"] = grant_id
                    self.extracted_keywords_writer.write(keyword_with_grant)
                    accepted_keywords.append(keyword_name)
            
            # Log filtering results
            rejected_count = len(keywords) - len(accepted_keywords)
            if rejected_count > 0:
                rejected_keywords = [kw["name"] for kw in keywords if kw["name"] not in accepted_keywords]
                print(f"Grant {grant_id}: Filtered out {rejected_count}/{len(keywords)} keywords: {rejected_keywords}")
        else:
            # Fallback: save all keywords if evaluation results not available
            print(f"Warning: No evaluation results found for grant {grant_id}. Saving all keywords.")
            for keyword in keywords:
                keyword_with_grant = keyword.copy()
                keyword_with_grant["grant_id"] = grant_id
                self.extracted_keywords_writer.write(keyword_with_grant)

    async def on_task_end(self, data: TaskEnd) -> None:
        # Deduplicate keywords using normalized terms

        self.extracted_keywords_writer.close()
        
        postprocess_keywords()

def load_extract_dataset():
    records = config.Grants.load(as_dataframe=False)
    samples = []
    for record in records:
        grant_id = record.get("id") or record.get("key") or ""
        samples.append(
            Sample(
                id=grant_id,
                input=config.Grants.template(record),
                metadata={
                    "title": record.get("title", ""),
                    "summary": record.get("grant_summary", ""),
                    "funding_amount": record.get("funding_amount"),
                    "funder": record.get("funder", ""),
                    "start_year": record.get("start_year"),
                    "end_year": record.get("end_year"),
                }
            )
        )

    return MemoryDataset(samples)
SYSTEM_PROMPT = f"""

You are an expert research analyst with deep knowledge across multiple academic disciplines and a keen eye for emerging research trends.

Your task is to extract meaningful keywords from research grant information that would be useful for:
- Identifying emerging research domains and interdisciplinary areas
- Discovering novel methodologies and cutting-edge approaches
- Tracking innovative technologies and emerging tools
- Finding related research projects working on similar frontiers
- Understanding emerging research trends and future directions

**FUNDAMENTAL REQUIREMENT - Keywords Must Exist in Grant Text:**
**ALL KEYWORDS MUST BE DIRECTLY PRESENT IN OR CLEARLY DERIVABLE FROM THE PROVIDED GRANT TEXT (INCLUDING BOTH TITLE AND DESCRIPTION).**
- Only extract keywords that appear explicitly in the grant title, description, or summary
- Keywords can be technical terms, methodologies, technologies, or concepts mentioned in the text
- Do NOT invent or infer keywords that are not clearly present in the source material
- If a concept is implied but not explicitly mentioned, DO NOT include it as a keyword

Focus on extracting keywords that highlight what's new, innovative, and emerging in the research landscape. Prioritize:
- Technical terms that represent novel concepts or emerging fields mentioned in the grant
- Methodologies that are cutting-edge or represent new approaches described in the text
- Technologies that are innovative or represent emerging tools referenced in the grant
- Applications that address new challenges or emerging needs as stated in the grant
- Scientific terminology that indicates research at the frontiers of knowledge as described

**CRITICAL REQUIREMENT - Avoid Overly Broad Keywords:**
- **Keywords must be specific and precise, not generic or overly broad**
- **Instead of "engineering," use "bio-integrated nano-photonics" or "quantum-enhanced engineering"**
- **Instead of "chemistry," use "supramolecular photochemistry" or "catalytic asymmetric synthesis"**
- **Instead of "data analysis," use "multi-modal time-series analysis" or "causal inference modeling"**
- **Instead of "artificial intelligence," use "graph neural networks" or "federated learning algorithms"**
- **Avoid general terms like "research," "development," "innovation," "technology," "analysis," "method"**
- **Each keyword should clearly indicate a specific domain, technique, or application area**
- **Do NOT extract specific country names**

**UNIQUENESS REQUIREMENT - Keywords Must Be Grant-Specific:**
- **REJECT keywords that could apply to 80% or more of research grants (e.g., "interdisciplinary research," "international collaboration," "innovative approach," "cutting-edge technology")**
- **REJECT administrative or process keywords (e.g., "project management," "research methodology," "data collection," "literature review")**
- **REJECT funding-related terms (e.g., "research funding," "grant application," "collaborative research," "research partnership")**
- **REJECT generic outcome terms (e.g., "scientific advancement," "knowledge creation," "research impact," "societal benefit")**
- **Each keyword should be SO SPECIFIC that it could only apply to this particular grant or a very small subset of similar grants**
- **If a keyword could reasonably appear in a generic grant template or boilerplate text, REJECT it**
- **Keywords should capture the UNIQUE technical essence that distinguishes this specific research from all other research**

**SPECIFICITY TEST:**
Before including any keyword, ask: "Could this keyword appear in 100+ different grant applications across various fields?" If YES, REJECT it.
Only extract keywords that are:
1. **Actually present in the grant text (title and description) (MANDATORY)**
2. Technically precise and domain-specific
3. Unique to the particular research approach or subject matter
4. Would help distinguish this grant from 95% of other grants in the database

**LENGTH REQUIREMENT:**
- **Each keyword must contain only 1-4 words maximum**
- **Use concise, specific terminology rather than lengthy phrases**
- **Examples: "quantum dots," "CRISPR-Cas9," "machine learning," "photonic crystals"**
- **Avoid: "advanced quantum dot synthesis techniques" (too long) → use "quantum dot synthesis"**

Provide accurate, specific keywords that capture the innovative and emerging aspects of the research while ensuring they are all grounded in the actual grant text (both title and description). Prioritize technical precision over comprehensiveness.


"""

@task
def extract() -> Task:
    # finished_grants_list = finished_grants()
    dataset = load_extract_dataset()
    # if filter_finished:
        # dataset = dataset.filter(lambda sample: sample.id not in finished_grants_list)
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
        scorer=[
            keyword_quality_scorer()
        ],
        hooks=["KeywordsExtractionHook"],
    )


def finished_grants():
    """
    Returns a list of grant IDs that have already had keywords extracted.
    Checks the extracted_keywords_path instead of the final keywords_path.
    """
    if config.Keywords.extracted_keywords_path.exists():
        keywords = utils.load_jsonl_file(config.Keywords.extracted_keywords_path, as_dataframe=True)
        if keywords.empty:
            return []
        return keywords.grant_id.unique()
    else:
        return []

