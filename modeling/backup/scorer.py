from inspect_ai.scorer import scorer, Score, Target, mean
from inspect_ai.solver import TaskState
import json
import utils

@scorer(metrics=[mean()])
def keywords_coverage():

    async def score(state: TaskState, target: Target) -> Score:
        """Score based on keyword coverage discrepancy."""
        # result = json.loads(state.output.completion)
        _, result = utils.parse_html(state.output)
        category_list = result["categories"]

        output_terms = [kw for category in category_list for kw in category['keywords']]
        input_keywords = state.metadata['keywords']
        input_terms = [kw for kw in input_keywords]

        covered_terms = set(input_terms).intersection(output_terms)
        missed_terms = set(input_terms).difference(output_terms)
        missed_terms = [term for term in input_terms if term in missed_terms]

        return Score(value=len(covered_terms) / len(input_terms), explanation=f"Coverage Score (missing {missed_terms})", metadata={"missing": list(missed_terms)})

    return score
