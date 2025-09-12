from inspect_ai.scorer import scorer, Score, Target, mean, std
from inspect_ai.solver import TaskState
import json
import utils

@scorer(metrics=[mean(), std()])
def keywords_confusion_matrix():
    # def __init__(self):
        # self.output_write = jsonlines.open(config.Categories.category_proposal_path, 'a')
        # self.unknown_write = jsonlines.open(config.Categories.unknown_keywords_path, 'a')
        # self.missing_write = jsonlines.open(config.Categories.missing_keywords_path, 'a')
    async def score(state: TaskState, target: Target) -> Score:
        """Compute confusion matrix metrics for keyword classification."""
        _, result = utils.parse_results(state.output.completion)
        category_list = result["categories"]

        output_terms = set([kw for category in category_list for kw in category['keywords']])
        input_keywords = state.metadata['keywords']
        input_terms = set([kw for kw in input_keywords])

        # Confusion matrix components
        true_positives = input_terms.intersection(output_terms)  # Correctly included keywords
        false_positives = output_terms.difference(input_terms)   # Hallucinated keywords
        false_negatives = input_terms.difference(output_terms)   # Missed keywords
        # True negatives: keywords that were correctly NOT included (infinite set, so we don't compute this)

        tp_count = len(true_positives)
        fp_count = len(false_positives)
        fn_count = len(false_negatives)
        
        # Compute standard metrics
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 1.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 1.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metadata = {
            "tp": list(true_positives),
            "fp": list(false_positives), 
            "fn": list(false_negatives),
            "tp_count": tp_count,
            "fp_count": fp_count,
            "fn_count": fn_count,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

        return Score(
            value=recall, 
            explanation=f"F1: {f1_score:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f} (TP:{tp_count}, FP:{fp_count}, FN:{fn_count})", 
            metadata=metadata
        )

    return score


@scorer(metrics=[mean(), std()])
def category_confusion_matrix():
    async def score(state: TaskState, target: Target) -> Score:
        """Compute confusion matrix metrics for category merging."""
        _, result = utils.parse_results(state.output.completion)
        merged_categories = result["categories"]

        # Extract referenced category names from merge output
        output_category_names = set()
        for merged_cat in merged_categories:
            output_category_names.update(merged_cat.get('source_categories', []))

        # Extract input category names from metadata
        input_category_names = set()
        if 'categories' in state.metadata:
            for cat in state.metadata['categories']:
                input_category_names.add(cat.get('name', ''))
        elif 'total_categories_in_batch' in state.metadata:
            # Fallback: try to extract from sample input if available
            try:
                # Parse the input to extract category names
                lines = state.input.split('\n')
                for line in lines:
                    if line.startswith('**') and '**:' in line:
                        name = line.split('**')[1].strip()
                        input_category_names.add(name)
            except:
                pass

        # Confusion matrix components
        true_positives = input_category_names.intersection(output_category_names)  # Correctly referenced categories
        false_positives = output_category_names.difference(input_category_names)   # Hallucinated category references
        false_negatives = input_category_names.difference(output_category_names)   # Missed categories

        tp_count = len(true_positives)
        fp_count = len(false_positives)
        fn_count = len(false_negatives)
        
        # Compute standard metrics
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 1.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 1.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metadata = {
            "tp": list(true_positives),
            "fp": list(false_positives), 
            "fn": list(false_negatives),
            "tp_count": tp_count,
            "fp_count": fp_count,
            "fn_count": fn_count,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

        return Score(
            value=recall, 
            explanation=f"Category F1: {f1_score:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f} (TP:{tp_count}, FP:{fp_count}, FN:{fn_count})", 
            metadata=metadata
        )

    return score
