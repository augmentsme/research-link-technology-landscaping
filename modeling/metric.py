from inspect_ai.scorer import scorer, Score, Target, metric, Metric, SampleScore

@metric
def count() -> Metric:
    """Count metric that sums up all numerical values.
    
    This is specifically designed for taxonomy category counting,
    where we want to sum the counts rather than calculate accuracy.
    
    Returns:
        Count metric that sums numerical values
    """
    
    def metric_impl(scores: list[SampleScore]) -> float:
        total = 0.0
        valid_scores = 0
        
        for item in scores:
            if isinstance(item.score.value, (int, float)):
                total += float(item.score.value)
                valid_scores += 1
        
        # Return the total count, or 0 if no valid scores
        return total if valid_scores > 0 else 0.0
    
    return metric_impl
