from inspect_ai.scorer import scorer, Score, Target, metric, Metric, SampleScore

@metric
def total() -> Metric:
    def metric_impl(scores: list[SampleScore]) -> float:
        return sum(float(item.score.value) for item in scores if item.score.value is not None)
    return metric_impl
