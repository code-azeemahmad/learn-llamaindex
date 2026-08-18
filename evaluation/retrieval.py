from llama_index.core.evaluation import RetrieverEvaluator


def create_retriever_evaluator(retriever):
    """
    Initializes a RetrieverEvaluator configured for Hit Rate and MRR metrics.
    """
    return RetrieverEvaluator.from_metric_names(
        metric_names=["hit_rate", "mrr"],
        retriever=retriever
    )