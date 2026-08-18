import sys
from pathlib import Path

# 1. Set path BEFORE importing local package modules
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import asyncio
import time
from typing import Any, Dict, List

from llama_index.core import Settings
from llama_index.core.evaluation import (
    AnswerRelevancyEvaluator,
    FaithfulnessEvaluator,
)

from evaluation.dataset import EVAL_DATASET
from evaluation.retrieval import create_retriever_evaluator


async def run_pipeline_evaluation(query_engine, retriever) -> List[Dict[str, Any]]:
    llm = Settings.llm

    retriever_evaluator = create_retriever_evaluator(retriever)
    faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
    answer_relevancy_evaluator = AnswerRelevancyEvaluator(llm=llm)

    results = []

    for item in EVAL_DATASET:
        query = item["query"]
        expected_ids = item["expected_node_ids"]

        # 1. Retrieval Stage Evaluation (Async)
        retrieval_res = await retriever_evaluator.aevaluate(
            query=query, expected_ids=expected_ids
        )

        # 2. System Latency and RAG Response Generation
        start_time = time.perf_counter()
        if hasattr(query_engine, "aquery"):
            response = await query_engine.aquery(query)
        else:
            response = query_engine.query(query)
        latency = time.perf_counter() - start_time

        # 3. Generation Faithfulness
        faith_res = await faithfulness_evaluator.aevaluate_response(response=response)

        # 4. Answer Relevancy
        relevancy_res = await answer_relevancy_evaluator.aevaluate(
            query=query, response=str(response)
        )

        # Extract metric values
        metrics = retrieval_res.metric_vals_dict

        # Safely extract scores defaulting None to 0.0
        faith_score = faith_res.score if faith_res.score is not None else 0.0
        relevancy_score = relevancy_res.score if relevancy_res.score is not None else 0.0

        results.append({
            "query": query,
            "hit_rate": metrics.get("hit_rate", 0.0) or 0.0,
            "mrr": metrics.get("mrr", 0.0) or 0.0,
            "faithfulness_score": faith_score,
            "faithfulness_pass": bool(faith_res.passing),
            "relevancy_score": relevancy_score,
            "relevancy_pass": bool(relevancy_res.passing),
            "latency_sec": round(latency, 3),
        })

    return results


def print_evaluation_report(results: List[Dict[str, Any]]):
    print("\n" + "=" * 60)
    print("                      EVALUATION REPORT                      ")
    print("=" * 60)

    total_queries = len(results)
    if total_queries == 0:
        print("No evaluation results to display.")
        return

    avg_hit_rate = sum(r["hit_rate"] for r in results) / total_queries
    avg_mrr = sum(r["mrr"] for r in results) / total_queries
    avg_faithfulness = sum(r["faithfulness_score"] for r in results) / total_queries
    avg_relevancy = sum(r["relevancy_score"] for r in results) / total_queries
    avg_latency = sum(r["latency_sec"] for r in results) / total_queries

    for r in results:
        print(f"\nQUERY: {r['query']}")
        print(f"  |-- Retrieval   : Hit Rate = {r['hit_rate']} | MRR = {r['mrr']:.3f}")
        print(f"  |-- Faithfulness: Score = {r['faithfulness_score']:.2f} (Pass: {r['faithfulness_pass']})")
        print(f"  |-- Relevancy   : Score = {r['relevancy_score']:.2f} (Pass: {r['relevancy_pass']})")
        print(f"  \\-- Latency     : {r['latency_sec']}s")

    print("\n" + "-" * 60)
    print("AGGREGATE SUMMARY:")
    print(f"  * Avg Hit Rate    : {avg_hit_rate:.3f}")
    print(f"  * Avg MRR         : {avg_mrr:.3f}")
    print(f"  * Avg Faithfulness: {avg_faithfulness:.3f}")
    print(f"  * Avg Relevancy   : {avg_relevancy:.3f}")
    print(f"  * Avg Latency     : {avg_latency:.3f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        from main import query_engine, retriever
    except ImportError:
        print("Error: Could not import 'query_engine' or 'retriever' from main.py.")
        print("Please ensure main.py exists in project root and exports these objects.")
        sys.exit(1)

    async def main():
        print("Starting RAG evaluation pipeline...")
        results = await run_pipeline_evaluation(query_engine, retriever)
        print_evaluation_report(results)

    asyncio.run(main())