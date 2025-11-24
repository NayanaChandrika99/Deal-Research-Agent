# ABOUTME: Prompt comparison CLI for evaluating baseline vs optimized reasoning prompts.
# ABOUTME: Loads DealGraph Bench, runs retrieval + reasoning, and aggregates quality metrics.

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from .bench_dataset import BenchQuery, load_dealgraph_bench
from ..agent import tools as agent_tools
from ..data.schemas import RankedDeal, DealReasoningOutput
from ..reasoning.reasoner import deal_reasoner
from ..reasoning.dspy.evaluator import PerformanceEvaluator

logger = logging.getLogger(__name__)


MetricsEvaluator = Callable[[DealReasoningOutput, BenchQuery], Dict[str, float]]
SearchRunner = Callable[[BenchQuery], List[RankedDeal]]
ReasonerRunner = Callable[[str, List[RankedDeal], str], DealReasoningOutput]


class PromptComparator:
    """Compare two prompt versions across DealGraph Bench."""

    def __init__(
        self,
        bench_queries: Iterable[BenchQuery],
        search_runner: SearchRunner,
        reasoner_runner: ReasonerRunner,
        metrics_evaluator: MetricsEvaluator,
    ):
        self._bench_queries = list(bench_queries)
        self._search_runner = search_runner
        self._reasoner_runner = reasoner_runner
        self._metrics_evaluator = metrics_evaluator

    def run(
        self,
        baseline_version: str,
        candidate_version: str,
        max_queries: Optional[int] = None,
    ) -> Dict[str, any]:
        """Execute comparison and return aggregate metrics."""
        queries = self._bench_queries
        if max_queries is not None:
            queries = queries[:max_queries]

        per_query = []
        baseline_metrics = []
        candidate_metrics = []

        for bench_query in queries:
            ranked_deals = self._search_runner(bench_query)
            baseline_output = self._reasoner_runner(
                bench_query.text, ranked_deals, baseline_version
            )
            candidate_output = self._reasoner_runner(
                bench_query.text, ranked_deals, candidate_version
            )

            baseline_result = self._metrics_evaluator(baseline_output, bench_query)
            candidate_result = self._metrics_evaluator(candidate_output, bench_query)

            baseline_metrics.append(baseline_result)
            candidate_metrics.append(candidate_result)
            per_query.append(
                {
                    "query_id": bench_query.id,
                    "baseline": baseline_result,
                    "candidate": candidate_result,
                    "delta": _delta_metrics(candidate_result, baseline_result),
                }
            )

        baseline_avg = _average_metrics(baseline_metrics)
        candidate_avg = _average_metrics(candidate_metrics)

        return {
            "baseline_version": baseline_version,
            "candidate_version": candidate_version,
            "queries_evaluated": len(per_query),
            "baseline": baseline_avg,
            "candidate": candidate_avg,
            "delta": _delta_metrics(candidate_avg, baseline_avg),
            "per_query": per_query,
        }


def _average_metrics(metric_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_dicts:
        return {}
    aggregates: Dict[str, List[float]] = {}
    for metrics in metric_dicts:
        for key, value in metrics.items():
            aggregates.setdefault(key, []).append(float(value))
    return {
        key: sum(values) / len(values)
        for key, values in aggregates.items()
        if values
    }


def _delta_metrics(
    candidate: Dict[str, float], baseline: Dict[str, float]
) -> Dict[str, float]:
    keys = set(candidate.keys()) | set(baseline.keys())
    return {key: candidate.get(key, 0.0) - baseline.get(key, 0.0) for key in keys}


def create_default_comparator(max_queries: Optional[int] = None) -> PromptComparator:
    """Factory that wires the real retrieval + reasoning stack."""
    bench_queries = load_dealgraph_bench()
    if not agent_tools._search_manager.is_ready():
        agent_tools.build_search_index()

    def search_runner(bench_query: BenchQuery) -> List[RankedDeal]:
        raw_results = agent_tools.tool_graph_semantic_search(
            bench_query.text,
            top_k=5,
            include_explanations=False,
        )
        return agent_tools.tool_rank_deals(bench_query.text, raw_results, top_k=5)

    evaluator = _PerformanceEvaluatorAdapter()

    return PromptComparator(
        bench_queries=bench_queries,
        search_runner=search_runner,
        reasoner_runner=_default_reasoner_runner,
        metrics_evaluator=evaluator,
    )


class _PerformanceEvaluatorAdapter:
    """Adapt PerformanceEvaluator to the comparator interface."""

    def __init__(self):
        self._evaluator = PerformanceEvaluator()

    def __call__(
        self, reasoning_output: DealReasoningOutput, bench_query: BenchQuery
    ) -> Dict[str, float]:
        reference = {
            "precedents": [
                {"deal_id": deal_id, "name": deal_id, "similarity_reason": ""}
                for deal_id in bench_query.relevant_deal_ids
            ]
        }
        return self._evaluator.evaluate_reasoning_output(
            reasoning_output.model_dump(), reference
        )


def _default_reasoner_runner(
    query_text: str, ranked_deals: List[RankedDeal], prompt_version: str
) -> DealReasoningOutput:
    return deal_reasoner(
        query=query_text,
        ranked_deals=ranked_deals,
        max_deals=len(ranked_deals),
        prompt_version=prompt_version,
    )


def compare_versions(
    baseline_version: str,
    candidate_version: str,
    max_queries: Optional[int] = None,
) -> Dict[str, any]:
    comparator = create_default_comparator(max_queries=max_queries)
    return comparator.run(
        baseline_version=baseline_version,
        candidate_version=candidate_version,
        max_queries=max_queries,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare DealGraph reasoning prompt versions."
    )
    parser.add_argument("--baseline", default="v1", help="Baseline prompt version.")
    parser.add_argument("--candidate", default="latest", help="Candidate prompt version.")
    parser.add_argument(
        "--max-queries", type=int, default=None, help="Limit number of bench queries."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write JSON results (prints to stdout if omitted).",
    )
    args = parser.parse_args()

    result = compare_versions(
        baseline_version=args.baseline,
        candidate_version=args.candidate,
        max_queries=args.max_queries,
    )

    serialized = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
        logger.info("Wrote comparison results to %s", args.output)
    else:
        print(serialized)


if __name__ == "__main__":
    main()
