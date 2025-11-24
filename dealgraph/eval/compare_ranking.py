# ABOUTME: CLI for comparing heuristic vs ML ranking on DealGraph Bench.
# ABOUTME: Computes retrieval metrics for baseline and DealRanker.

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from ..agent import tools as agent_tools
from ..eval.bench_dataset import load_dealgraph_bench
from ..eval.metrics import precision_at_k, recall_at_k, ndcg_at_k
from ..config.settings import settings

logger = logging.getLogger(__name__)


def compare_ranking(
    model_path: Optional[Path] = None,
    max_queries: Optional[int] = None,
) -> Dict[str, object]:
    bench = load_dealgraph_bench()
    if max_queries is not None:
        bench = bench[:max_queries]

    if not agent_tools._search_manager.is_ready():
        agent_tools.build_search_index()

    baseline_metrics: List[Dict[str, float]] = []
    ml_metrics: List[Dict[str, float]] = []
    per_query_results = []

    for query in bench:
        raw_candidates = agent_tools.tool_graph_semantic_search(
            query=query.text,
            top_k=10,
            include_explanations=False,
        )
        if not raw_candidates:
            continue

        baseline_ids = [candidate["deal_id"] for candidate in raw_candidates]
        ml_ranked = agent_tools.tool_rank_deals_ml(
            query.text,
            raw_candidates,
            model_path=str(model_path) if model_path else None,
        )
        ml_ids = [ranked.candidate.deal.id for ranked in ml_ranked]

        baseline_metrics.append(_compute_metrics(baseline_ids, query.relevant_deal_ids))
        ml_metrics.append(_compute_metrics(ml_ids, query.relevant_deal_ids))
        per_query_results.append(
            {
                "query_id": query.id,
                "baseline": baseline_metrics[-1],
                "ml": ml_metrics[-1],
                "delta": _delta_metrics(ml_metrics[-1], baseline_metrics[-1]),
            }
        )

    summary = {
        "queries_evaluated": len(per_query_results),
        "baseline": _average_metrics(baseline_metrics),
        "ml_ranker": _average_metrics(ml_metrics),
        "delta": _delta_metrics(
            _average_metrics(ml_metrics),
            _average_metrics(baseline_metrics),
        ),
        "per_query": per_query_results,
    }
    return summary


def _compute_metrics(predicted_ids: List[str], relevant_ids: List[str]) -> Dict[str, float]:
    return {
        "precision@3": precision_at_k(predicted_ids, relevant_ids, 3),
        "recall@3": recall_at_k(predicted_ids, relevant_ids, 3),
        "ndcg@3": ndcg_at_k(predicted_ids, relevant_ids, 3),
        "precision@5": precision_at_k(predicted_ids, relevant_ids, 5),
        "recall@5": recall_at_k(predicted_ids, relevant_ids, 5),
        "ndcg@5": ndcg_at_k(predicted_ids, relevant_ids, 5),
    }


def _average_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {}
    aggregated: Dict[str, List[float]] = {}
    for metric in metrics:
        for key, value in metric.items():
            aggregated.setdefault(key, []).append(value)
    return {key: sum(values) / len(values) for key, values in aggregated.items()}


def _delta_metrics(candidate: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    keys = set(candidate.keys()) | set(baseline.keys())
    return {key: candidate.get(key, 0.0) - baseline.get(key, 0.0) for key in keys}


def main():
    parser = argparse.ArgumentParser(
        description="Compare heuristic vs ML ranking on DealGraph Bench."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=settings.MODELS_DIR / "deal_ranker_v1.pkl",
        help="Path to trained ranker.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit number of bench queries.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output file.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = compare_ranking(args.model, args.max_queries)
    serialized = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
        logger.info("Ranking comparison written to %s", args.output)
    else:
        print(serialized)


if __name__ == "__main__":
    from ..config.settings import settings
    main()
