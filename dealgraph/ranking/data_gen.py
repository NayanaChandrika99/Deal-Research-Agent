# ABOUTME: Reverse-query data generator for ranking training records.
# ABOUTME: Samples clusters from DealGraph and asks the LLM to propose realistic queries.

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from ..config.settings import settings
from ..data.ingest import load_all, DealDataset
from ..data.schemas import Deal, Snippet, EventType
from ..reasoning.llm_client import LLMClient, LLMClientError

logger = logging.getLogger(__name__)


@dataclass
class DealCluster:
    cluster_id: str
    platform: Deal
    addons: List[Deal]
    snippets: List[Snippet]


def generate_synthetic_training_data(
    num_clusters: int = 50,
    queries_per_cluster: int = 3,
    negatives_per_query: int = 2,
    seed: Optional[int] = None,
    llm_client: Optional[LLMClient] = None,
    dataset: Optional[DealDataset] = None,
) -> List[Dict[str, object]]:
    """
    Generate synthetic (query, candidate, label) examples using reverse queries.
    """
    dataset = dataset or load_all(_default_data_path())
    rng = random.Random(seed)
    clusters = _sample_clusters(dataset, num_clusters, rng)
    if not clusters:
        return []

    query_generator = ReverseQueryGenerator(llm_client)
    all_deals = {deal.id: deal for deal in dataset.deals}
    records: List[Dict[str, object]] = []

    for cluster in clusters:
        positives = [cluster.platform] + cluster.addons
        positive_ids = {deal.id for deal in positives}
        negative_pool = [
            deal for deal in dataset.deals if deal.id not in positive_ids
        ]

        queries = query_generator.generate(cluster, queries_per_cluster)

        for query_text in queries:
            for deal in positives:
                records.append(
                    _make_record(
                        query=query_text,
                        deal=deal,
                        label=1,
                        cluster_id=cluster.cluster_id,
                        reason="cluster_positive",
                    )
                )

            if negatives_per_query and negative_pool:
                sample_count = min(negatives_per_query, len(negative_pool))
                negative_choices = rng.sample(negative_pool, sample_count)
                for deal in negative_choices:
                    records.append(
                        _make_record(
                            query=query_text,
                            deal=deal,
                            label=0,
                            cluster_id=cluster.cluster_id,
                            reason="random_negative",
                        )
                    )

    logger.info(
        "Generated %s synthetic ranking records across %s clusters",
        len(records),
        len(clusters),
    )
    return records


def save_training_data(
    records: Sequence[Dict[str, object]],
    output_path: str | Path,
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    """Persist records (and optional metadata) to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(records), indent=2))
    if metadata:
        meta_file = path.with_suffix(path.suffix + ".metadata.json")
        meta_file.write_text(json.dumps(metadata, indent=2))


class ReverseQueryGenerator:
    """Wraps LLMClient to request realistic reverse queries."""

    SYSTEM_PROMPT = (
        "You create realistic private-equity precedent search queries. "
        "Return concise, domain-specific questions that a deal team would type."
    )

    def __init__(self, llm_client: Optional[LLMClient]):
        self.llm_client = llm_client

    def generate(self, cluster: DealCluster, count: int) -> List[str]:
        if not self.llm_client:
            return self._fallback_queries(cluster, count)

        user_prompt = _cluster_prompt(cluster, count)
        try:
            response = self.llm_client.complete_json(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.4,
                max_tokens=600,
            )
        except LLMClientError as exc:
            logger.warning("LLM query generation failed: %s", exc)
            return self._fallback_queries(cluster, count)

        queries = response.get("queries")
        if not isinstance(queries, list):
            logger.warning("Unexpected query payload: %s", response)
            return self._fallback_queries(cluster, count)
        return [str(query).strip() for query in queries if str(query).strip()]

    def _fallback_queries(self, cluster: DealCluster, count: int) -> List[str]:
        base = cluster.platform.name
        return [
            f"{base} precedent search concept #{idx + 1}"
            for idx in range(count)
        ]


def _make_record(
    query: str,
    deal: Deal,
    label: int,
    cluster_id: str,
    reason: str,
) -> Dict[str, object]:
    return {
        "query": query,
        "candidate_id": deal.id,
        "label": label,
        "cluster_id": cluster_id,
        "metadata": {
            "deal_name": deal.name,
            "sector_id": deal.sector_id,
            "region_id": deal.region_id,
            "query_source": "reverse_query",
            "reason": reason,
        },
    }


def _sample_clusters(
    dataset: DealDataset, num_clusters: int, rng: random.Random
) -> List[DealCluster]:
    deals_by_id: Dict[str, Deal] = {deal.id: deal for deal in dataset.deals}
    snippets_by_deal: Dict[str, List[Snippet]] = {}
    for snippet in dataset.snippets:
        snippets_by_deal.setdefault(snippet.deal_id, []).append(snippet)

    addon_map: Dict[str, List[str]] = {}
    for event in dataset.events:
        if event.type == EventType.ADDON and event.related_deal_id:
            addon_map.setdefault(event.deal_id, []).append(event.related_deal_id)

    platforms = [deal for deal in dataset.deals if deal.is_platform]
    rng.shuffle(platforms)
    clusters: List[DealCluster] = []

    for platform in platforms:
        related_ids = addon_map.get(platform.id, [])
        addons = [deals_by_id[deal_id] for deal_id in related_ids if deal_id in deals_by_id]
        if not addons:
            continue
        cluster_id = f"cluster_{platform.id}"
        clusters.append(
            DealCluster(
                cluster_id=cluster_id,
                platform=platform,
                addons=addons,
                snippets=snippets_by_deal.get(platform.id, []),
            )
        )
        if len(clusters) >= num_clusters:
            break

    return clusters


def _cluster_prompt(cluster: DealCluster, count: int) -> str:
    lines = [
        "Create realistic PE research queries that would surface the following deals.",
        f"Platform: {cluster.platform.name} ({cluster.platform.sector_id}, {cluster.platform.region_id})",
    ]
    for addon in cluster.addons:
        lines.append(
            f"- Add-on: {addon.name} ({addon.sector_id}, {addon.region_id})"
        )
    if cluster.snippets:
        lines.append("Snippets:")
        for snippet in cluster.snippets[:2]:
            lines.append(f"  • {snippet.text}")

    lines.append(f"Return {count} queries as a JSON list under the key 'queries'.")
    return "\n".join(lines)


def _default_data_path() -> Path:
    return settings.DATA_DIR / "raw"


def main():
    parser = argparse.ArgumentParser(
        description="Generate reverse-query training data for ranking."
    )
    parser.add_argument("--clusters", type=int, default=50, help="Number of clusters to sample.")
    parser.add_argument("--queries-per-cluster", type=int, default=3, help="Queries per cluster.")
    parser.add_argument("--negatives-per-query", type=int, default=2, help="Negative examples per query.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file for training data.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM calls and emit deterministic placeholder queries.",
    )
    args = parser.parse_args()

    llm_client = None if args.dry_run else LLMClient()
    records = generate_synthetic_training_data(
        num_clusters=args.clusters,
        queries_per_cluster=args.queries_per_cluster,
        negatives_per_query=args.negatives_per_query,
        seed=args.seed,
        llm_client=llm_client,
    )
    metadata = {
        "clusters": args.clusters,
        "queries_per_cluster": args.queries_per_cluster,
        "negatives_per_query": args.negatives_per_query,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "record_count": len(records),
    }
    save_training_data(records, args.output, metadata=metadata)
    logger.info("Training data saved to %s", args.output)


if __name__ == "__main__":
    main()
