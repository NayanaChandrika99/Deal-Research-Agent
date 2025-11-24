# ABOUTME: Training script for DealRanker using synthetic data.
# ABOUTME: Loads reverse-query dataset, builds features, and saves model.

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from ..data.graph_builder import DealGraph
from ..data.schemas import CandidateDeal, Deal, Snippet
from ..data.ingest import load_all, DealDataset
from ..config.settings import settings
from ..retrieval.features import (
    compute_graph_features,
    enhance_features_with_text,
    extract_query_context,
)
from ..ranking.features import candidate_to_features
from ..ranking.model import DealRanker

logger = logging.getLogger(__name__)


def load_training_records(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Training data must be a JSON list")
    return data


def build_candidate(
    deal: Deal,
    query: str,
    graph: DealGraph,
    dataset: DealDataset,
    similarity: float,
) -> CandidateDeal:
    snippets = [
        snippet
        for snippet in dataset.snippets
        if snippet.deal_id == deal.id
    ]
    all_deals = dataset.deals
    known_sectors = [sector.id for sector in dataset.sectors]
    known_regions = [region.id for region in dataset.regions]
    context = extract_query_context(query, known_sectors, known_regions)
    graph_features = compute_graph_features(
        deal_graph=graph,
        deal=deal,
        query_sectors=context.get("sectors"),
        query_regions=context.get("regions"),
        all_deals=all_deals,
    )
    enhanced = enhance_features_with_text(
        features=graph_features,
        deal=deal,
        query=query,
        text_similarity=similarity,
    )
    candidate = CandidateDeal(
        deal=deal,
        snippets=snippets,
        text_similarity=similarity,
        graph_features=enhanced,
    )
    return candidate


def lexical_similarity(query: str, deal: Deal) -> float:
    query_tokens = set(query.lower().split())
    description_tokens = set(deal.description.lower().split())
    if not query_tokens or not description_tokens:
        return 0.0
    overlap = query_tokens & description_tokens
    union = query_tokens | description_tokens
    return len(overlap) / len(union)


def train_ranker(
    records: List[Dict[str, object]],
    dataset: DealDataset,
) -> DealRanker:
    graph = DealGraph()
    graph.build_from_dataset(dataset)
    deal_lookup: Dict[str, Deal] = {deal.id: deal for deal in dataset.deals}

    feature_rows: List[np.ndarray] = []
    labels: List[float] = []
    for record in records:
        deal = deal_lookup.get(record["candidate_id"])
        if not deal:
            continue
        similarity = lexical_similarity(record["query"], deal)
        candidate = build_candidate(
            deal, record["query"], graph, dataset, similarity
        )
        feature_rows.append(candidate_to_features(candidate))
        labels.append(float(record["label"]))

    if not feature_rows:
        raise ValueError("No training samples were created.")

    X = np.vstack(feature_rows)
    y = np.array(labels, dtype=np.float32)

    unique, counts = np.unique(y, return_counts=True)
    if len(y) < 4 or counts.min() < 2:
        X_train, X_val, y_train, y_val = X, X, y, y
        stratified = False
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        stratified = True

    ranker = DealRanker()
    ranker.fit(X_train, y_train)

    val_scores = ranker.predict_scores(X_val)
    try:
        auc = roc_auc_score(y_val, val_scores)
    except ValueError:
        auc = float("nan")
    metric_label = "ROC-AUC" if stratified else "ROC-AUC (train-set proxy)"
    logger.info("%s: %.3f", metric_label, auc)

    return ranker


def main():
    parser = argparse.ArgumentParser(
        description="Train DealRanker using synthetic data."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=settings.DATA_DIR / "processed" / "ranking_training_data.json",
        help="Path to synthetic ranking dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.MODELS_DIR / "deal_ranker_v1.pkl",
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional limit on number of records to load.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    records = load_training_records(args.data)
    if args.max_records:
        records = records[: args.max_records]
    dataset = load_all(settings.DATA_DIR / "raw")
    ranker = train_ranker(records, dataset)
    ranker.save(args.output)
    logger.info("Model saved to %s", args.output)


if __name__ == "__main__":
    main()
