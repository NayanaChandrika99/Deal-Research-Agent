"""Tests for ranking training pipeline."""

import json
import sys
from pathlib import Path

import pytest

from dealgraph.data.schemas import (
    Deal,
    Sector,
    Region,
    Event,
    EventType,
    Snippet,
    DealDataset,
)
from dealgraph.data.graph_builder import DealGraph
from dealgraph.ranking import train as train_module


def _dataset() -> DealDataset:
    sectors = [Sector(id="software", name="Software")]
    regions = [Region(id="us", name="United States")]
    deals = [
        Deal(
            id="deal_pos",
            name="Positive Deal",
            sector_id="software",
            region_id="us",
            is_platform=True,
            status="current",
            description="Software rollout",
        ),
        Deal(
            id="deal_neg",
            name="Negative Deal",
            sector_id="software",
            region_id="us",
            is_platform=False,
            status="current",
            description="Hardware focus",
        ),
    ]
    events = [
        Event(
            id="evt_add",
            type=EventType.ADDON,
            deal_id="deal_pos",
            related_deal_id="deal_neg",
        )
    ]
    snippets = [
        Snippet(
            id="snip1",
            deal_id="deal_pos",
            source="news",
            text="Rollout snippet",
        )
    ]
    return DealDataset(
        sectors=sectors,
        regions=regions,
        deals=deals,
        events=events,
        snippets=snippets,
    )


def test_train_ranker_builds_model(monkeypatch):
    dataset = _dataset()
    records = [
        {"query": "software rollout", "candidate_id": "deal_pos", "label": 1, "cluster_id": "c1"},
        {"query": "software rollout", "candidate_id": "deal_neg", "label": 0, "cluster_id": "c1"},
    ]
    ranker = train_module.train_ranker(records, dataset)
    graph = DealGraph()
    graph.build_from_dataset(dataset)
    pos_candidate = train_module.build_candidate(
        dataset.deals[0], "software rollout", graph, dataset, similarity=1.0
    )
    neg_candidate = train_module.build_candidate(
        dataset.deals[1], "software rollout", graph, dataset, similarity=0.0
    )
    ranked = ranker.rank([neg_candidate, pos_candidate])
    assert ranked[0].candidate.deal.id == "deal_pos"


def test_cli_runs_and_saves_model(tmp_path, monkeypatch):
    dataset = _dataset()
    train_data = [
        {"query": "software rollout", "candidate_id": "deal_pos", "label": 1, "cluster_id": "c1"},
        {"query": "software rollout", "candidate_id": "deal_neg", "label": 0, "cluster_id": "c1"},
    ]
    data_file = tmp_path / "training.json"
    data_file.write_text(json.dumps(train_data))

    monkeypatch.setattr(train_module, "load_all", lambda _: dataset)

    output_file = tmp_path / "model.pkl"
    argv = [
        "train",
        "--data",
        str(data_file),
        "--output",
        str(output_file),
        "--max-records",
        "2",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    train_module.main()
    assert output_file.exists()
