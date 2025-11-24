"""Tests for reverse-query training data generation."""

import json
import sys
from pathlib import Path

import pytest

from dealgraph.ranking import data_gen
from dealgraph.data.schemas import (
    Deal,
    Sector,
    Region,
    Event,
    EventType,
    Snippet,
    DealDataset,
)


class StubLLM:
    """Deterministic stub for the LLM client."""

    def __init__(self):
        self.calls = 0

    def complete_json(self, system_prompt: str, user_prompt: str, **_kwargs):
        self.calls += 1
        return {
            "queries": [
                f"Test query {self.calls}-A",
                f"Test query {self.calls}-B",
            ]
        }


def _build_dataset() -> DealDataset:
    sectors = [
        Sector(id="software", name="Software"),
        Sector(id="industrial", name="Industrial"),
    ]
    regions = [
        Region(id="us", name="United States"),
        Region(id="eu", name="Europe"),
    ]
    deals = [
        Deal(
            id="platform_alpha",
            name="Alpha Platform",
            sector_id="software",
            region_id="us",
            is_platform=True,
            status="current",
            description="Software roll-up platform",
        ),
        Deal(
            id="addon_alpha_1",
            name="Alpha Addon 1",
            sector_id="software",
            region_id="us",
            is_platform=False,
            status="current",
            description="Cybersecurity add-on",
        ),
        Deal(
            id="platform_beta",
            name="Beta Platform",
            sector_id="industrial",
            region_id="eu",
            is_platform=True,
            status="current",
            description="Industrial platform",
        ),
    ]
    events = [
        Event(
            id="evt_addon_alpha",
            type=EventType.ADDON,
            deal_id="platform_alpha",
            related_deal_id="addon_alpha_1",
        )
    ]
    snippets = [
        Snippet(
            id="snippet_1",
            deal_id="platform_alpha",
            source="case_study",
            text="Alpha Platform expanded via cybersecurity acquisitions.",
        )
    ]
    return DealDataset(
        sectors=sectors,
        regions=regions,
        deals=deals,
        events=events,
        snippets=snippets,
    )


def test_generate_synthetic_training_data_with_stub_llm():
    dataset = _build_dataset()
    stub_llm = StubLLM()

    records = data_gen.generate_synthetic_training_data(
        num_clusters=1,
        queries_per_cluster=2,
        negatives_per_query=1,
        seed=42,
        llm_client=stub_llm,
        dataset=dataset,
    )

    # Cluster contains platform_alpha + addon_alpha_1 = 2 positives
    assert len(records) == 2 * (2 + 1)  # queries * (positives + negatives)
    labels = {record["label"] for record in records}
    assert labels == {0, 1}
    positives = [r for r in records if r["label"] == 1]
    assert {r["candidate_id"] for r in positives} == {"platform_alpha", "addon_alpha_1"}
    assert all(r["cluster_id"].startswith("cluster_platform_alpha") for r in records)


def test_cli_dry_run_generates_file(tmp_path, monkeypatch):
    dataset = _build_dataset()
    monkeypatch.setattr(data_gen, "load_all", lambda _: dataset)

    output = tmp_path / "training.json"
    argv = [
        "data_gen",
        "--clusters",
        "1",
        "--queries-per-cluster",
        "1",
        "--negatives-per-query",
        "1",
        "--output",
        str(output),
        "--dry-run",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    data_gen.main()

    assert output.exists()
    records = json.loads(output.read_text())
    assert records
    assert all("query" in record for record in records)
