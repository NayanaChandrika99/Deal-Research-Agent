"""Unit tests for data schemas validation and constraints."""

import pytest
from typing import List

from dealgraph.data.schemas import (
    Sector, Region, Event, EventType, Snippet, Deal,
    CandidateDeal, RankedDeal, Precedent, DealReasoningOutput,
    DealDataset, DealList, SectorList
)


class TestEventType:
    """Test EventType enum values."""
    
    def test_event_type_values(self):
        """EventType should have correct string values."""
        assert EventType.ADDON == "addon"
        assert EventType.EXIT == "exit"


class TestSector:
    """Test Sector model validation and constraints."""
    
    def test_sector_creation(self):
        """Sector should be created with id and name."""
        sector = Sector(id="healthcare", name="Healthcare Services")
        
        assert sector.id == "healthcare"
        assert sector.name == "Healthcare Services"
    
    def test_sector_required_fields(self):
        """Sector should require id and name fields."""
        with pytest.raises(Exception):  # pydantic validation error
            Sector(id="test")
        
        with pytest.raises(Exception):
            Sector(name="test")


class TestRegion:
    """Test Region model validation and constraints."""
    
    def test_region_creation(self):
        """Region should be created with id and name."""
        region = Region(id="us", name="United States")
        
        assert region.id == "us"
        assert region.name == "United States"


class TestEvent:
    """Test Event model validation and constraints."""
    
    def test_event_creation_addon(self):
        """Event should be created for addon type."""
        event = Event(
            id="evt_001",
            type=EventType.ADDON,
            deal_id="deal_001",
            related_deal_id="deal_002"
        )
        
        assert event.id == "evt_001"
        assert event.type == EventType.ADDON
        assert event.deal_id == "deal_001"
        assert event.related_deal_id == "deal_002"
    
    def test_event_creation_exit(self):
        """Event should be created for exit type."""
        event = Event(
            id="evt_002",
            type=EventType.EXIT,
            deal_id="deal_003",
            date="2023-12-01"
        )
        
        assert event.type == EventType.EXIT
        assert event.date == "2023-12-01"
    
    def test_event_required_fields(self):
        """Event should require id, type, and deal_id."""
        with pytest.raises(Exception):
            Event(type=EventType.ADDON, deal_id="deal_001")
        
        with pytest.raises(Exception):
            Event(id="evt_001", deal_id="deal_001")


class TestSnippet:
    """Test Snippet model validation and constraints."""
    
    def test_snippet_creation(self):
        """Snippet should be created with id, deal_id, source, and text."""
        snippet = Snippet(
            id="snippet_001",
            deal_id="deal_001",
            source="news",
            text="Company announced strategic acquisition in Q3"
        )
        
        assert snippet.id == "snippet_001"
        assert snippet.deal_id == "deal_001"
        assert snippet.source == "news"
        assert "acquisition" in snippet.text


class TestDeal:
    """Test Deal model validation and constraints."""
    
    def test_deal_creation_minimal(self):
        """Deal should be created with minimal required fields."""
        deal = Deal(
            id="deal_001",
            name="MedTech Corp",
            sector_id="healthcare",
            region_id="us",
            is_platform=True,
            status="current",
            description="Medical device company"
        )
        
        assert deal.id == "deal_001"
        assert deal.name == "MedTech Corp"
        assert deal.sector_id == "healthcare"
        assert deal.is_platform is True
        assert deal.status == "current"
        assert deal.year_invested is None
        assert deal.year_exited is None
    
    def test_deal_creation_with_years(self):
        """Deal should be created with optional year fields."""
        deal = Deal(
            id="deal_002",
            name="Industrial Co",
            sector_id="industrial",
            region_id="us",
            is_platform=False,
            status="realized",
            description="Manufacturing company",
            year_invested=2020,
            year_exited=2023
        )
        
        assert deal.year_invested == 2020
        assert deal.year_exited == 2023
    
    def test_deal_required_fields(self):
        """Deal should require all core fields."""
        with pytest.raises(Exception):
            Deal(
                name="Test",
                sector_id="tech",
                region_id="us",
                is_platform=True,
                status="current",
                description="Test"
                # missing id
            )


class TestCandidateDeal:
    """Test CandidateDeal model validation and constraints."""
    
    def test_candidate_deal_creation(self):
        """CandidateDeal should combine Deal with similarity and features."""
        deal = Deal(
            id="deal_001",
            name="TechCorp",
            sector_id="software",
            region_id="us",
            is_platform=True,
            status="current",
            description="Software company"
        )
        
        snippets = [
            Snippet(id="snip_001", deal_id="deal_001", source="case_study", text="Growth story")
        ]
        
        candidate = CandidateDeal(
            deal=deal,
            snippets=snippets,
            text_similarity=0.85,
            graph_features={"sector_match": 1, "num_addons": 2}
        )
        
        assert candidate.deal == deal
        assert candidate.text_similarity == 0.85
        assert candidate.graph_features["sector_match"] == 1
        assert len(candidate.snippets) == 1


class TestRankedDeal:
    """Test RankedDeal model validation and constraints."""
    
    def test_ranked_deal_creation(self):
        """RankedDeal should combine CandidateDeal with score and rank."""
        deal = Deal(
            id="deal_001",
            name="TestCorp",
            sector_id="tech",
            region_id="us",
            is_platform=False,
            status="current",
            description="Test company"
        )
        
        candidate = CandidateDeal(
            deal=deal,
            snippets=[],
            text_similarity=0.75,
            graph_features={"sector_match": 1}
        )
        
        ranked = RankedDeal(
            candidate=candidate,
            score=0.82,
            rank=1
        )
        
        assert ranked.score == 0.82
        assert ranked.rank == 1
        assert ranked.candidate.deal.id == "deal_001"


class TestPrecedent:
    """Test Precedent model validation and constraints."""
    
    def test_precedent_creation(self):
        """Precedent should capture deal similarity reasoning."""
        precedent = Precedent(
            deal_id="deal_001",
            name="SimilarCorp",
            similarity_reason="Similar sector and growth pattern"
        )
        
        assert precedent.deal_id == "deal_001"
        assert "Similar" in precedent.name
        assert "similar" in precedent.similarity_reason.lower()


class TestDealReasoningOutput:
    """Test DealReasoningOutput model validation and constraints."""
    
    def test_reasoning_output_creation(self):
        """DealReasoningOutput should contain structured analysis."""
        precedents = [
            Precedent(
                deal_id="deal_001",
                name="Precedent Corp",
                similarity_reason="Strong sector match"
            )
        ]
        
        output = DealReasoningOutput(
            precedents=precedents,
            playbook_levers=["Operational improvements", "Add-on acquisitions"],
            risk_themes=["Market concentration", "Integration risk"],
            narrative_summary="This deal represents a strategic opportunity..."
        )
        
        assert len(output.precedents) == 1
        assert len(output.playbook_levers) == 2
        assert len(output.risk_themes) == 2
        assert "strategic opportunity" in output.narrative_summary


class TestDealDataset:
    """Test DealDataset model validation and constraints."""
    
    def test_dataset_creation(self):
        """DealDataset should bundle all data types."""
        sectors = [Sector(id="tech", name="Technology")]
        regions = [Region(id="us", name="United States")]
        deals = [Deal(
            id="deal_001",
            name="TestCorp",
            sector_id="tech",
            region_id="us",
            is_platform=True,
            status="current",
            description="Test"
        )]
        events = []
        snippets = []
        
        dataset = DealDataset(
            sectors=sectors,
            regions=regions,
            deals=deals,
            events=events,
            snippets=snippets
        )
        
        assert len(dataset.sectors) == 1
        assert len(dataset.deals) == 1
        assert dataset.deals[0].id == "deal_001"


class TestTypeAliases:
    """Test type aliases work correctly."""
    
    def test_deal_list_alias(self):
        """DealList should work as type alias for List[Deal]."""
        assert DealList == List[Deal]
        assert SectorList == List[Sector]
        # etc. - these are type aliases, not runtime objects


class TestIntegration:
    """Test integration between schemas."""
    
    def test_complete_workflow(self):
        """Test creating a complete deal workflow with all schemas."""
        # Create basic entities
        sector = Sector(id="software", name="Software")
        region = Region(id="us", name="United States")
        
        # Create deal
        deal = Deal(
            id="platform_001",
            name="Platform Corp",
            sector_id="software",
            region_id="us",
            is_platform=True,
            status="current",
            description="Enterprise software platform",
            year_invested=2021
        )
        
        # Create events
        addon_event = Event(
            id="evt_001",
            type=EventType.ADDON,
            deal_id="platform_001",
            related_deal_id="addon_001",
            date="2022-06-01"
        )
        
        # Create snippets
        snippets = [
            Snippet(
                id="snip_001",
                deal_id="platform_001",
                source="news",
                text="Platform Corp announces strategic acquisition"
            ),
            Snippet(
                id="snip_002",
                deal_id="platform_001",
                source="case_study",
                text="Successful integration of acquired technologies"
            )
        ]
        
        # Create candidate deal
        candidate = CandidateDeal(
            deal=deal,
            snippets=snippets,
            text_similarity=0.89,
            graph_features={
                "sector_match": 1,
                "region_match": 1,
                "num_addons": 1,
                "has_exit": 0,
                "degree": 3
            }
        )
        
        # Create ranked deal
        ranked = RankedDeal(
            candidate=candidate,
            score=0.92,
            rank=1
        )
        
        # Create reasoning output
        reasoning = DealReasoningOutput(
            precedents=[
                Precedent(
                    deal_id="platform_001",
                    name="Platform Corp",
                    similarity_reason="Enterprise software with successful add-on strategy"
                )
            ],
            playbook_levers=[
                "Technology platform expansion",
                "Strategic add-on acquisitions",
                "Customer base expansion"
            ],
            risk_themes=[
                "Integration complexity",
                "Market competition"
            ],
            narrative_summary="This represents a compelling platform investment..."
        )
        
        # Verify the complete workflow
        assert ranked.rank == 1
        assert candidate.graph_features["num_addons"] == 1
        assert len(reasoning.playbook_levers) == 3
        assert "platform investment" in reasoning.narrative_summary
