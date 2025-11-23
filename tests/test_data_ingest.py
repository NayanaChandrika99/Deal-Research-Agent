"""Unit tests for data ingestion module."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from dealgraph.data.ingest import (
    load_sectors, load_regions, load_deals, load_events, load_snippets,
    load_all, validate_referential_integrity, get_data_statistics,
    DataIngestionError, ValidationError
)
from dealgraph.data.schemas import EventType, Sector, Region, Deal, Event, Snippet


class TestLoadSectors:
    """Test sector loading functionality."""
    
    def test_load_sectors_success(self, tmp_path):
        """Test successful loading of valid sectors."""
        sectors_data = [
            {"id": "tech", "name": "Technology"},
            {"id": "healthcare", "name": "Healthcare"}
        ]
        
        sectors_file = tmp_path / "sectors.json"
        with open(sectors_file, 'w') as f:
            json.dump(sectors_data, f)
        
        sectors = load_sectors(sectors_file)
        
        assert len(sectors) == 2
        assert sectors[0].id == "tech"
        assert sectors[0].name == "Technology"
        assert sectors[1].id == "healthcare"
        assert sectors[1].name == "Healthcare"
    
    def test_load_sectors_file_not_found(self, tmp_path):
        """Test error when sector file doesn't exist."""
        with pytest.raises(DataIngestionError, match="Failed to load sectors"):
            load_sectors(tmp_path / "nonexistent.json")
    
    def test_load_sectors_invalid_json(self, tmp_path):
        """Test error when sector file contains invalid JSON."""
        sectors_file = tmp_path / "sectors.json"
        with open(sectors_file, 'w') as f:
            f.write("invalid json {")
        
        with pytest.raises(DataIngestionError):
            load_sectors(sectors_file)
    
    def test_load_sectors_invalid_data(self, tmp_path):
        """Test error when sector data is invalid."""
        sectors_data = [{"id": "tech"}]  # missing 'name'
        
        sectors_file = tmp_path / "sectors.json"
        with open(sectors_file, 'w') as f:
            json.dump(sectors_data, f)
        
        with pytest.raises(ValidationError, match="Invalid sector data"):
            load_sectors(sectors_file)


class TestLoadRegions:
    """Test region loading functionality."""
    
    def test_load_regions_success(self, tmp_path):
        """Test successful loading of valid regions."""
        regions_data = [
            {"id": "us", "name": "United States"},
            {"id": "eu", "name": "Europe"}
        ]
        
        regions_file = tmp_path / "regions.json"
        with open(regions_file, 'w') as f:
            json.dump(regions_data, f)
        
        regions = load_regions(regions_file)
        
        assert len(regions) == 2
        assert regions[0].id == "us"
        assert regions[1].id == "eu"


class TestLoadDeals:
    """Test deal loading functionality."""
    
    def test_load_deals_success(self, tmp_path):
        """Test successful loading of valid deals."""
        deals_data = [
            {
                "id": "deal_001",
                "name": "Test Corp",
                "sector_id": "tech",
                "region_id": "us",
                "is_platform": True,
                "status": "current",
                "description": "Test company"
            }
        ]
        
        deals_file = tmp_path / "deals.json"
        with open(deals_file, 'w') as f:
            json.dump(deals_data, f)
        
        deals = load_deals(deals_file)
        
        assert len(deals) == 1
        assert deals[0].id == "deal_001"
        assert deals[0].is_platform is True
        assert deals[0].status == "current"
    
    def test_load_deals_with_optional_fields(self, tmp_path):
        """Test loading deals with optional fields."""
        deals_data = [
            {
                "id": "deal_002",
                "name": "Exit Corp",
                "sector_id": "tech",
                "region_id": "us",
                "is_platform": False,
                "status": "realized",
                "description": "Company with exit",
                "year_invested": 2020,
                "year_exited": 2023
            }
        ]
        
        deals_file = tmp_path / "deals.json"
        with open(deals_file, 'w') as f:
            json.dump(deals_data, f)
        
        deals = load_deals(deals_file)
        
        assert deals[0].year_invested == 2020
        assert deals[0].year_exited == 2023


class TestLoadEvents:
    """Test event loading functionality."""
    
    def test_load_events_success(self, tmp_path):
        """Test successful loading of valid events."""
        events_data = [
            {
                "id": "evt_001",
                "type": "addon",
                "deal_id": "deal_001",
                "related_deal_id": "deal_002"
            }
        ]
        
        events_file = tmp_path / "events.json"
        with open(events_file, 'w') as f:
            json.dump(events_data, f)
        
        events = load_events(events_file)
        
        assert len(events) == 1
        assert events[0].id == "evt_001"
        assert events[0].type == EventType.ADDON
        assert events[0].deal_id == "deal_001"
    
    def test_load_events_exit_type(self, tmp_path):
        """Test loading exit events."""
        events_data = [
            {
                "id": "evt_002",
                "type": "exit",
                "deal_id": "deal_003",
                "date": "2023-12-01"
            }
        ]
        
        events_file = tmp_path / "events.json"
        with open(events_file, 'w') as f:
            json.dump(events_data, f)
        
        events = load_events(events_file)
        
        assert len(events) == 1
        assert events[0].type == EventType.EXIT
        assert events[0].date == "2023-12-01"


class TestLoadSnippets:
    """Test snippet loading functionality."""
    
    def test_load_snippets_success(self, tmp_path):
        """Test successful loading of valid snippets."""
        snippets_data = [
            {
                "id": "snippet_001",
                "deal_id": "deal_001",
                "source": "news",
                "text": "Company announced acquisition"
            }
        ]
        
        snippets_file = tmp_path / "snippets.json"
        with open(snippets_file, 'w') as f:
            json.dump(snippets_data, f)
        
        snippets = load_snippets(snippets_file)
        
        assert len(snippets) == 1
        assert snippets[0].id == "snippet_001"
        assert snippets[0].deal_id == "deal_001"
        assert snippets[0].source == "news"
        assert "acquisition" in snippets[0].text


class TestValidation:
    """Test data validation functionality."""
    
    def test_validate_referential_integrity_success(self):
        """Test successful validation of referentially correct data."""
        sectors = [Sector(id="tech", name="Technology")]
        regions = [Region(id="us", name="United States")]
        deals = [Deal(
            id="deal_001",
            name="Test Corp",
            sector_id="tech",
            region_id="us",
            is_platform=True,
            status="current",
            description="Test"
        )]
        events = [Event(
            id="evt_001",
            type=EventType.ADDON,
            deal_id="deal_001"
        )]
        snippets = [Snippet(
            id="snippet_001",
            deal_id="deal_001",
            source="news",
            text="Test"
        )]
        
        # Should not raise any exceptions
        validate_referential_integrity(sectors, regions, deals, events, snippets)
    
    def test_validate_invalid_sector_reference(self):
        """Test validation fails when deal references unknown sector."""
        sectors = [Sector(id="tech", name="Technology")]
        regions = [Region(id="us", name="United States")]
        deals = [Deal(
            id="deal_001",
            name="Test Corp",
            sector_id="invalid_sector",  # This doesn't exist
            region_id="us",
            is_platform=True,
            status="current",
            description="Test"
        )]
        events = []
        snippets = []
        
        with pytest.raises(ValidationError, match="references unknown sector_id"):
            validate_referential_integrity(sectors, regions, deals, events, snippets)
    
    def test_validate_invalid_region_reference(self):
        """Test validation fails when deal references unknown region."""
        sectors = [Sector(id="tech", name="Technology")]
        regions = [Region(id="us", name="United States")]
        deals = [Deal(
            id="deal_001",
            name="Test Corp",
            sector_id="tech",
            region_id="invalid_region",  # This doesn't exist
            is_platform=True,
            status="current",
            description="Test"
        )]
        events = []
        snippets = []
        
        with pytest.raises(ValidationError, match="references unknown region_id"):
            validate_referential_integrity(sectors, regions, deals, events, snippets)
    
    def test_validate_invalid_deal_reference_in_event(self):
        """Test validation fails when event references unknown deal."""
        sectors = [Sector(id="tech", name="Technology")]
        regions = [Region(id="us", name="United States")]
        deals = [Deal(
            id="deal_001",
            name="Test Corp",
            sector_id="tech",
            region_id="us",
            is_platform=True,
            status="current",
            description="Test"
        )]
        events = [Event(
            id="evt_001",
            type=EventType.ADDON,
            deal_id="invalid_deal"  # This doesn't exist
        )]
        snippets = []
        
        with pytest.raises(ValidationError, match="references unknown deal_id"):
            validate_referential_integrity(sectors, regions, deals, events, snippets)
    
    def test_validate_invalid_related_deal_reference(self):
        """Test validation fails when event references unknown related deal."""
        sectors = [Sector(id="tech", name="Technology")]
        regions = [Region(id="us", name="United States")]
        deals = [Deal(
            id="deal_001",
            name="Test Corp",
            sector_id="tech",
            region_id="us",
            is_platform=True,
            status="current",
            description="Test"
        )]
        events = [Event(
            id="evt_001",
            type=EventType.ADDON,
            deal_id="deal_001",
            related_deal_id="invalid_deal"  # This doesn't exist
        )]
        snippets = []
        
        with pytest.raises(ValidationError, match="references unknown related_deal_id"):
            validate_referential_integrity(sectors, regions, deals, events, snippets)
    
    def test_validate_invalid_snippet_reference(self):
        """Test validation fails when snippet references unknown deal."""
        sectors = [Sector(id="tech", name="Technology")]
        regions = [Region(id="us", name="United States")]
        deals = [Deal(
            id="deal_001",
            name="Test Corp",
            sector_id="tech",
            region_id="us",
            is_platform=True,
            status="current",
            description="Test"
        )]
        events = []
        snippets = [Snippet(
            id="snippet_001",
            deal_id="invalid_deal",  # This doesn't exist
            source="news",
            text="Test"
        )]
        
        with pytest.raises(ValidationError, match="references unknown deal_id"):
            validate_referential_integrity(sectors, regions, deals, events, snippets)


class TestLoadAll:
    """Test complete data loading functionality."""
    
    def test_load_all_success(self, tmp_path):
        """Test loading all data types successfully."""
        # Create all required data files
        sectors_data = [{"id": "tech", "name": "Technology"}]
        regions_data = [{"id": "us", "name": "United States"}]
        deals_data = [{
            "id": "deal_001",
            "name": "Test Corp",
            "sector_id": "tech",
            "region_id": "us",
            "is_platform": True,
            "status": "current",
            "description": "Test company"
        }]
        events_data = []
        snippets_data = []
        
        # Write files
        (tmp_path / "sectors.json").write_text(json.dumps(sectors_data))
        (tmp_path / "regions.json").write_text(json.dumps(regions_data))
        (tmp_path / "deals.json").write_text(json.dumps(deals_data))
        (tmp_path / "events.json").write_text(json.dumps(events_data))
        (tmp_path / "snippets.json").write_text(json.dumps(snippets_data))
        
        # Load all data
        dataset = load_all(tmp_path)
        
        assert len(dataset.sectors) == 1
        assert len(dataset.regions) == 1
        assert len(dataset.deals) == 1
        assert len(dataset.events) == 0
        assert len(dataset.snippets) == 0
    
    def test_load_all_missing_file(self, tmp_path):
        """Test error when required file is missing."""
        sectors_data = [{"id": "tech", "name": "Technology"}]
        (tmp_path / "sectors.json").write_text(json.dumps(sectors_data))
        # Missing other required files
        
        with pytest.raises(DataIngestionError, match="Failed to load regions"):
            load_all(tmp_path)
    
    def test_load_all_with_validation_error(self, tmp_path):
        """Test error when data validation fails."""
        # Create inconsistent data
        sectors_data = [{"id": "tech", "name": "Technology"}]
        regions_data = [{"id": "us", "name": "United States"}]
        deals_data = [{
            "id": "deal_001",
            "name": "Test Corp",
            "sector_id": "invalid_sector",  # This doesn't exist in sectors
            "region_id": "us",
            "is_platform": True,
            "status": "current",
            "description": "Test company"
        }]
        events_data = []
        snippets_data = []
        
        (tmp_path / "sectors.json").write_text(json.dumps(sectors_data))
        (tmp_path / "regions.json").write_text(json.dumps(regions_data))
        (tmp_path / "deals.json").write_text(json.dumps(deals_data))
        (tmp_path / "events.json").write_text(json.dumps(events_data))
        (tmp_path / "snippets.json").write_text(json.dumps(snippets_data))
        
        with pytest.raises(ValidationError):
            load_all(tmp_path)


class TestDataStatistics:
    """Test data statistics functionality."""
    
    def test_get_data_statistics(self):
        """Test computing data statistics."""
        sectors = [Sector(id="tech", name="Technology")]
        regions = [Region(id="us", name="United States")]
        deals = [
            Deal(
                id="platform_001",
                name="Platform Corp",
                sector_id="tech",
                region_id="us",
                is_platform=True,
                status="current",
                description="Platform"
            ),
            Deal(
                id="addon_001",
                name="Addon Corp",
                sector_id="tech",
                region_id="us",
                is_platform=False,
                status="realized",
                description="Addon"
            )
        ]
        events = [
            Event(id="evt_001", type=EventType.ADDON, deal_id="platform_001"),
            Event(id="evt_002", type=EventType.EXIT, deal_id="addon_001")
        ]
        snippets = [Snippet(id="snippet_001", deal_id="platform_001", source="news", text="Text")]
        
        from dealgraph.data.schemas import DealDataset
        dataset = DealDataset(
            sectors=sectors,
            regions=regions,
            deals=deals,
            events=events,
            snippets=snippets
        )
        
        stats = get_data_statistics(dataset)
        
        assert stats["total_deals"] == 2
        assert stats["platform_deals"] == 1
        assert stats["addon_deals"] == 1
        assert stats["current_deals"] == 1
        assert stats["realized_deals"] == 1
        assert stats["total_events"] == 2
        assert stats["addon_events"] == 1
        assert stats["exit_events"] == 1
        assert stats["total_snippets"] == 1
        assert stats["sectors"]["tech"] == 2
        assert stats["regions"]["us"] == 2


class TestIntegration:
    """Integration tests using real sample data."""
    
    def test_load_real_sample_data(self):
        """Test loading the actual sample data provided with the project."""
        dataset = load_all("data/raw")
        
        # Verify data was loaded
        assert len(dataset.sectors) == 5
        assert len(dataset.regions) == 4
        assert len(dataset.deals) == 12
        assert len(dataset.events) == 8
        assert len(dataset.snippets) == 12
        
        # Verify some specific deals
        deal_ids = {deal.id for deal in dataset.deals}
        assert "platform_tech_001" in deal_ids
        assert "addon_tech_001" in deal_ids
        assert "realized_001" in deal_ids
        
        # Verify platform vs addon classification
        platforms = [d for d in dataset.deals if d.is_platform]
        addons = [d for d in dataset.deals if not d.is_platform]
        
        assert len(platforms) == 7
        assert len(addons) == 5
        
        # Verify event relationships
        addon_events = [e for e in dataset.events if e.type == EventType.ADDON]
        assert len(addon_events) == 6
        
        # Verify statistics
        stats = get_data_statistics(dataset)
        assert stats["platform_deals"] == 7
        assert stats["addon_deals"] == 5
        assert stats["current_deals"] == 10
        assert stats["realized_deals"] == 2
