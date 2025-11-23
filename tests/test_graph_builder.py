"""Unit tests for graph builder functionality."""

import pytest
import networkx as nx

from dealgraph.data.graph_builder import DealGraph, EdgeType
from dealgraph.data.schemas import (
    EventType, Deal, Sector, Region, Event, Snippet, DealDataset
)


class TestDealGraph:
    """Test DealGraph class functionality."""
    
    def test_deal_graph_initialization(self):
        """Test DealGraph initializes with empty MultiDiGraph."""
        graph = DealGraph()
        
        assert isinstance(graph.graph, nx.MultiDiGraph)
        assert graph.graph.number_of_nodes() == 0
        assert graph.graph.number_of_edges() == 0
        assert len(graph._node_type_mapping) == 0
    
    def test_add_sectors(self):
        """Test adding sectors to the graph."""
        graph = DealGraph()
        
        sectors = [
            Sector(id="tech", name="Technology"),
            Sector(id="healthcare", name="Healthcare")
        ]
        
        graph.add_sectors(sectors)
        
        assert graph.graph.number_of_nodes() == 2
        assert "tech" in graph.graph.nodes()
        assert "healthcare" in graph.graph.nodes()
        
        # Check node attributes
        assert graph.graph.nodes["tech"]["type"] == "sector"
        assert graph.graph.nodes["tech"]["name"] == "Technology"
        assert graph._node_type_mapping["tech"] == "sector"
    
    def test_add_regions(self):
        """Test adding regions to the graph."""
        graph = DealGraph()
        
        regions = [
            Region(id="us", name="United States"),
            Region(id="eu", name="Europe")
        ]
        
        graph.add_regions(regions)
        
        assert graph.graph.number_of_nodes() == 2
        assert graph.graph.nodes["us"]["name"] == "United States"
        assert graph.graph.nodes["eu"]["name"] == "Europe"
    
    def test_add_deals(self):
        """Test adding deals to the graph."""
        graph = DealGraph()
        
        deals = [
            Deal(
                id="deal_001",
                name="Test Corp",
                sector_id="tech",
                region_id="us",
                is_platform=True,
                status="current",
                description="Test company"
            ),
            Deal(
                id="deal_002",
                name="Health Corp",
                sector_id="healthcare",
                region_id="us",
                is_platform=False,
                status="current",
                description="Health company"
            )
        ]
        
        graph.add_deals(deals)
        
        assert graph.graph.number_of_nodes() == 2
        
        # Check deal attributes
        deal_node = graph.graph.nodes["deal_001"]
        assert deal_node["type"] == "deal"
        assert deal_node["name"] == "Test Corp"
        assert deal_node["sector_id"] == "tech"
        assert deal_node["is_platform"] is True
        assert deal_node["status"] == "current"
        assert deal_node["year_invested"] is None  # Optional field not set
    
    def test_add_events_addon(self):
        """Test adding addon events to the graph."""
        graph = DealGraph()
        
        # First add deals to reference
        deals = [
            Deal(id="platform_001", name="Platform", sector_id="tech", region_id="us", 
                 is_platform=True, status="current", description="Platform"),
            Deal(id="addon_001", name="Addon", sector_id="tech", region_id="us", 
                 is_platform=False, status="current", description="Addon")
        ]
        graph.add_deals(deals)
        
        # Add addon event
        event = Event(
            id="evt_001",
            type=EventType.ADDON,
            deal_id="platform_001",
            related_deal_id="addon_001",
            date="2023-01-01"
        )
        
        graph.add_events([event])
        
        # Check event node
        assert graph.graph.has_node("evt_001")
        event_node = graph.graph.nodes["evt_001"]
        assert event_node["type"] == "event"
        assert event_node["event_type"] == "addon"
        
        # Check ADDON_TO edge
        assert graph.has_edge("platform_001", "addon_001")
        edge_data = graph.graph["platform_001"]["addon_001"][0]  # First edge
        assert edge_data["relation"] == EdgeType.ADDON_TO
        assert edge_data["event_id"] == "evt_001"
    
    def test_add_events_exit(self):
        """Test adding exit events to the graph."""
        graph = DealGraph()
        
        # Add deal first
        deals = [Deal(
            id="deal_001", name="Test Corp", sector_id="tech", region_id="us",
            is_platform=True, status="current", description="Test"
        )]
        graph.add_deals(deals)
        
        # Add exit event
        event = Event(
            id="evt_002",
            type=EventType.EXIT,
            deal_id="deal_001",
            date="2023-12-01",
            description="IPO completed"
        )
        
        graph.add_events([event])
        
        # Check EXITED_VIA edge
        assert graph.has_edge("deal_001", "evt_002")
        edge_data = graph.graph["deal_001"]["evt_002"][0]
        assert edge_data["relation"] == EdgeType.EXITED_VIA
        assert edge_data["date"] == "2023-12-01"
    
    def test_add_snippets(self):
        """Test adding snippets to the graph."""
        graph = DealGraph()
        
        # Add deal first
        deals = [Deal(
            id="deal_001", name="Test Corp", sector_id="tech", region_id="us",
            is_platform=True, status="current", description="Test"
        )]
        graph.add_deals(deals)
        
        # Add snippets
        snippets = [
            Snippet(
                id="snippet_001",
                deal_id="deal_001",
                source="news",
                text="Company announced acquisition"
            ),
            Snippet(
                id="snippet_002", 
                deal_id="deal_001",
                source="case_study",
                text="Success story of growth"
            )
        ]
        
        graph.add_snippets(snippets)
        
        # Check snippet nodes
        assert graph.graph.has_node("snippet_001")
        assert graph.graph.has_node("snippet_002")
        
        # Check DESCRIBED_IN edges
        assert graph.has_edge("deal_001", "snippet_001")
        assert graph.has_edge("deal_001", "snippet_002")
        
        # Verify edge data
        edge_data = graph.graph["deal_001"]["snippet_001"][0]
        assert edge_data["relation"] == EdgeType.DESCRIBED_IN
        assert edge_data["source"] == "news"
    
    def test_build_from_dataset(self):
        """Test building complete graph from DealDataset."""
        from dealgraph.data.schemas import DealDataset
        
        graph = DealGraph()
        
        # Create sample dataset
        dataset = DealDataset(
            sectors=[Sector(id="tech", name="Technology")],
            regions=[Region(id="us", name="United States")],
            deals=[Deal(
                id="deal_001", name="Test Corp", sector_id="tech", region_id="us",
                is_platform=True, status="current", description="Test"
            )],
            events=[],
            snippets=[Snippet(
                id="snippet_001", deal_id="deal_001", source="news", text="Test text"
            )]
        )
        
        graph.build_from_dataset(dataset)
        
        # Verify all nodes are added
        assert graph.graph.number_of_nodes() == 4  # sector + region + deal + snippet
        assert "tech" in graph.graph.nodes()
        assert "us" in graph.graph.nodes()
        assert "deal_001" in graph.graph.nodes()
        assert "snippet_001" in graph.graph.nodes()
        
        # Verify taxonomy edges
        assert graph.has_edge("deal_001", "tech")  # IN_SECTOR
        assert graph.has_edge("deal_001", "us")   # IN_REGION
        assert graph.has_edge("deal_001", "snippet_001")  # DESCRIBED_IN
    
    def test_get_deal_neighbors(self):
        """Test getting neighbors of a deal."""
        graph = DealGraph()
        
        # Build sample graph
        dataset = DealDataset(
            sectors=[Sector(id="tech", name="Technology")],
            regions=[Region(id="us", name="United States")],
            deals=[
                Deal(id="platform_001", name="Platform", sector_id="tech", region_id="us",
                     is_platform=True, status="current", description="Platform"),
                Deal(id="addon_001", name="Addon", sector_id="tech", region_id="us", 
                     is_platform=False, status="current", description="Addon")
            ],
            events=[Event(id="evt_001", type=EventType.ADDON, deal_id="platform_001", 
                         related_deal_id="addon_001", date="2023-01-01")],
            snippets=[Snippet(id="snippet_001", deal_id="platform_001", source="news", text="Text")]
        )
        
        graph.build_from_dataset(dataset)
        
        # Test neighbors
        neighbors = graph.get_deal_neighbors("platform_001")
        
        # Verify neighbor categories
        assert len(neighbors["sectors"]) == 1
        assert neighbors["sectors"][0]["id"] == "tech"
        
        assert len(neighbors["regions"]) == 1
        assert neighbors["regions"][0]["id"] == "us"
        
        assert len(neighbors["events"]) == 0  # No exit events
        
        assert len(neighbors["snippets"]) == 1
        assert neighbors["snippets"][0]["id"] == "snippet_001"
        
        assert len(neighbors["related_deals"]) == 1
        assert neighbors["related_deals"][0]["id"] == "addon_001"
        
        assert len(neighbors["addons"]) == 1
        assert neighbors["addons"][0]["id"] == "addon_001"
        
        assert len(neighbors["platforms"]) == 0  # Platform has no platforms
    
    def test_get_deal_neighbors_nonexistent_deal(self):
        """Test error when getting neighbors for nonexistent deal."""
        graph = DealGraph()
        
        with pytest.raises(ValueError, match="Deal 'nonexistent' not found"):
            graph.get_deal_neighbors("nonexistent")
    
    def test_get_deal_graph_metrics(self):
        """Test getting graph metrics."""
        graph = DealGraph()
        
        # Build sample graph
        dataset = DealDataset(
            sectors=[Sector(id="tech", name="Technology")],
            regions=[Region(id="us", name="United States")],
            deals=[Deal(id="deal_001", name="Test", sector_id="tech", region_id="us",
                       is_platform=True, status="current", description="Test")],
            events=[],
            snippets=[Snippet(id="snippet_001", deal_id="deal_001", source="news", text="Text")]
        )
        
        graph.build_from_dataset(dataset)
        
        metrics = graph.get_deal_graph_metrics()
        
        # Verify metrics
        assert metrics["total_nodes"] == 4  # sector + region + deal + snippet
        assert metrics["total_edges"] == 3  # IN_SECTOR + IN_REGION + DESCRIBED_IN
        assert "deal" in metrics["node_counts"]
        assert "sector" in metrics["node_counts"]
        assert "region" in metrics["node_counts"]
        assert "snippet" in metrics["node_counts"]
        assert "IN_SECTOR" in metrics["edge_counts"]
        assert "IN_REGION" in metrics["edge_counts"]
        assert "DESCRIBED_IN" in metrics["edge_counts"]
        assert isinstance(metrics["is_connected"], bool)
        assert isinstance(metrics["density"], float)
        assert isinstance(metrics["avg_degree"], float)
    
    def test_find_deals_by_criteria(self):
        """Test finding deals by specific criteria."""
        graph = DealGraph()
        
        # Build sample graph
        dataset = DealDataset(
            sectors=[Sector(id="tech", name="Technology"), Sector(id="healthcare", name="Healthcare")],
            regions=[Region(id="us", name="United States")],
            deals=[
                Deal(id="platform_tech", name="Tech Platform", sector_id="tech", region_id="us",
                     is_platform=True, status="current", description="Tech"),
                Deal(id="addon_tech", name="Tech Addon", sector_id="tech", region_id="us", 
                     is_platform=False, status="current", description="Tech Addon"),
                Deal(id="platform_health", name="Health Platform", sector_id="healthcare", region_id="us",
                     is_platform=True, status="current", description="Health"),
                Deal(id="realized_tech", name="Realized Tech", sector_id="tech", region_id="us",
                     is_platform=True, status="realized", description="Realized")
            ],
            events=[],
            snippets=[]
        )
        
        graph.build_from_dataset(dataset)
        
        # Test filtering by sector
        tech_deals = graph.find_deals_by_criteria(sector_id="tech")
        assert len(tech_deals) == 3
        
        # Test filtering by platform status
        platforms = graph.find_deals_by_criteria(is_platform=True)
        assert len(platforms) == 3
        
        # Test filtering by status
        current_deals = graph.find_deals_by_criteria(status="current")
        assert len(current_deals) == 3
        
        # Test filtering by region
        us_deals = graph.find_deals_by_criteria(region_id="us")
        assert len(us_deals) == 4
        
        # Test combined filtering
        tech_platforms = graph.find_deals_by_criteria(sector_id="tech", is_platform=True)
        assert len(tech_platforms) == 2
        assert all(d["sector_id"] == "tech" for d in tech_platforms)
        assert all(d["is_platform"] for d in tech_platforms)
    
    def test_get_sectors_by_deal(self):
        """Test getting sectors for a specific deal."""
        graph = DealGraph()
        
        dataset = DealDataset(
            sectors=[Sector(id="tech", name="Technology")],
            regions=[Region(id="us", name="United States")],
            deals=[Deal(id="deal_001", name="Test", sector_id="tech", region_id="us",
                       is_platform=True, status="current", description="Test")],
            events=[],
            snippets=[]
        )
        
        graph.build_from_dataset(dataset)
        
        sectors = graph.get_sectors_by_deal("deal_001")
        
        assert len(sectors) == 1
        assert sectors[0]["id"] == "tech"
        assert sectors[0]["name"] == "Technology"
    
    def test_get_addon_relationships(self):
        """Test getting addon relationships for a deal."""
        graph = DealGraph()
        
        dataset = DealDataset(
            sectors=[Sector(id="tech", name="Technology")],
            regions=[Region(id="us", name="United States")],
            deals=[
                Deal(id="platform", name="Platform", sector_id="tech", region_id="us",
                     is_platform=True, status="current", description="Platform"),
                Deal(id="addon1", name="Addon1", sector_id="tech", region_id="us", 
                     is_platform=False, status="current", description="Addon1"),
                Deal(id="addon2", name="Addon2", sector_id="tech", region_id="us", 
                     is_platform=False, status="current", description="Addon2")
            ],
            events=[
                Event(id="evt1", type=EventType.ADDON, deal_id="platform", related_deal_id="addon1"),
                Event(id="evt2", type=EventType.ADDON, deal_id="platform", related_deal_id="addon2")
            ],
            snippets=[]
        )
        
        graph.build_from_dataset(dataset)
        
        # Test addon relationships for platform
        addons = graph.get_addon_relationships("platform")
        assert len(addons) == 2
        addon_ids = {a["id"] for a in addons}
        assert addon_ids == {"addon1", "addon2"}
        
        # Test addon relationships for addon (should be empty)
        addons_from_addon = graph.get_addon_relationships("addon1")
        assert len(addons_from_addon) == 0
    
    def test_export_adjacency_matrix(self):
        """Test exporting adjacency matrix for deals."""
        graph = DealGraph()
        
        # Build sample graph with add-on relationships
        dataset = DealDataset(
            sectors=[Sector(id="tech", name="Technology")],
            regions=[Region(id="us", name="United States")],
            deals=[
                Deal(id="platform", name="Platform", sector_id="tech", region_id="us",
                     is_platform=True, status="current", description="Platform"),
                Deal(id="addon", name="Addon", sector_id="tech", region_id="us", 
                     is_platform=False, status="current", description="Addon")
            ],
            events=[Event(id="evt", type=EventType.ADDON, deal_id="platform", related_deal_id="addon")],
            snippets=[]
        )
        
        graph.build_from_dataset(dataset)
        
        # Export adjacency matrix for deals
        node_ids, matrix = graph.export_adjacency_matrix("deal")
        
        assert len(node_ids) == 2
        assert set(node_ids) == {"platform", "addon"}
        
        # Matrix should have 1 for ADDON_TO relationship
        platform_idx = node_ids.index("platform")
        addon_idx = node_ids.index("addon")
        
        assert matrix[platform_idx][addon_idx] == 1.0  # platform -> addon
        assert matrix[addon_idx][platform_idx] == 0.0  # addon -> platform (no reverse edge)


class TestEdgeType:
    """Test EdgeType constants."""
    
    def test_edge_type_constants(self):
        """Test that edge type constants have expected values."""
        assert EdgeType.IN_SECTOR == "IN_SECTOR"
        assert EdgeType.IN_REGION == "IN_REGION"
        assert EdgeType.ADDON_TO == "ADDON_TO"
        assert EdgeType.EXITED_VIA == "EXITED_VIA"
        assert EdgeType.DESCRIBED_IN == "DESCRIBED_IN"


class TestIntegration:
    """Integration tests using real sample data."""
    
    def test_build_graph_with_real_sample_data(self):
        """Test building graph with actual sample data."""
        from dealgraph.data.ingest import load_all
        
        # Load real sample data
        dataset = load_all("data/raw")
        
        # Build graph
        graph = DealGraph()
        graph.build_from_dataset(dataset)
        
        # Verify graph structure
        metrics = graph.get_deal_graph_metrics()
        
        assert metrics["total_nodes"] > 0
        assert metrics["total_edges"] > 0
        assert "deal" in metrics["node_counts"]
        assert "sector" in metrics["node_counts"]
        assert "IN_SECTOR" in metrics["edge_counts"]
        assert "IN_REGION" in metrics["edge_counts"]
        assert "ADDON_TO" in metrics["edge_counts"]
        assert "DESCRIBED_IN" in metrics["edge_counts"]
        
        # Test specific deal neighbors
        cloudtech_neighbors = graph.get_deal_neighbors("platform_tech_001")
        
        assert len(cloudtech_neighbors["sectors"]) > 0
        assert len(cloudtech_neighbors["regions"]) > 0
        assert len(cloudtech_neighbors["related_deals"]) > 0
        
        # Verify CloudTech has add-ons
        assert len(cloudtech_neighbors["addons"]) > 0
        
        # Test finding deals by criteria
        software_deals = graph.find_deals_by_criteria(sector_id="software")
        assert len(software_deals) > 0
        
        platform_deals = graph.find_deals_by_criteria(is_platform=True)
        assert len(platform_deals) > 0
        
        current_deals = graph.find_deals_by_criteria(status="current")
        assert len(current_deals) > 0
    
    def test_graph_connectivity(self):
        """Test that the built graph is properly connected."""
        from dealgraph.data.ingest import load_all
        
        # Load and build graph
        dataset = load_all("data/raw")
        graph = DealGraph()
        graph.build_from_dataset(dataset)
        
        # Get graph metrics
        metrics = graph.get_deal_graph_metrics()
        
        # Verify connectivity - for directed graph, we expect weak connectivity to be handled differently
        # All entity nodes should exist in the graph
        expected_nodes = len(dataset.sectors) + len(dataset.regions) + len(dataset.deals) + len(dataset.events) + len(dataset.snippets)
        assert metrics["total_nodes"] == expected_nodes
        
        # Verify that all deals are connected to their sectors and regions
        for deal in dataset.deals:
            # Each deal should have at least IN_SECTOR and IN_REGION edges (neighbors going OUT from deal)
            deal_neighbors = list(graph.graph[deal.id].keys())  # Get neighbors that this deal has edges TO
            sector_neighbor = deal.sector_id in deal_neighbors
            region_neighbor = deal.region_id in deal_neighbors
            assert sector_neighbor, f"Deal {deal.id} should be connected to sector {deal.sector_id}"
            assert region_neighbor, f"Deal {deal.id} should be connected to region {deal.region_id}"
        
        # Verify all deal relationships
        for deal in dataset.deals:
            if deal.status == "realized":
                # Realized deals should have exit events
                neighbors = graph.get_deal_neighbors(deal.id)
                assert len(neighbors["events"]) > 0
            
            if deal.is_platform:
                # Platform deals should potentially have add-on relationships
                neighbors = graph.get_deal_neighbors(deal.id)
                # Could have add-ons or not, but we can check the structure
                assert isinstance(neighbors["related_deals"], list)
                assert isinstance(neighbors["addons"], list)
    
    def test_sector_and_region_mappings(self):
        """Test that sector and region relationships are properly mapped."""
        from dealgraph.data.ingest import load_all
        
        dataset = load_all("data/raw")
        graph = DealGraph()
        graph.build_from_dataset(dataset)
        
        # Test that all deals have proper sector and region connections
        for deal in dataset.deals:
            sectors = graph.get_sectors_by_deal(deal.id)
            assert len(sectors) > 0
            assert sectors[0]["id"] == deal.sector_id
            
            neighbors = graph.get_deal_neighbors(deal.id)
            assert len(neighbors["regions"]) > 0
            assert neighbors["regions"][0]["id"] == deal.region_id
