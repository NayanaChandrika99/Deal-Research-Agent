# ABOUTME: NetworkX-based DealGraph builder for creating graph structure from deal data.
# ABOUTME: Builds MultiDiGraph with typed edges connecting deals, sectors, regions, events, and snippets.

import networkx as nx
from typing import Dict, List, Any, Optional, Set, Tuple
from ..data.schemas import (
    Deal, Sector, Region, Event, EventType, Snippet, DealList, 
    SectorList, RegionList, EventList, SnippetList
)


class EdgeType:
    """Constants for edge types in the DealGraph."""
    IN_SECTOR = "IN_SECTOR"
    IN_REGION = "IN_REGION"
    ADDON_TO = "ADDON_TO"
    EXITED_VIA = "EXITED_VIA"
    DESCRIBED_IN = "DESCRIBED_IN"


class DealGraph:
    """
    NetworkX-based graph representing deal relationships and metadata.
    
    Uses MultiDiGraph to support multiple edges between the same nodes
    with different relationship types.
    """
    
    def __init__(self):
        """Initialize empty DealGraph with MultiDiGraph."""
        self.graph = nx.MultiDiGraph()
        self._node_type_mapping = {}  # node_id -> node_type for fast lookup
    
    def add_deals(self, deals: DealList) -> None:
        """
        Add deal nodes to the graph.
        
        Args:
            deals: List of Deal objects to add as nodes
        """
        for deal in deals:
            # Add deal node with attributes
            self.graph.add_node(
                deal.id,
                type="deal",
                name=deal.name,
                sector_id=deal.sector_id,
                region_id=deal.region_id,
                is_platform=deal.is_platform,
                status=deal.status,
                description=deal.description,
                year_invested=deal.year_invested,
                year_exited=deal.year_exited
            )
            
            # Store node type mapping
            self._node_type_mapping[deal.id] = "deal"
    
    def add_sectors(self, sectors: SectorList) -> None:
        """
        Add sector nodes to the graph.
        
        Args:
            sectors: List of Sector objects to add as nodes
        """
        for sector in sectors:
            self.graph.add_node(
                sector.id,
                type="sector",
                name=sector.name
            )
            
            self._node_type_mapping[sector.id] = "sector"
    
    def add_regions(self, regions: RegionList) -> None:
        """
        Add region nodes to the graph.
        
        Args:
            regions: List of Region objects to add as nodes
        """
        for region in regions:
            self.graph.add_node(
                region.id,
                type="region",
                name=region.name
            )
            
            self._node_type_mapping[region.id] = "region"
    
    def add_events(self, events: EventList) -> None:
        """
        Add event nodes and relationships to the graph.
        
        Args:
            events: List of Event objects to add as nodes and edges
        """
        for event in events:
            # Add event node
            self.graph.add_node(
                event.id,
                type="event",
                event_type=event.type.value,
                deal_id=event.deal_id,
                related_deal_id=event.related_deal_id,
                date=event.date,
                description=event.description
            )
            
            self._node_type_mapping[event.id] = "event"
            
            # Add relationship edges based on event type
            if event.type == EventType.ADDON:
                if event.related_deal_id:
                    # ADDON_TO: Deal (platform) -> Deal (addon)
                    self.graph.add_edge(
                        event.deal_id,
                        event.related_deal_id,
                        relation=EdgeType.ADDON_TO,
                        event_id=event.id,
                        date=event.date
                    )
            
            elif event.type == EventType.EXIT:
                # EXITED_VIA: Deal -> Event
                self.graph.add_edge(
                    event.deal_id,
                    event.id,
                    relation=EdgeType.EXITED_VIA,
                    date=event.date,
                    description=event.description
                )
    
    def add_snippets(self, snippets: SnippetList) -> None:
        """
        Add snippet nodes and text relationship edges.
        
        Args:
            snippets: List of Snippet objects to add as nodes and edges
        """
        for snippet in snippets:
            # Add snippet node
            self.graph.add_node(
                snippet.id,
                type="snippet",
                deal_id=snippet.deal_id,
                source=snippet.source,
                text=snippet.text
            )
            
            self._node_type_mapping[snippet.id] = "snippet"
            
            # Add DESCRIBED_IN edge: Deal -> Snippet
            self.graph.add_edge(
                snippet.deal_id,
                snippet.id,
                relation=EdgeType.DESCRIBED_IN,
                source=snippet.source
            )
    
    def build_from_dataset(self, dataset: Any) -> None:
        """
        Build complete graph from DealDataset.
        
        Args:
            dataset: DealDataset object containing all data entities
        """
        # Add all node types first
        self.add_sectors(dataset.sectors)
        self.add_regions(dataset.regions)
        self.add_deals(dataset.deals)
        self.add_events(dataset.events)
        self.add_snippets(dataset.snippets)
        
        # Add relationship edges for deals
        self._add_deal_taxonomy_edges(dataset.deals, dataset.sectors, dataset.regions)
    
    def _add_deal_taxonomy_edges(
        self, 
        deals: DealList, 
        sectors: SectorList, 
        regions: RegionList
    ) -> None:
        """
        Add taxonomy relationship edges for deals.
        
        Args:
            deals: List of deals
            sectors: List of sectors  
            regions: List of regions
        """
        # Create lookup sets for validation
        sector_ids = {s.id for s in sectors}
        region_ids = {r.id for r in regions}
        
        for deal in deals:
            # Add IN_SECTOR edge
            if deal.sector_id in sector_ids:
                self.graph.add_edge(
                    deal.id,
                    deal.sector_id,
                    relation=EdgeType.IN_SECTOR
                )
            
            # Add IN_REGION edge
            if deal.region_id in region_ids:
                self.graph.add_edge(
                    deal.id,
                    deal.region_id,
                    relation=EdgeType.IN_REGION
                )
    
    def has_edge(self, source: str, target: str) -> bool:
        """
        Check if an edge exists between source and target nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            True if edge exists, False otherwise
        """
        return self.graph.has_edge(source, target)

    def get_deal_neighbors(self, deal_id: str) -> Dict[str, Any]:
        """
        Get all neighbors of a deal categorized by relationship type.
        
        Args:
            deal_id: ID of the deal to get neighbors for
            
        Returns:
            Dictionary with neighbor categories:
            {
                'sectors': List of sector nodes,
                'regions': List of region nodes, 
                'events': List of event nodes,
                'snippets': List of snippet nodes,
                'related_deals': List of related deal nodes,
                'addons': List of addon deal nodes (subset of related_deals),
                'platforms': List of platform deal nodes (subset of related_deals)
            }
        """
        if not self.graph.has_node(deal_id):
            raise ValueError(f"Deal '{deal_id}' not found in graph")
        
        # Get all neighbors using different edge types
        neighbors = {
            'sectors': [],
            'regions': [],
            'events': [],
            'snippets': [],
            'related_deals': [],
            'addons': [],
            'platforms': []
        }
        
        # Get all neighbors with edge data
        for neighbor_id, edge_data_dict in self.graph[deal_id].items():
            neighbor_type = self._node_type_mapping.get(neighbor_id, 'unknown')
            
            # Handle case where there might be multiple edges (get first one)
            if isinstance(edge_data_dict, dict):
                edge_data = edge_data_dict
            else:
                # Multiple edges case - get first edge data
                edge_data = list(edge_data_dict.values())[0] if edge_data_dict else {}
            
            edge_relation = edge_data.get('relation', 'unknown')
            
            # Categorize neighbor based on edge type
            if edge_relation == EdgeType.IN_SECTOR and neighbor_type == 'sector':
                neighbors['sectors'].append({
                    'id': neighbor_id,
                    'name': self.graph.nodes[neighbor_id].get('name'),
                    'type': 'sector'
                })
            
            elif edge_relation == EdgeType.IN_REGION and neighbor_type == 'region':
                neighbors['regions'].append({
                    'id': neighbor_id,
                    'name': self.graph.nodes[neighbor_id].get('name'),
                    'type': 'region'
                })
            
            elif edge_relation == EdgeType.EXITED_VIA and neighbor_type == 'event':
                neighbors['events'].append({
                    'id': neighbor_id,
                    'type': self.graph.nodes[neighbor_id].get('event_type'),
                    'date': self.graph.nodes[neighbor_id].get('date'),
                    'description': self.graph.nodes[neighbor_id].get('description'),
                    'type': 'event'
                })
            
            elif edge_relation == EdgeType.DESCRIBED_IN and neighbor_type == 'snippet':
                neighbors['snippets'].append({
                    'id': neighbor_id,
                    'source': self.graph.nodes[neighbor_id].get('source'),
                    'text': self.graph.nodes[neighbor_id].get('text'),
                    'type': 'snippet'
                })
            
            elif edge_relation == EdgeType.ADDON_TO and neighbor_type == 'deal':
                related_deal = {
                    'id': neighbor_id,
                    'name': self.graph.nodes[neighbor_id].get('name'),
                    'is_platform': self.graph.nodes[neighbor_id].get('is_platform'),
                    'status': self.graph.nodes[neighbor_id].get('status'),
                    'type': 'deal'
                }
                
                neighbors['related_deals'].append(related_deal)
                
                # Categorize as addon or platform
                if self.graph.nodes[neighbor_id].get('is_platform', False):
                    neighbors['platforms'].append(related_deal)
                else:
                    neighbors['addons'].append(related_deal)
        
        return neighbors
    
    def get_deal_graph_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the DealGraph structure.
        
        Returns:
            Dictionary with graph statistics
        """
        # Count nodes by type
        node_counts = {}
        for node_id in self.graph.nodes():
            node_type = self._node_type_mapping.get(node_id, 'unknown')
            node_counts[node_type] = node_counts.get(node_type, 0) + 1
        
        # Count edges by type
        edge_counts = {}
        for _, _, edge_data in self.graph.edges(data=True):
            edge_relation = edge_data.get('relation', 'unknown')
            edge_counts[edge_relation] = edge_counts.get(edge_relation, 0) + 1
        
        # Calculate graph metrics
        metrics = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_counts': node_counts,
            'edge_counts': edge_counts,
            'is_connected': nx.is_weakly_connected(self.graph),
            'density': nx.density(self.graph)
        }
        
        # Calculate degree statistics
        degrees = dict(self.graph.degree())
        if degrees:
            metrics['avg_degree'] = sum(degrees.values()) / len(degrees)
            metrics['max_degree'] = max(degrees.values())
            metrics['min_degree'] = min(degrees.values())
        
        return metrics
    
    def find_deals_by_criteria(
        self,
        sector_id: Optional[str] = None,
        region_id: Optional[str] = None,
        is_platform: Optional[bool] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find deals matching specific criteria.
        
        Args:
            sector_id: Filter by sector ID
            region_id: Filter by region ID  
            is_platform: Filter by platform status
            status: Filter by deal status
            
        Returns:
            List of deal dictionaries matching criteria
        """
        matching_deals = []
        
        for deal_id in self.graph.nodes():
            if self._node_type_mapping.get(deal_id) != 'deal':
                continue
                
            deal_data = self.graph.nodes[deal_id]
            
            # Apply filters
            if sector_id and deal_data.get('sector_id') != sector_id:
                continue
            if region_id and deal_data.get('region_id') != region_id:
                continue
            if is_platform is not None and deal_data.get('is_platform') != is_platform:
                continue
            if status and deal_data.get('status') != status:
                continue
            
            matching_deals.append({
                'id': deal_id,
                'name': deal_data.get('name'),
                'sector_id': deal_data.get('sector_id'),
                'region_id': deal_data.get('region_id'),
                'is_platform': deal_data.get('is_platform'),
                'status': deal_data.get('status'),
                'description': deal_data.get('description')
            })
        
        return matching_deals
    
    def get_sectors_by_deal(self, deal_id: str) -> List[Dict[str, Any]]:
        """
        Get sectors associated with a specific deal.
        
        Args:
            deal_id: ID of the deal
            
        Returns:
            List of sector dictionaries
        """
        neighbors = self.get_deal_neighbors(deal_id)
        return neighbors['sectors']
    
    def get_events_by_deal(self, deal_id: str) -> List[Dict[str, Any]]:
        """
        Get events associated with a specific deal.
        
        Args:
            deal_id: ID of the deal
            
        Returns:
            List of event dictionaries
        """
        neighbors = self.get_deal_neighbors(deal_id)
        return neighbors['events']
    
    def get_addon_relationships(self, deal_id: str) -> List[Dict[str, Any]]:
        """
        Get addon relationships for a specific deal.
        
        Args:
            deal_id: ID of the deal (can be platform or addon)
            
        Returns:
            List of addon relationship dictionaries
        """
        neighbors = self.get_deal_neighbors(deal_id)
        return neighbors['addons']
    
    def get_platform_relationships(self, deal_id: str) -> List[Dict[str, Any]]:
        """
        Get platform relationships for a specific deal.
        
        Args:
            deal_id: ID of the deal
            
        Returns:
            List of platform relationship dictionaries
        """
        neighbors = self.get_deal_neighbors(deal_id)
        return neighbors['platforms']
    
    def export_adjacency_matrix(self, node_type: str = "deal") -> Tuple[List[str], List[List[float]]]:
        """
        Export adjacency matrix for a specific node type.
        
        Args:
            node_type: Type of nodes to include ('deal', 'sector', 'region', etc.)
            
        Returns:
            Tuple of (node_ids, adjacency_matrix)
        """
        # Get all nodes of specified type
        node_ids = [
            node_id for node_id in self.graph.nodes()
            if self._node_type_mapping.get(node_id) == node_type
        ]
        
        if not node_ids:
            return [], []
        
        # Create adjacency matrix
        node_index = {node_id: i for i, node_id in enumerate(node_ids)}
        matrix = [[0.0 for _ in node_ids] for _ in node_ids]
        
        for i, node_id in enumerate(node_ids):
            for neighbor_id in self.graph[node_id]:
                if neighbor_id in node_index:
                    j = node_index[neighbor_id]
                    matrix[i][j] = 1.0
        
        return node_ids, matrix
