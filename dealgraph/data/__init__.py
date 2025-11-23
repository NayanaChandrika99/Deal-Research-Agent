"""Data layer for DealGraph - schemas, ingestion, and graph building."""

from .schemas import (
    # Core entities
    Sector,
    Region, 
    Event,
    EventType,
    Snippet,
    Deal,
    # Search & ranking
    CandidateDeal,
    RankedDeal,
    # Reasoning outputs
    Precedent,
    DealReasoningOutput,
    # Data bundles
    DealDataset,
    # Type aliases
    DealList,
    SectorList,
    RegionList,
    EventList,
    SnippetList,
    CandidateDealList,
    RankedDealList,
)

from .ingest import (
    # Loading functions
    load_sectors,
    load_regions,
    load_deals,
    load_events,
    load_snippets,
    load_all,
    # Validation
    validate_referential_integrity,
    get_data_statistics,
    # Exceptions
    DataIngestionError,
    ValidationError,
)

from .graph_builder import (
    # Graph builder
    DealGraph,
    EdgeType,
)

__all__ = [
    # Core entities
    "Sector", "Region", "Event", "EventType", "Snippet", "Deal",
    # Search & ranking
    "CandidateDeal", "RankedDeal",
    # Reasoning outputs
    "Precedent", "DealReasoningOutput",
    # Data bundles
    "DealDataset",
    # Type aliases
    "DealList", "SectorList", "RegionList", "EventList", "SnippetList",
    "CandidateDealList", "RankedDealList",
    # Loading functions
    "load_sectors", "load_regions", "load_deals", "load_events", 
    "load_snippets", "load_all",
    # Validation
    "validate_referential_integrity", "get_data_statistics",
    # Exceptions
    "DataIngestionError", "ValidationError",
    # Graph builder
    "DealGraph", "EdgeType",
]
