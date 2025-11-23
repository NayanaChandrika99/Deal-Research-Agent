# ABOUTME: Data ingestion functions for loading raw deal data and validating referential integrity.
# ABOUTME: Loads JSON/CSV files and converts them to Pydantic models with comprehensive validation.

import json
from pathlib import Path
from typing import List, Dict, Any
from ..data.schemas import (
    Sector, Region, Event, EventType, Snippet, Deal, DealDataset,
    SectorList, RegionList, EventList, SnippetList, DealList
)


class DataIngestionError(Exception):
    """Raised when data ingestion fails due to validation or file errors."""
    pass


class ValidationError(Exception):
    """Raised when data validation fails (missing IDs, broken references, etc.)."""
    pass


def load_sectors(path: str | Path) -> SectorList:
    """
    Load sectors from JSON file.
    
    Args:
        path: Path to sectors JSON file
        
    Returns:
        List of Sector objects
        
    Raises:
        DataIngestionError: If file cannot be loaded or parsed
        ValidationError: If sector data is invalid
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        raise DataIngestionError(f"Failed to load sectors from {path}: {e}")
    
    sectors = []
    for item in data:
        try:
            sector = Sector(**item)
            sectors.append(sector)
        except Exception as e:
            raise ValidationError(f"Invalid sector data: {item}, error: {e}")
    
    return sectors


def load_regions(path: str | Path) -> RegionList:
    """
    Load regions from JSON file.
    
    Args:
        path: Path to regions JSON file
        
    Returns:
        List of Region objects
        
    Raises:
        DataIngestionError: If file cannot be loaded or parsed
        ValidationError: If region data is invalid
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        raise DataIngestionError(f"Failed to load regions from {path}: {e}")
    
    regions = []
    for item in data:
        try:
            region = Region(**item)
            regions.append(region)
        except Exception as e:
            raise ValidationError(f"Invalid region data: {item}, error: {e}")
    
    return regions


def load_deals(path: str | Path) -> DealList:
    """
    Load deals from JSON file.
    
    Args:
        path: Path to deals JSON file
        
    Returns:
        List of Deal objects
        
    Raises:
        DataIngestionError: If file cannot be loaded or parsed
        ValidationError: If deal data is invalid
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        raise DataIngestionError(f"Failed to load deals from {path}: {e}")
    
    deals = []
    for item in data:
        try:
            deal = Deal(**item)
            deals.append(deal)
        except Exception as e:
            raise ValidationError(f"Invalid deal data: {item}, error: {e}")
    
    return deals


def load_events(path: str | Path) -> EventList:
    """
    Load events from JSON file.
    
    Args:
        path: Path to events JSON file
        
    Returns:
        List of Event objects
        
    Raises:
        DataIngestionError: If file cannot be loaded or parsed
        ValidationError: If event data is invalid
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        raise DataIngestionError(f"Failed to load events from {path}: {e}")
    
    events = []
    for item in data:
        try:
            # Convert event type string to EventType enum
            if 'type' in item and isinstance(item['type'], str):
                item['type'] = EventType(item['type'])
            
            event = Event(**item)
            events.append(event)
        except Exception as e:
            raise ValidationError(f"Invalid event data: {item}, error: {e}")
    
    return events


def load_snippets(path: str | Path) -> SnippetList:
    """
    Load snippets from JSON file.
    
    Args:
        path: Path to snippets JSON file
        
    Returns:
        List of Snippet objects
        
    Raises:
        DataIngestionError: If file cannot be loaded or parsed
        ValidationError: If snippet data is invalid
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        raise DataIngestionError(f"Failed to load snippets from {path}: {e}")
    
    snippets = []
    for item in data:
        try:
            snippet = Snippet(**item)
            snippets.append(snippet)
        except Exception as e:
            raise ValidationError(f"Invalid snippet data: {item}, error: {e}")
    
    return snippets


def validate_referential_integrity(
    sectors: SectorList,
    regions: RegionList, 
    deals: DealList,
    events: EventList,
    snippets: SnippetList
) -> None:
    """
    Validate referential integrity across all data entities.
    
    Args:
        sectors: List of sectors
        regions: List of regions
        deals: List of deals
        events: List of events
        snippets: List of snippets
        
    Raises:
        ValidationError: If any referential integrity issues are found
    """
    # Create lookup sets for validation
    sector_ids = {s.id for s in sectors}
    region_ids = {r.id for r in regions}
    deal_ids = {d.id for d in deals}
    
    # Validate deal references
    for deal in deals:
        if deal.sector_id not in sector_ids:
            raise ValidationError(
                f"Deal '{deal.id}' references unknown sector_id '{deal.sector_id}'"
            )
        if deal.region_id not in region_ids:
            raise ValidationError(
                f"Deal '{deal.id}' references unknown region_id '{deal.region_id}'"
            )
    
    # Validate event references
    for event in events:
        if event.deal_id not in deal_ids:
            raise ValidationError(
                f"Event '{event.id}' references unknown deal_id '{event.deal_id}'"
            )
        if event.related_deal_id and event.related_deal_id not in deal_ids:
            raise ValidationError(
                f"Event '{event.id}' references unknown related_deal_id '{event.related_deal_id}'"
            )
    
    # Validate snippet references
    for snippet in snippets:
        if snippet.deal_id not in deal_ids:
            raise ValidationError(
                f"Snippet '{snippet.id}' references unknown deal_id '{snippet.deal_id}'"
            )


def load_all(base_path: str | Path) -> DealDataset:
    """
    Load all data from raw data directory and return validated dataset.
    
    Args:
        base_path: Path to directory containing raw data files
        
    Returns:
        DealDataset with all loaded and validated data
        
    Raises:
        DataIngestionError: If any files cannot be loaded
        ValidationError: If data validation fails
    """
    base_path = Path(base_path)
    
    # Load all data files
    sectors = load_sectors(base_path / "sectors.json")
    regions = load_regions(base_path / "regions.json")
    deals = load_deals(base_path / "deals.json")
    events = load_events(base_path / "events.json")
    snippets = load_snippets(base_path / "snippets.json")
    
    # Validate referential integrity
    validate_referential_integrity(sectors, regions, deals, events, snippets)
    
    return DealDataset(
        sectors=sectors,
        regions=regions,
        deals=deals,
        events=events,
        snippets=snippets
    )


def get_data_statistics(dataset: DealDataset) -> Dict[str, Any]:
    """
    Get statistics about the loaded dataset.
    
    Args:
        dataset: DealDataset to analyze
        
    Returns:
        Dictionary with data statistics
    """
    # Count platform vs add-on deals
    platforms = [d for d in dataset.deals if d.is_platform]
    addons = [d for d in dataset.deals if not d.is_platform]
    
    # Count current vs realized deals
    current_deals = [d for d in dataset.deals if d.status == "current"]
    realized_deals = [d for d in dataset.deals if d.status == "realized"]
    
    # Count events by type
    addon_events = [e for e in dataset.events if e.type == EventType.ADDON]
    exit_events = [e for e in dataset.events if e.type == EventType.EXIT]
    
    # Calculate sectors and regions distribution
    sector_counts = {}
    region_counts = {}
    
    for deal in dataset.deals:
        sector_counts[deal.sector_id] = sector_counts.get(deal.sector_id, 0) + 1
        region_counts[deal.region_id] = region_counts.get(deal.region_id, 0) + 1
    
    return {
        "total_deals": len(dataset.deals),
        "platform_deals": len(platforms),
        "addon_deals": len(addons),
        "current_deals": len(current_deals),
        "realized_deals": len(realized_deals),
        "total_events": len(dataset.events),
        "addon_events": len(addon_events),
        "exit_events": len(exit_events),
        "total_snippets": len(dataset.snippets),
        "sectors": sector_counts,
        "regions": region_counts,
        "sector_names": {s.id: s.name for s in dataset.sectors},
        "region_names": {r.id: r.name for r in dataset.regions}
    }
