# ABOUTME: Pydantic data models for DealGraph entities and structures.
# ABOUTME: Defines core schemas for deals, sectors, events, snippets, and reasoning outputs.

from typing import List, Optional
from enum import Enum
from pydantic import BaseModel


class EventType(str, Enum):
    ADDON = "addon"
    EXIT = "exit"


class Sector(BaseModel):
    """Taxonomy node for deal categorization (e.g., Healthcare, Industrial Services)."""
    id: str
    name: str


class Region(BaseModel):
    """Geography node for deal locations (e.g., United States, Europe)."""
    id: str
    name: str


class Event(BaseModel):
    """Time-based actions related to deals (add-ons, exits, etc.)."""
    id: str
    type: EventType
    deal_id: str  # deal this event belongs to
    related_deal_id: Optional[str] = None  # for ADDON_TO relationships
    date: Optional[str] = None  # ISO date as string
    description: Optional[str] = None


class Snippet(BaseModel):
    """Textual evidence supporting a deal (news articles, case studies)."""
    id: str
    deal_id: str
    source: str  # e.g., "news", "case_study"
    text: str


class Deal(BaseModel):
    """Portfolio company investment with metadata and relationships."""
    id: str
    name: str
    sector_id: str
    region_id: str
    is_platform: bool
    status: str  # e.g., "current", "realized"
    description: str
    # optional fields:
    year_invested: Optional[int] = None
    year_exited: Optional[int] = None


class CandidateDeal(BaseModel):
    """Search result containing a deal with similarity scores and graph features."""
    deal: Deal
    snippets: List[Snippet]
    text_similarity: float  # cosine similarity from embedding search
    graph_features: dict  # e.g., {"sector_match": 1, "num_addons": 3}


class RankedDeal(BaseModel):
    """Candidate deal with ML ranking score and position."""
    candidate: CandidateDeal
    score: float  # ML model score
    rank: int  # position in ranked list (1-based)


class Precedent(BaseModel):
    """Selected historical deal with explanation of similarity."""
    deal_id: str
    name: str
    similarity_reason: str


class DealReasoningOutput(BaseModel):
    """LLM analysis output containing precedents, insights, and narrative."""
    precedents: List[Precedent]
    playbook_levers: List[str]
    risk_themes: List[str]
    narrative_summary: str


# Convenience type aliases
DealList = List[Deal]
SectorList = List[Sector]
RegionList = List[Region]
EventList = List[Event]
SnippetList = List[Snippet]
CandidateDealList = List[CandidateDeal]
RankedDealList = List[RankedDeal]


# Data bundles for ingestion
class DealDataset(BaseModel):
    """Complete dataset bundle for DealGraph construction."""
    sectors: SectorList
    regions: RegionList
    deals: DealList
    events: EventList
    snippets: SnippetList
