# ABOUTME: Graph-derived feature computation for deal retrieval and ranking.
# ABOUTME: Extracts relationship features from DealGraph to enhance semantic search.

from typing import List, Dict, Any, Optional
from ..data.schemas import Deal
from ..data.graph_builder import DealGraph


def compute_graph_features(
    deal_graph: DealGraph,
    deal: Deal,
    query_sectors: Optional[List[str]] = None,
    query_regions: Optional[List[str]] = None,
    all_deals: Optional[List[Deal]] = None
) -> Dict[str, Any]:
    """
    Compute graph-derived features for a deal given query context.
    
    This function extracts various relationship-based features from the DealGraph
    that can be used to enhance semantic search and ranking of deals.
    
    Args:
        deal_graph: DealGraph instance containing all relationships
        deal: Deal to compute features for
        query_sectors: Optional list of relevant sectors from query
        query_regions: Optional list of relevant regions from query
        all_deals: Optional list of all deals for additional context
        
    Returns:
        Dictionary containing computed features:
        {
            'sector_match': int,     # 1 if deal sector matches query sectors
            'region_match': int,     # 1 if deal region matches query regions
            'num_addons': int,       # Number of add-on acquisitions
            'has_exit': int,         # 1 if deal has exit event
            'degree': int,           # Number of connections in graph
            'is_platform': int,      # 1 if deal is a platform
            'sector_degree': int,    # Number of deals in same sector
            'region_degree': int,    # Number of deals in same region
            'out_degree': int,       # Number of outgoing edges
            'in_degree': int         # Number of incoming edges
        }
    """
    # Initialize default features
    features = {
        'sector_match': 0,
        'region_match': 0,
        'num_addons': 0,
        'has_exit': 0,
        'degree': 0,
        'is_platform': 1 if deal.is_platform else 0,
        'sector_degree': 0,
        'region_degree': 0,
        'out_degree': 0,
        'in_degree': 0
    }
    
    # Get deal neighbors from graph
    try:
        neighbors = deal_graph.get_deal_neighbors(deal.id)
    except ValueError:
        # Deal not found in graph
        return features
    
    # Compute basic graph features
    total_degree = 0
    for category in neighbors.values():
        if isinstance(category, list):
            total_degree += len(category)
    
    features['degree'] = total_degree
    
    # Count add-ons (deals that this deal acquired)
    addons = neighbors.get('addons', [])
    features['num_addons'] = len(addons)
    
    # Count exits (deals with exit events)
    events = neighbors.get('events', [])
    exit_events = [event for event in events if event.get('type') == 'exit']
    features['has_exit'] = 1 if exit_events else 0
    
    # Compute sector and region matches if query context provided
    if query_sectors and deal.sector_id in query_sectors:
        features['sector_match'] = 1
    
    if query_regions and deal.region_id in query_regions:
        features['region_match'] = 1
    
    # Compute sector and region degrees (how many other deals share same sector/region)
    if all_deals:
        # Count deals in same sector
        sector_deals = [d for d in all_deals if d.sector_id == deal.sector_id]
        features['sector_degree'] = len(sector_deals) - 1  # Exclude self
        
        # Count deals in same region
        region_deals = [d for d in all_deals if d.region_id == deal.region_id]
        features['region_degree'] = len(region_deals) - 1  # Exclude self
    
    # Compute directional degrees using NetworkX
    if hasattr(deal_graph, 'graph'):
        node_id = deal.id
        if deal_graph.graph.has_node(node_id):
            features['out_degree'] = deal_graph.graph.out_degree(node_id)
            features['in_degree'] = deal_graph.graph.in_degree(node_id)
    
    return features


def compute_batch_graph_features(
    deal_graph: DealGraph,
    deals: List[Deal],
    query_sectors: Optional[List[str]] = None,
    query_regions: Optional[List[str]] = None
) -> Dict[str, Deal]:
    """
    Compute graph features for multiple deals in batch.
    
    Args:
        deal_graph: DealGraph instance
        deals: List of deals to compute features for
        query_sectors: Optional list of relevant sectors from query
        query_regions: Optional list of relevant regions from query
        
    Returns:
        Dictionary mapping deal IDs to their computed features
    """
    features_map = {}
    
    for deal in deals:
        features = compute_graph_features(
            deal_graph=deal_graph,
            deal=deal,
            query_sectors=query_sectors,
            query_regions=query_regions,
            all_deals=deals
        )
        features_map[deal.id] = features
    
    return features_map


def extract_query_context(
    query: str,
    known_sectors: Optional[List[str]] = None,
    known_regions: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """
    Extract sector and region context from a natural language query.
    
    This is a simple implementation that can be enhanced with more sophisticated
    NLP techniques (NER, entity matching, etc.).
    
    Args:
        query: Natural language query string
        known_sectors: Optional list of known sector names/IDs to match against
        known_regions: Optional list of known region names/IDs to match against
        
    Returns:
        Dictionary with 'sectors' and 'regions' lists
    """
    query_lower = query.lower()
    query_sectors = []
    query_regions = []
    
    # Simple keyword matching (can be enhanced)
    sector_keywords = {
        'tech': ['software', 'technology', 'saas', 'cloud', 'ai', 'data'],
        'healthcare': ['healthcare', 'health', 'medical', 'pharma', 'biotech'],
        'industrial': ['industrial', 'manufacturing', 'materials', 'logistics'],
        'financial_services': ['financial', 'fintech', 'banking', 'insurance'],
        'business_services': ['services', 'consulting', 'outsourcing', 'bpo']
    }
    
    region_keywords = {
        'us': ['united states', 'usa', 'america', 'us', 'american'],
        'europe': ['europe', 'european', 'eu', 'uk', 'germany', 'france'],
        'canada': ['canada', 'canadian'],
        'apac': ['asia', 'pacific', 'apac', 'china', 'japan', 'india']
    }
    
    # Match sectors
    if known_sectors:
        for sector_id in known_sectors:
            # Check both the sector ID and potential keywords
            if sector_id in query_lower:
                query_sectors.append(sector_id)
            elif sector_id in sector_keywords:
                for keyword in sector_keywords[sector_id]:
                    if keyword in query_lower:
                        query_sectors.append(sector_id)
                        break
    
    # Match regions  
    if known_regions:
        for region_id in known_regions:
            # Check both the region ID and potential keywords
            if region_id in query_lower:
                query_regions.append(region_id)
            elif region_id in region_keywords:
                for keyword in region_keywords[region_id]:
                    if keyword in query_lower:
                        query_regions.append(region_id)
                        break
    
    # Remove duplicates while preserving order
    query_sectors = list(dict.fromkeys(query_sectors))
    query_regions = list(dict.fromkeys(query_regions))
    
    return {
        'sectors': query_sectors,
        'regions': query_regions
    }


def enhance_features_with_text(
    features: Dict[str, Any],
    deal: Deal,
    query: str,
    text_similarity: float
) -> Dict[str, Any]:
    """
    Enhance graph features with text similarity and metadata.
    
    Args:
        features: Base graph features dictionary
        deal: Deal object
        query: Original query string
        text_similarity: Cosine similarity score from embeddings
        
    Returns:
        Enhanced features dictionary with additional text/metadata features
    """
    # Create a copy to avoid modifying the original
    enhanced = features.copy()
    
    # Add text similarity
    enhanced['text_similarity'] = text_similarity
    
    # Add text length features (proxy for deal description richness)
    description_words = len(deal.description.split()) if deal.description else 0
    enhanced['description_length'] = description_words
    
    # Add temporal features
    enhanced['has_investment_year'] = 1 if deal.year_invested else 0
    enhanced['has_exit_year'] = 1 if deal.year_exited else 0
    
    # Calculate deal age if investment year known
    if deal.year_invested:
        # Assuming current year is 2024 for calculation
        import datetime
        current_year = datetime.datetime.now().year
        enhanced['deal_age_years'] = current_year - deal.year_invested
    else:
        enhanced['deal_age_years'] = 0
    
    # Add composite features
    enhanced['is_mature_deal'] = 1 if enhanced.get('deal_age_years', 0) >= 3 else 0
    enhanced['has_complete_data'] = 1 if (enhanced['has_investment_year'] and enhanced['description_length'] > 10) else 0
    
    # Text-graph alignment score
    # Higher when text similarity and graph relevance align
    graph_relevance = (
        enhanced['sector_match'] * 0.4 +
        enhanced['region_match'] * 0.3 +
        min(enhanced['num_addons'] / 5.0, 1.0) * 0.2 +  # Normalized add-on count
        enhanced['is_platform'] * 0.1
    )
    enhanced['text_graph_alignment'] = (text_similarity + graph_relevance) / 2.0
    
    return enhanced


def get_feature_names() -> List[str]:
    """
    Get the complete list of feature names computed by this module.
    
    Returns:
        List of feature names in consistent order
    """
    return [
        'sector_match',
        'region_match', 
        'num_addons',
        'has_exit',
        'degree',
        'is_platform',
        'sector_degree',
        'region_degree',
        'out_degree',
        'in_degree',
        'text_similarity',
        'description_length',
        'has_investment_year',
        'has_exit_year',
        'deal_age_years',
        'is_mature_deal',
        'has_complete_data',
        'text_graph_alignment'
    ]


def validate_features(features: Dict[str, Any]) -> bool:
    """
    Validate that computed features are well-formed.
    
    Args:
        features: Features dictionary to validate
        
    Returns:
        True if features are valid, False otherwise
    """
    required_types = {
        'sector_match': int,
        'region_match': int,
        'num_addons': int,
        'has_exit': int,
        'degree': int,
        'is_platform': int,
        'sector_degree': int,
        'region_degree': int,
        'out_degree': int,
        'in_degree': int,
        'text_similarity': float,
        'description_length': int,
        'has_investment_year': int,
        'has_exit_year': int,
        'deal_age_years': int,
        'is_mature_deal': int,
        'has_complete_data': int,
        'text_graph_alignment': float
    }
    
    # Check all required features are present
    for feature_name in required_types:
        if feature_name not in features:
            return False
        
        # Check type
        if not isinstance(features[feature_name], required_types[feature_name]):
            return False
    
    # Check value ranges for binary features
    binary_features = ['sector_match', 'region_match', 'has_exit', 'is_platform', 
                      'has_investment_year', 'has_exit_year', 'is_mature_deal', 'has_complete_data']
    
    for feature_name in binary_features:
        if features[feature_name] not in [0, 1]:
            return False
    
    # Check non-negative counts
    count_features = ['num_addons', 'degree', 'sector_degree', 'region_degree', 
                     'out_degree', 'in_degree', 'description_length', 'deal_age_years']
    
    for feature_name in count_features:
        if features[feature_name] < 0:
            return False
    
    # Check similarity scores are in valid range
    similarity_features = ['text_similarity', 'text_graph_alignment']
    for feature_name in similarity_features:
        if not (0.0 <= features[feature_name] <= 1.0):
            return False
    
    return True
