# ABOUTME: Graph-aware semantic search combining vector similarity with relationship features.
# ABOUTME: Implements the core retrieval functionality for finding relevant deals.

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ..data.schemas import Deal, Snippet, CandidateDeal, DealList, SnippetList
from .features import (
    compute_graph_features, 
    compute_batch_graph_features, 
    extract_query_context,
    enhance_features_with_text,
    get_feature_names
)
from ..embeddings import EmbeddingEncoder, DealEmbeddingIndex
from ..data.graph_builder import DealGraph


def graph_semantic_search(
    query: str,
    encoder: EmbeddingEncoder,
    deal_index: DealEmbeddingIndex,
    deal_graph: DealGraph,
    deals: DealList,
    snippets_by_deal: Dict[str, SnippetList],
    top_k: int = 50,
    filter_sectors: Optional[List[str]] = None,
    filter_regions: Optional[List[str]] = None,
    filter_platforms: Optional[Optional[bool]] = None,
    min_similarity: float = 0.0
) -> List[CandidateDeal]:
    """
    Perform graph-aware semantic search for relevant deals.
    
    This function combines semantic similarity (from embeddings) with graph-derived
    features to retrieve and rank candidate deals that match the query.
    
    Args:
        query: Natural language search query
        encoder: EmbeddingEncoder for text vectorization
        deal_index: DealEmbeddingIndex for similarity search
        deal_graph: DealGraph for relationship queries
        deals: List of all available deals
        snippets_by_deal: Dictionary mapping deal IDs to their snippets
        top_k: Maximum number of results to return
        filter_sectors: Optional list of sector IDs to filter results
        filter_regions: Optional list of region IDs to filter results  
        filter_platforms: Optional filter for platform (True) vs add-on (False) deals
        min_similarity: Minimum text similarity threshold (0.0 to 1.0)
        
    Returns:
        List of CandidateDeal objects with computed similarity and features,
        sorted by relevance (text similarity + graph features)
    """
    # Step 1: Encode the query
    try:
        query_vector = encoder.embed_text(query)
    except Exception as e:
        raise ValueError(f"Failed to encode query: {e}")
    
    # Step 2: Extract query context (sectors, regions) for enhanced features
    known_sectors = [s.id for s in getattr(deal_graph, '_sectors', [])]
    known_regions = [r.id for r in getattr(deal_graph, '_regions', [])]
    
    # If sectors/regions aren't stored in graph, extract from deals
    if not known_sectors:
        known_sectors = list(set(deal.sector_id for deal in deals))
    if not known_regions:
        known_regions = list(set(deal.region_id for deal in deals))
    
    query_context = extract_query_context(query, known_sectors, known_regions)
    
    # Step 3: Perform semantic search using embeddings
    search_results = deal_index.search(query_vector, top_k=top_k * 2)  # Get more for filtering
    
    if not search_results:
        return []
    
    # Step 4: Filter search results based on criteria
    filtered_deal_ids = []
    for deal_id, similarity_score in search_results:
        # Apply minimum similarity threshold
        if similarity_score < min_similarity:
            continue
        
        # Find the deal object
        deal = next((d for d in deals if d.id == deal_id), None)
        if not deal:
            continue
        
        # Apply sector filter
        if filter_sectors and deal.sector_id not in filter_sectors:
            continue
        
        # Apply region filter  
        if filter_regions and deal.region_id not in filter_regions:
            continue
        
        # Apply platform filter
        if filter_platforms is not None and deal.is_platform != filter_platforms:
            continue
        
        filtered_deal_ids.append(deal_id)
    
    if not filtered_deal_ids:
        return []
    
    # Step 5: Compute graph features for filtered deals
    candidate_deals = []
    
    for deal_id in filtered_deal_ids:
        deal = next(d for d in deals if d.id == deal_id)
        
        # Get text similarity from search results
        similarity_score = next(score for id, score in search_results if id == deal_id)
        
        # Compute graph features
        graph_features = compute_graph_features(
            deal_graph=deal_graph,
            deal=deal,
            query_sectors=query_context['sectors'],
            query_regions=query_context['regions'],
            all_deals=deals
        )
        
        # Enhance features with text similarity and metadata
        enhanced_features = enhance_features_with_text(
            features=graph_features,
            deal=deal,
            query=query,
            text_similarity=similarity_score
        )
        
        # Get relevant snippets
        relevant_snippets = snippets_by_deal.get(deal_id, [])
        
        # Create CandidateDeal
        candidate = CandidateDeal(
            deal=deal,
            snippets=relevant_snippets,
            text_similarity=similarity_score,
            graph_features=enhanced_features
        )
        
        candidate_deals.append(candidate)
    
    # Step 6: Rank candidates by combined relevance score
    ranked_candidates = rank_candidates(candidate_deals)
    
    # Step 7: Return top k results
    return ranked_candidates[:top_k]


def rank_candidates(candidates: List[CandidateDeal]) -> List[CandidateDeal]:
    """
    Rank candidate deals by combined relevance score.
    
    Args:
        candidates: List of CandidateDeal objects to rank
        
    Returns:
        Same list sorted by relevance (descending)
    """
    def compute_relevance_score(candidate: CandidateDeal) -> float:
        """Compute combined relevance score from text and graph features."""
        features = candidate.graph_features
        
        # Text similarity weight (40%)
        text_score = features.get('text_similarity', 0.0) * 0.4
        
        # Graph relevance weights (60%)
        # Sector match (20%)
        sector_score = features.get('sector_match', 0) * 0.2
        
        # Region match (15%)
        region_score = features.get('region_match', 0) * 0.15
        
        # Platform bonus (10%) - platforms often more relevant as search results
        platform_score = features.get('is_platform', 0) * 0.1
        
        # Add-on activity (10%) - more active deals often more relevant
        addon_score = min(features.get('num_addons', 0) / 3.0, 1.0) * 0.1
        
        # Exit status (5%) - realized deals can be good precedents
        exit_score = features.get('has_exit', 0) * 0.05
        
        # Text-graph alignment (bonus factor)
        alignment_bonus = features.get('text_graph_alignment', 0.0) * 0.1
        
        total_score = (text_score + sector_score + region_score + 
                      platform_score + addon_score + exit_score + alignment_bonus)
        
        return total_score
    
    # Sort by relevance score (descending)
    ranked = sorted(candidates, key=compute_relevance_score, reverse=True)
    
    return ranked


def batch_search(
    queries: List[str],
    encoder: EmbeddingEncoder,
    deal_index: DealEmbeddingIndex,
    deal_graph: DealGraph,
    deals: DealList,
    snippets_by_deal: Dict[str, SnippetList],
    **search_kwargs
) -> List[List[CandidateDeal]]:
    """
    Perform batch search for multiple queries.
    
    Args:
        queries: List of search queries
        encoder: EmbeddingEncoder for text vectorization
        deal_index: DealEmbeddingIndex for similarity search
        deal_graph: DealGraph for relationship queries
        deals: List of all available deals
        snippets_by_deal: Dictionary mapping deal IDs to their snippets
        **search_kwargs: Additional arguments passed to graph_semantic_search
        
    Returns:
        List of result lists, one for each query
    """
    results = []
    
    for query in queries:
        try:
            search_results = graph_semantic_search(
                query=query,
                encoder=encoder,
                deal_index=deal_index,
                deal_graph=deal_graph,
                deals=deals,
                snippets_by_deal=snippets_by_deal,
                **search_kwargs
            )
            results.append(search_results)
        except Exception as e:
            # Log error and return empty results for this query
            print(f"Search failed for query '{query}': {e}")
            results.append([])
    
    return results


def get_search_explanation(candidate: CandidateDeal, query: str) -> Dict[str, Any]:
    """
    Generate an explanation for why a candidate deal was ranked highly.
    
    Args:
        candidate: CandidateDeal to explain
        query: Original search query
        
    Returns:
        Dictionary with explanation details
    """
    features = candidate.graph_features
    deal = candidate.deal
    
    text_similarity = (
        features['text_similarity']
        if 'text_similarity' in features
        else getattr(candidate, 'text_similarity', 0.0)
    )
    
    explanation = {
        'deal_id': deal.id,
        'deal_name': deal.name,
        'query': query,
        'text_similarity': text_similarity,
        'relevance_factors': []
    }
    
    # Text similarity explanation
    if text_similarity > 0.7:
        explanation['relevance_factors'].append(f"High text similarity ({text_similarity:.3f})")
    elif text_similarity > 0.5:
        explanation['relevance_factors'].append(f"Moderate text similarity ({text_similarity:.3f})")
    
    # Sector match
    if features.get('sector_match', 0):
        explanation['relevance_factors'].append(f"Sector match: {deal.sector_id}")
    
    # Region match
    if features.get('region_match', 0):
        explanation['relevance_factors'].append(f"Region match: {deal.region_id}")
    
    # Platform status
    if features.get('is_platform', 0):
        explanation['relevance_factors'].append("Platform deal")
    
    # Add-on activity
    num_addons = features.get('num_addons', 0)
    if num_addons > 0:
        explanation['relevance_factors'].append(f"Has {num_addons} add-on acquisitions")
    
    # Exit status
    if features.get('has_exit', 0):
        explanation['relevance_factors'].append("Has successful exit")
    
    # Text-graph alignment
    alignment = features.get('text_graph_alignment', 0)
    if alignment > 0.7:
        explanation['relevance_factors'].append(f"Strong text-graph alignment ({alignment:.3f})")
    
    return explanation


def search_with_explanations(
    query: str,
    encoder: EmbeddingEncoder,
    deal_index: DealEmbeddingIndex,
    deal_graph: DealGraph,
    deals: DealList,
    snippets_by_deal: Dict[str, SnippetList],
    top_k: int = 10,
    **search_kwargs
) -> Tuple[List[CandidateDeal], List[Dict[str, Any]]]:
    """
    Perform search and return results with explanations.
    
    Args:
        query: Search query
        encoder: EmbeddingEncoder
        deal_index: DealEmbeddingIndex  
        deal_graph: DealGraph
        deals: List of deals
        snippets_by_deal: Snippets by deal ID
        top_k: Number of results
        **search_kwargs: Additional search parameters
        
    Returns:
        Tuple of (results, explanations)
    """
    # Perform search
    results = graph_semantic_search(
        query=query,
        encoder=encoder,
        deal_index=deal_index,
        deal_graph=deal_graph,
        deals=deals,
        snippets_by_deal=snippets_by_deal,
        top_k=top_k,
        **search_kwargs
    )
    
    # Generate explanations
    explanations = []
    for candidate in results:
        explanation = get_search_explanation(candidate, query)
        explanations.append(explanation)
    
    return results, explanations


def build_snippets_index(snippets: SnippetList) -> Dict[str, SnippetList]:
    """
    Build a mapping from deal IDs to their snippets for efficient lookup.
    
    Args:
        snippets: List of all snippets
        
    Returns:
        Dictionary mapping deal IDs to lists of snippets
    """
    snippets_by_deal = {}
    
    for snippet in snippets:
        if snippet.deal_id not in snippets_by_deal:
            snippets_by_deal[snippet.deal_id] = []
        snippets_by_deal[snippet.deal_id].append(snippet)
    
    return snippets_by_deal


def validate_search_parameters(**kwargs) -> None:
    """
    Validate search parameters and raise appropriate errors.
    
    Args:
        **kwargs: Search parameters to validate
        
    Raises:
        ValueError: If parameters are invalid
    """
    # Validate top_k
    top_k = kwargs.get('top_k', 50)
    if not isinstance(top_k, int) or top_k < 1 or top_k > 1000:
        raise ValueError("top_k must be an integer between 1 and 1000")
    
    # Validate min_similarity
    min_similarity = kwargs.get('min_similarity', 0.0)
    if not isinstance(min_similarity, (int, float)) or not (0.0 <= min_similarity <= 1.0):
        raise ValueError("min_similarity must be a float between 0.0 and 1.0")
    
    # Validate filter lists
    for filter_name in ['filter_sectors', 'filter_regions']:
        filter_value = kwargs.get(filter_name)
        if filter_value is not None:
            if not isinstance(filter_value, list):
                raise ValueError(f"{filter_name} must be a list")
            if not all(isinstance(item, str) for item in filter_value):
                raise ValueError(f"All items in {filter_name} must be strings")
