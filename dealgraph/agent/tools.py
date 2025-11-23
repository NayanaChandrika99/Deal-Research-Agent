# ABOUTME: Agent-facing tool wrappers for graph-aware semantic search functionality.
# ABOUTME: Provides easy-to-use tools for the agent orchestrator to perform deal retrieval.

from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING

import numpy as np
import logging
from pathlib import Path
import pickle
import time

from ..data.ingest import load_all
from ..data.graph_builder import DealGraph
from ..embeddings import DealEmbeddingIndex, get_embedding_encoder

if TYPE_CHECKING:
    from ..embeddings.encoder import EmbeddingEncoder
from ..retrieval import (
    graph_semantic_search,
    batch_search,
    search_with_explanations,
    get_search_explanation,
    build_snippets_index,
    validate_search_parameters,
    get_feature_names
)
from ..data.schemas import Deal, Snippet, CandidateDeal, DealList


class SearchIndexManager:
    """
    Manages the built search index for efficient retrieval operations.
    
    Provides caching and lifecycle management for the DealGraph and DealEmbeddingIndex.
    """
    
    def __init__(self):
        self._graph: Optional[DealGraph] = None
        self._index: Optional[DealEmbeddingIndex] = None
        self._deals: Optional[DealList] = None
        self._snippets_by_deal: Optional[Dict[str, List[Snippet]]] = None
        self._encoder = None
        self._cache_path: Optional[Path] = None
        self._last_built: Optional[float] = None
        self._build_lock = False
    
    def build_index(
        self, 
        data_path: str = "data/raw",
        cache_path: Optional[str] = None,
        force_rebuild: bool = False
    ) -> Tuple[DealGraph, DealEmbeddingIndex]:
        """
        Build complete search index from data.
        
        Args:
            data_path: Path to raw data directory
            cache_path: Optional path to cache built index
            force_rebuild: Force rebuild even if cache exists
            
        Returns:
            Tuple of (DealGraph, DealEmbeddingIndex)
        """
        if self._build_lock:
            raise RuntimeError("Index build already in progress")
        
        self._build_lock = True
        
        try:
            # Check cache first
            if cache_path and not force_rebuild:
                cache_file = Path(cache_path) / "search_index.pkl"
                if cache_file.exists():
                    logging.info(f"Loading search index from cache: {cache_file}")
                    return self._load_from_cache(cache_file)
            
            logging.info("Building search index from scratch...")
            start_time = time.time()
            
            # Load data
            dataset = load_all(data_path)
            logging.info(f"Loaded {len(dataset.deals)} deals, {len(dataset.snippets)} snippets")
            
            # Build graph
            graph = DealGraph()
            graph.build_from_dataset(dataset)
            logging.info(
                "Built graph: %s nodes, %s edges",
                graph.graph.number_of_nodes(),
                graph.graph.number_of_edges()
            )
            
            # Build snippets lookup once
            snippets_by_deal = build_snippets_index(dataset.snippets)
            
            # Create embeddings and index
            encoder = get_embedding_encoder()
            deal_text_pairs = [
                (deal, self._compose_embedding_text(deal, snippets_by_deal.get(deal.id, [])))
                for deal in dataset.deals
            ]
            text_blocks = [text for _, text in deal_text_pairs]
            vectors, encoder = self._generate_embeddings(encoder, text_blocks)
            index = DealEmbeddingIndex(encoder.get_dimension(), encoder=encoder)
            
            for (deal, _), vector in zip(deal_text_pairs, vectors):
                index.add(deal.id, vector, {
                    'name': deal.name,
                    'description': deal.description,
                    'sector_id': deal.sector_id,
                    'region_id': deal.region_id,
                    'is_platform': deal.is_platform,
                    'status': deal.status
                })
            
            logging.info(f"Created embeddings index: {index.size()} embeddings")
            
            # Store in manager
            self._graph = graph
            self._index = index
            self._deals = dataset.deals
            self._snippets_by_deal = snippets_by_deal
            self._encoder = encoder
            self._cache_path = Path(cache_path) if cache_path else None
            self._last_built = time.time()
            
            # Save to cache
            if cache_path:
                self._save_to_cache(cache_file)
            
            build_time = time.time() - start_time
            logging.info(f"Search index built in {build_time:.2f} seconds")
            
            return graph, index
            
        finally:
            self._build_lock = False
    
    def _load_from_cache(self, cache_file: Path) -> Tuple[DealGraph, DealEmbeddingIndex]:
        """Load index from cache file."""
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        self._graph = cache_data['graph']
        self._index = cache_data['index']
        self._deals = cache_data['deals']
        self._snippets_by_deal = cache_data['snippets_by_deal']
        self._encoder = cache_data['encoder']
        self._last_built = cache_data['timestamp']
        
        logging.info(f"Loaded search index from cache (built {time.time() - self._last_built:.0f}s ago)")
        return self._graph, self._index
    
    def _save_to_cache(self, cache_file: Path) -> None:
        """Save index to cache file."""
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        cache_data = {
            'graph': self._graph,
            'index': self._index,
            'deals': self._deals,
            'snippets_by_deal': self._snippets_by_deal,
            'encoder': self._encoder,
            'timestamp': time.time()
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logging.info(f"Saved search index to cache: {cache_file}")
    
    def get_components(self) -> Tuple[DealGraph, DealEmbeddingIndex, DealList, Dict[str, List[Snippet]], "EmbeddingEncoder"]:
        """Get all loaded components for search operations."""
        if not all([self._graph, self._index, self._deals, self._snippets_by_deal, self._encoder]):
            raise RuntimeError("Search index not built. Call build_index() first.")
        
        return self._graph, self._index, self._deals, self._snippets_by_deal, self._encoder
    
    def is_ready(self) -> bool:
        """Check if search index is ready for operations."""
        return all([self._graph, self._index, self._deals, self._snippets_by_deal, self._encoder])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded search index."""
        if not self.is_ready():
            return {"ready": False}
        
        graph_metrics = self._graph.get_deal_graph_metrics()
        index_stats = self._index.get_stats()
        
        return {
            "ready": True,
            "last_built": self._last_built,
            "cache_path": str(self._cache_path) if self._cache_path else None,
            "graph": {
                "total_nodes": graph_metrics["total_nodes"],
                "total_edges": graph_metrics["total_edges"],
                "node_counts": graph_metrics["node_counts"],
                "edge_counts": graph_metrics["edge_counts"]
            },
            "index": {
                "size": index_stats["size"],
                "dimension": index_stats["embedding_dim"],
                "memory_usage_mb": index_stats["memory_usage_mb"]
            },
            "data": {
                "total_deals": len(self._deals),
                "deals_with_snippets": len(self._snippets_by_deal),
                "sectors": len(set(deal.sector_id for deal in self._deals)),
                "regions": len(set(deal.region_id for deal in self._deals))
            }
        }

    def reset(self) -> None:
        """Reset the cached components (primarily for tests)."""
        self._graph = None
        self._index = None
        self._deals = None
        self._snippets_by_deal = None
        self._encoder = None
        self._cache_path = None
        self._last_built = None

    def _compose_embedding_text(self, deal: Deal, snippets: List[Snippet]) -> str:
        """Create a descriptive text block for embedding generation."""
        snippet_text = " ".join(snippet.text for snippet in snippets[:2])
        parts = [
            deal.name,
            deal.description or "",
            f"Sector: {deal.sector_id}. Region: {deal.region_id}.",
            snippet_text
        ]
        return " ".join(part.strip() for part in parts if part)

    def _generate_embeddings(self, encoder, texts: List[str]) -> Tuple[List[np.ndarray], Any]:
        """
        Embed text blocks, falling back to the local encoder if the remote provider fails.
        """
        try:
            return encoder.embed_texts(texts), encoder
        except Exception as exc:
            logging.warning("Embedding provider failed (%s). Falling back to hashing encoder.", exc)
            fallback_encoder = get_embedding_encoder(force_local=True)
            return fallback_encoder.embed_texts(texts), fallback_encoder


# Global search index manager
_search_manager = SearchIndexManager()


def _ensure_search_components() -> Tuple[DealGraph, DealEmbeddingIndex, DealList, Dict[str, List[Snippet]], Any]:
    """Guarantee that search components are loaded and return them."""
    try:
        return _search_manager.get_components()
    except RuntimeError:
        logging.info("Search index not ready, building...")
        build_search_index()
        return _search_manager.get_components()


def build_search_index(
    data_path: str = "data/raw",
    cache_path: Optional[str] = None,
    force_rebuild: bool = False
) -> Tuple[DealGraph, DealEmbeddingIndex]:
    """
    Build complete search index from data.
    
    This is the main entry point for building the search infrastructure.
    
    Args:
        data_path: Path to raw data directory containing JSON files
        cache_path: Optional path to cache built index
        force_rebuild: Force rebuild even if cache exists
        
    Returns:
        Tuple of (DealGraph, DealEmbeddingIndex) ready for search operations
    """
    return _search_manager.build_index(data_path, cache_path, force_rebuild)


def tool_load_search_data() -> Tuple[DealGraph, DealEmbeddingIndex, Dict[str, Any]]:
    """
    Load search data components for agent operations.
    
    This is a convenience function for the agent orchestrator to get
    all necessary components for search operations.
    
    Returns:
        Tuple of (DealGraph, DealEmbeddingIndex, metadata_dict)
    """
    graph, index, deals, snippets_by_deal, _ = _ensure_search_components()
    
    metadata = {
        "ready": True,
        "num_deals": len(deals),
        "num_snippets": sum(len(snippets) for snippets in snippets_by_deal.values()),
        "feature_names": get_feature_names(),
        "search_components_loaded": True
    }
    
    return graph, index, metadata


def tool_graph_semantic_search(
    query: str,
    top_k: int = 10,
    filter_sectors: Optional[List[str]] = None,
    filter_regions: Optional[List[str]] = None,
    filter_platforms: Optional[bool] = None,
    min_similarity: float = 0.0,
    include_explanations: bool = True
) -> List[Dict[str, Any]]:
    """
    Agent-facing wrapper for graph-aware semantic search.
    
    This is the main search tool that the agent orchestrator will use.
    
    Args:
        query: Natural language search query
        top_k: Maximum number of results to return
        filter_sectors: Optional list of sector IDs to filter results
        filter_regions: Optional list of region IDs to filter results
        filter_platforms: Optional filter for platform vs add-on deals
        min_similarity: Minimum text similarity threshold
        include_explanations: Whether to include search explanations
        
    Returns:
        List of search results with explanations
    """
    # Validate parameters
    try:
        validate_search_parameters(
            top_k=top_k,
            min_similarity=min_similarity,
            filter_sectors=filter_sectors,
            filter_regions=filter_regions
        )
    except ValueError as e:
        logging.error(f"Invalid search parameters: {e}")
        return []
    
    try:
        graph, index, deals, snippets_by_deal, encoder = _ensure_search_components()
        logging.info("Search query: '%s' (top_k=%s)", query, top_k)
        
        if include_explanations:
            candidates, explanations = search_with_explanations(
                query=query,
                encoder=encoder,
                deal_index=index,
                deal_graph=graph,
                deals=deals,
                snippets_by_deal=snippets_by_deal,
                top_k=top_k,
                filter_sectors=filter_sectors,
                filter_regions=filter_regions,
                filter_platforms=filter_platforms,
                min_similarity=min_similarity
            )
            
            results = []
            for candidate, explanation in zip(candidates, explanations):
                result = {
                    "deal_id": candidate.deal.id,
                    "deal_name": candidate.deal.name,
                    "sector_id": candidate.deal.sector_id,
                    "region_id": candidate.deal.region_id,
                    "is_platform": candidate.deal.is_platform,
                    "status": candidate.deal.status,
                    "description": candidate.deal.description,
                    "text_similarity": candidate.text_similarity,
                    "graph_features": candidate.graph_features,
                    "snippets": [
                        {
                            "id": snippet.id,
                            "source": snippet.source,
                            "text": snippet.text
                        }
                        for snippet in candidate.snippets
                    ],
                    "explanation": explanation,
                    "relevance_score": _compute_relevance_score(candidate)
                }
                results.append(result)
            return results
        
        candidates = graph_semantic_search(
            query=query,
            encoder=encoder,
            deal_index=index,
            deal_graph=graph,
            deals=deals,
            snippets_by_deal=snippets_by_deal,
            top_k=top_k,
            filter_sectors=filter_sectors,
            filter_regions=filter_regions,
            filter_platforms=filter_platforms,
            min_similarity=min_similarity
        )
        
        results = []
        for candidate in candidates:
            result = {
                "deal_id": candidate.deal.id,
                "deal_name": candidate.deal.name,
                "sector_id": candidate.deal.sector_id,
                "region_id": candidate.deal.region_id,
                "is_platform": candidate.deal.is_platform,
                "status": candidate.deal.status,
                "description": candidate.deal.description,
                "text_similarity": candidate.text_similarity,
                "graph_features": candidate.graph_features,
                "snippets": [
                    {
                        "id": snippet.id,
                        "source": snippet.source,
                        "text": snippet.text
                    }
                    for snippet in candidate.snippets
                ],
                "relevance_score": _compute_relevance_score(candidate)
            }
            results.append(result)
        return results
    except Exception as e:
        logging.error(f"Search failed: {e}")
        return []


def tool_batch_search(
    queries: List[str],
    **search_kwargs
) -> List[List[Dict[str, Any]]]:
    """
    Agent-facing wrapper for batch search operations.
    
    Args:
        queries: List of search queries
        **search_kwargs: Additional search parameters
        
    Returns:
        List of result lists, one for each query
    """
    results = []
    
    for query in queries:
        try:
            query_results = tool_graph_semantic_search(query, **search_kwargs)
            results.append(query_results)
        except Exception as e:
            logging.error(f"Batch search failed for query '{query}': {e}")
            results.append([])
    
    return results


def tool_get_search_explanations(
    candidates: List[Dict[str, Any]],
    query: str
) -> List[Dict[str, Any]]:
    """
    Generate explanations for search results.
    
    Args:
        candidates: Search results from tool_graph_semantic_search
        query: Original search query
        
    Returns:
        Enhanced results with explanations
    """
    explanations = []
    
    for candidate in candidates:
        try:
            explanation = {
                "deal_id": candidate["deal_id"],
                "deal_name": candidate["deal_name"],
                "query": query,
                "relevance_factors": _extract_relevance_factors(candidate),
                "similarity_breakdown": _analyze_similarity(candidate),
                "graph_insights": _analyze_graph_features(candidate),
                "recommendation": _generate_recommendation(candidate)
            }
            explanations.append(explanation)
        except Exception as e:
            logging.error(f"Failed to generate explanation: {e}")
            explanations.append({
                "deal_id": candidate.get("deal_id", "unknown"),
                "deal_name": candidate.get("deal_name", "Unknown"),
                "query": query,
                "relevance_factors": ["Error generating explanation"],
                "similarity_breakdown": {},
                "graph_insights": {},
                "recommendation": "Unable to analyze this result"
            })
    
    return explanations


def _compute_relevance_score(candidate: CandidateDeal) -> float:
    """Compute overall relevance score for a candidate."""
    features = candidate.graph_features
    return (
        features.get('text_similarity', 0.0) * 0.4 +
        features.get('sector_match', 0) * 0.2 +
        features.get('region_match', 0) * 0.15 +
        features.get('is_platform', 0) * 0.1 +
        min(features.get('num_addons', 0) / 5.0, 1.0) * 0.1 +
        features.get('text_graph_alignment', 0.0) * 0.05
    )


def _extract_relevance_factors(candidate: Dict[str, Any]) -> List[str]:
    """Extract relevance factors from candidate result."""
    features = candidate.get('graph_features', {})
    factors = []
    
    if features.get('text_similarity', 0) > 0.7:
        factors.append(f"High text similarity ({features['text_similarity']:.3f})")
    elif features.get('text_similarity', 0) > 0.5:
        factors.append(f"Good text similarity ({features['text_similarity']:.3f})")
    
    if features.get('sector_match', 0):
        factors.append(f"Sector match: {candidate['sector_id']}")
    
    if features.get('region_match', 0):
        factors.append(f"Region match: {candidate['region_id']}")
    
    if features.get('is_platform', 0):
        factors.append("Platform deal")
    
    if features.get('num_addons', 0) > 0:
        factors.append(f"Has {features['num_addons']} add-on acquisitions")
    
    if features.get('has_exit', 0):
        factors.append("Has successful exit")
    
    return factors


def _analyze_similarity(candidate: Dict[str, Any]) -> Dict[str, float]:
    """Analyze text similarity components."""
    features = candidate.get('graph_features', {})
    return {
        "text_similarity": features.get('text_similarity', 0.0),
        "description_length": features.get('description_length', 0),
        "text_graph_alignment": features.get('text_graph_alignment', 0.0)
    }


def _analyze_graph_features(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze graph relationship features."""
    features = candidate.get('graph_features', {})
    return {
        "sector_match": features.get('sector_match', 0),
        "region_match": features.get('region_match', 0),
        "is_platform": features.get('is_platform', 0),
        "num_addons": features.get('num_addons', 0),
        "has_exit": features.get('has_exit', 0),
        "graph_degree": features.get('degree', 0)
    }


def _generate_recommendation(candidate: Dict[str, Any]) -> str:
    """Generate recommendation based on candidate analysis."""
    features = candidate.get('graph_features', {})
    
    if features.get('sector_match', 0) and features.get('region_match', 0):
        return "Strong match for query context"
    elif features.get('is_platform', 0) and features.get('num_addons', 0) > 0:
        return "Active platform with add-on strategy"
    elif features.get('has_exit', 0):
        return "Successful precedent with exit"
    else:
        return "Relevant based on text similarity"


def validate_search_setup() -> Dict[str, Any]:
    """
    Validate that the search setup is ready for operations.
    
    Returns:
        Dictionary with validation status and component information
    """
    try:
        graph, index, deals, snippets_by_deal, _ = _ensure_search_components()
        return {
            "ready": True,
            "components": {
                "graph": graph.graph.number_of_nodes(),
                "index": index.size(),
                "deals": len(deals),
                "snippets": sum(len(s) for s in snippets_by_deal.values())
            },
            "feature_names": get_feature_names(),
            "search_capabilities": {
                "semantic_search": True,
                "graph_features": True,
                "batch_search": True,
                "explanations": True
            }
        }
    except Exception as e:
        return {
            "ready": False,
            "error": str(e),
            "components": {},
            "search_capabilities": {}
        }
