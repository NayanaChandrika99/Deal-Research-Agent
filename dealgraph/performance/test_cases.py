# ABOUTME: Standardized test case generator for performance benchmarking.
# ABOUTME: Creates diverse test scenarios for comprehensive prompt evaluation.

import logging
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from dealgraph.data.schemas import Deal, Snippet, CandidateDeal, RankedDeal
from dealgraph.data.ingest import load_all


@dataclass
class DealTestCase:
    """Individual test case for benchmarking."""
    
    case_id: str
    query: str
    ranked_deals: List[RankedDeal]
    category: str
    difficulty: str  # "easy", "medium", "hard"
    reference_data: Optional[Dict[str, Any]] = None
    expected_metrics: Optional[Dict[str, float]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class TestCaseGenerator:
    """
    Generator for standardized test cases for performance benchmarking.
    
    Creates diverse test scenarios based on deal categories, difficulty levels,
    and realistic investment scenarios.
    """
    
    def __init__(self, data_source: Optional[str] = None):
        """
        Initialize test case generator.
        
        Args:
            data_source: Path to data source (defaults to 'data/raw')
        """
        self.data_source = data_source or 'data/raw'
        self.logger = logging.getLogger(__name__)
        
        # Load reference data
        self.dataset = self._load_reference_data()
        
        # Define query templates by category
        self.query_templates = {
            "technology": [
                "Looking for software platform acquisitions in the enterprise SaaS space",
                "Interested in AI/ML technology companies with strong growth metrics",
                "Seeking cybersecurity companies with recurring revenue models",
                "Looking for fintech companies with regulatory advantages"
            ],
            "healthcare": [
                "Looking for medical device companies with FDA approval",
                "Interested in healthcare IT companies serving hospitals",
                "Seeking pharmaceutical companies with strong pipelines",
                "Looking for medical imaging companies with AI capabilities"
            ],
            "manufacturing": [
                "Looking for industrial automation companies",
                "Interested in aerospace component manufacturers",
                "Seeking specialty chemical companies",
                "Looking for automotive parts suppliers with EV focus"
            ],
            "services": [
                "Looking for business services companies with sticky customers",
                "Interested in professional services firms with recurring revenue",
                "Seeking marketing services companies with digital capabilities",
                "Looking for facility management companies"
            ],
            "consumer": [
                "Looking for consumer brands with direct-to-consumer capability",
                "Interested in e-commerce companies with strong logistics",
                "Seeking food and beverage companies with premium positioning",
                "Looking for consumer electronics with subscription models"
            ]
        }
        
        # Define difficulty patterns
        self.difficulty_patterns = {
            "easy": {
                "num_candidates": 3,
                "similarity_range": (0.7, 0.9),
                "clear_precedents": True,
                "expected_precedents": (2, 4)
            },
            "medium": {
                "num_candidates": 6,
                "similarity_range": (0.5, 0.8),
                "clear_precedents": False,
                "expected_precedents": (1, 3)
            },
            "hard": {
                "num_candidates": 10,
                "similarity_range": (0.3, 0.7),
                "clear_precedents": False,
                "expected_precedents": (0, 2)
            }
        }
    
    def generate_test_suite(
        self,
        num_cases: int = 100,
        categories: List[str] = None,
        difficulty_distribution: Dict[str, float] = None
    ) -> List[DealTestCase]:
        """
        Generate comprehensive test suite.
        
        Args:
            num_cases: Number of test cases to generate
            categories: List of categories to include (defaults to all)
            difficulty_distribution: Distribution of difficulty levels
            
        Returns:
            List of DealTestCase objects
        """
        if categories is None:
            categories = list(self.query_templates.keys())
        
        if difficulty_distribution is None:
            difficulty_distribution = {"easy": 0.3, "medium": 0.5, "hard": 0.2}
        
        self.logger.info(f"Generating {num_cases} test cases across {len(categories)} categories")
        
        test_cases = []
        cases_per_category = num_cases // len(categories)
        
        for category in categories:
            category_cases = self._generate_category_cases(
                category, cases_per_category, difficulty_distribution
            )
            test_cases.extend(category_cases)
        
        # Shuffle and limit to requested number
        random.shuffle(test_cases)
        return test_cases[:num_cases]
    
    def _generate_category_cases(
        self,
        category: str,
        num_cases: int,
        difficulty_distribution: Dict[str, float]
    ) -> List[DealTestCase]:
        """Generate test cases for a specific category."""
        cases = []
        category_deals = self._get_deals_by_category(category)
        
        if not category_deals:
            self.logger.warning(f"No deals found for category: {category}")
            return cases
        
        for i in range(num_cases):
            # Select difficulty
            difficulty = self._select_difficulty(difficulty_distribution)
            
            # Generate test case
            case = self._generate_single_case(
                category=category,
                difficulty=difficulty,
                available_deals=category_deals,
                case_index=i
            )
            
            if case:
                cases.append(case)
        
        return cases
    
    def _generate_single_case(
        self,
        category: str,
        difficulty: str,
        available_deals: List[Deal],
        case_index: int
    ) -> Optional[DealTestCase]:
        """Generate a single test case."""
        try:
            pattern = self.difficulty_patterns[difficulty]
            
            # Select query template
            query = random.choice(self.query_templates[category])
            
            # Select candidate deals
            num_candidates = pattern["num_candidates"]
            selected_deals = random.sample(
                available_deals,
                min(num_candidates, len(available_deals))
            )
            
            # Create ranked deals with realistic similarity scores
            ranked_deals = []
            for j, deal in enumerate(selected_deals):
                similarity = self._generate_realistic_similarity(
                    deal, pattern["similarity_range"], j
                )
                
                # Create snippets for the deal
                snippets = self._generate_snippets_for_deal(deal)
                
                # Create candidate deal
                candidate = CandidateDeal(
                    deal=deal,
                    snippets=snippets,
                    text_similarity=similarity,
                    graph_features=self._generate_graph_features(deal, category)
                )
                
                # Create ranked deal
                ranked_deal = RankedDeal(
                    candidate=candidate,
                    score=similarity,
                    rank=j + 1
                )
                
                ranked_deals.append(ranked_deal)
            
            # Sort by score (descending)
            ranked_deals.sort(key=lambda x: x.score, reverse=True)
            
            # Generate case ID
            case_id = f"{category}_{difficulty}_{case_index:03d}"
            
            # Create test case
            test_case = DealTestCase(
                case_id=case_id,
                query=query,
                ranked_deals=ranked_deals,
                category=category,
                difficulty=difficulty
            )
            
            return test_case
            
        except Exception as e:
            self.logger.warning(f"Failed to generate test case: {e}")
            return None
    
    def _load_reference_data(self) -> Any:
        """Load reference dataset for test case generation."""
        try:
            return load_all(self.data_source)
        except Exception as e:
            self.logger.warning(f"Failed to load reference data: {e}")
            return None
    
    def _get_deals_by_category(self, category: str) -> List[Deal]:
        """Get deals filtered by category."""
        if not self.dataset:
            return []
        
        # Filter deals by sector/region matching the category
        # This is a simplified approach - in practice, you'd have more sophisticated filtering
        category_deals = []
        
        for deal in self.dataset.deals:
            if self._deal_matches_category(deal, category):
                category_deals.append(deal)
        
        return category_deals
    
    def _deal_matches_category(self, deal: Deal, category: str) -> bool:
        """Check if a deal matches the given category."""
        # Simple category matching based on sector/description
        category_keywords = {
            "technology": ["software", "tech", "digital", "ai", "saas", "platform"],
            "healthcare": ["health", "medical", "pharma", "biotech", "hospital"],
            "manufacturing": ["manufacturing", "industrial", "automotive", "aerospace"],
            "services": ["services", "consulting", "professional", "business"],
            "consumer": ["consumer", "retail", "brand", "food", "beverage"]
        }
        
        if category.lower() not in category_keywords:
            return False
        
        keywords = category_keywords[category.lower()]
        deal_text = f"{deal.name} {deal.description}".lower()
        
        return any(keyword in deal_text for keyword in keywords)
    
    def _select_difficulty(self, distribution: Dict[str, float]) -> str:
        """Select difficulty level based on distribution."""
        rand = random.random()
        cumulative = 0
        
        for difficulty, probability in distribution.items():
            cumulative += probability
            if rand <= cumulative:
                return difficulty
        
        return "medium"  # Fallback
    
    def _generate_realistic_similarity(
        self,
        deal: Deal,
        similarity_range: tuple,
        rank: int
    ) -> float:
        """Generate realistic similarity score for a deal."""
        min_sim, max_sim = similarity_range
        
        # Base similarity within range
        base_similarity = random.uniform(min_sim, max_sim)
        
        # Add rank-based degradation (higher rank = higher similarity)
        rank_bonus = max(0, 0.1 - (rank * 0.01))
        
        # Add some randomness
        noise = random.uniform(-0.05, 0.05)
        
        final_similarity = base_similarity + rank_bonus + noise
        
        # Ensure it's within reasonable bounds
        return max(0.0, min(1.0, final_similarity))
    
    def _generate_snippets_for_deal(self, deal: Deal) -> List[Snippet]:
        """Generate realistic snippets for a deal."""
        snippets = []
        
        # Create 1-3 snippets per deal
        num_snippets = random.randint(1, 3)
        
        for i in range(num_snippets):
            snippet = Snippet(
                id=f"snippet_{deal.id}_{i}",
                deal_id=deal.id,
                source=random.choice(["news", "case_study", "press_release"]),
                text=f"Sample text about {deal.name} - {deal.description}"
            )
            snippets.append(snippet)
        
        return snippets
    
    def _generate_graph_features(self, deal: Deal, category: str) -> Dict[str, float]:
        """Generate realistic graph features for a deal."""
        features = {}
        
        # Sector match feature
        features["sector_match"] = 1.0 if self._deal_matches_category(deal, category) else 0.5
        
        # Region match (simplified)
        features["region_match"] = random.uniform(0.3, 1.0)
        
        # Platform indicator
        features["is_platform"] = 1.0 if deal.is_platform else 0.0
        
        # Status-based features
        features["status_current"] = 1.0 if deal.status == "current" else 0.0
        features["status_realized"] = 1.0 if deal.status == "realized" else 0.0
        
        # Size indicators (if available)
        if hasattr(deal, 'enterprise_value'):
            features["size_large"] = 1.0 if deal.enterprise_value > 100 else 0.3
        
        return features
    
    def generate_edge_cases(self) -> List[DealTestCase]:
        """Generate edge cases for comprehensive testing."""
        edge_cases = []
        
        # Edge case: No candidates
        no_candidates_case = DealTestCase(
            case_id="edge_no_candidates",
            query="Looking for deals in a very niche market",
            ranked_deals=[],
            category="technology",
            difficulty="hard"
        )
        edge_cases.append(no_candidates_case)
        
        # Edge case: Single candidate
        single_candidate_case = DealTestCase(
            case_id="edge_single_candidate",
            query="Looking for a very specific type of deal",
            ranked_deals=[],
            category="healthcare",
            difficulty="easy"
        )
        edge_cases.append(single_candidate_case)
        
        # Edge case: Very low similarity scores
        if self.dataset and self.dataset.deals:
            deal = self.dataset.deals[0]
            low_similarity_deal = RankedDeal(
                candidate=CandidateDeal(
                    deal=deal,
                    snippets=[],
                    text_similarity=0.1,
                    graph_features={}
                ),
                score=0.1,
                rank=1
            )
            
            low_sim_case = DealTestCase(
                case_id="edge_low_similarity",
                query="Looking for deals with very different characteristics",
                ranked_deals=[low_similarity_deal],
                category="manufacturing",
                difficulty="hard"
            )
            edge_cases.append(low_similarity_case)
        
        return edge_cases
    
    def get_categories(self) -> List[str]:
        """Get available categories."""
        return list(self.query_templates.keys())
    
    def get_difficulty_levels(self) -> List[str]:
        """Get available difficulty levels."""
        return list(self.difficulty_patterns.keys())
    
    def validate_test_case(self, test_case: DealTestCase) -> Dict[str, Any]:
        """Validate a test case for correctness."""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required fields
        if not test_case.case_id:
            validation["valid"] = False
            validation["errors"].append("Missing case_id")
        
        if not test_case.query:
            validation["valid"] = False
            validation["errors"].append("Missing query")
        
        if not test_case.ranked_deals:
            validation["warnings"].append("No ranked deals provided")
        
        # Check ranked deals
        for ranked_deal in test_case.ranked_deals:
            if ranked_deal.score < 0 or ranked_deal.score > 1:
                validation["warnings"].append(f"Invalid similarity score: {ranked_deal.score}")
            
            if ranked_deal.rank < 1:
                validation["warnings"].append(f"Invalid rank: {ranked_deal.rank}")
        
        # Check category
        if test_case.category not in self.get_categories():
            validation["warnings"].append(f"Unknown category: {test_case.category}")
        
        # Check difficulty
        if test_case.difficulty not in self.get_difficulty_levels():
            validation["warnings"].append(f"Unknown difficulty: {test_case.difficulty}")
        
        return validation
