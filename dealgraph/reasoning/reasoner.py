# ABOUTME: Deal reasoning module using LLM analysis for generating insights and narratives.
# ABOUTME: Implements naive baseline reasoning without DSPy optimization.

from typing import List, Dict, Any, Optional, Tuple
import json
import logging
from ..data.schemas import RankedDeal, DealReasoningOutput, Precedent, CandidateDeal
from .prompts import load_prompt, DEAL_REASONER_NAIVE_PROMPT
from .llm_client import get_llm_client, LLMClientError


class ReasoningError(Exception):
    """Raised when reasoning operations fail."""
    pass


def deal_reasoner(
    query: str,
    ranked_deals: List[RankedDeal],
    max_deals: int = 10,
    prompt_version: str = "latest"
) -> DealReasoningOutput:
    """
    Analyze ranked deals and generate structured reasoning output.
    
    This is the main reasoning function that combines search results with LLM analysis
    to produce structured insights about deal precedents, playbooks, and risks.
    
    Args:
        query: User's original search query
        ranked_deals: Ranked list of candidate deals
        max_deals: Maximum number of deals to include in reasoning
        prompt_version: Version of prompt to use ("latest", "v1", "v2", etc.)
        
    Returns:
        DealReasoningOutput with structured analysis
        
    Raises:
        ReasoningError: If reasoning fails
    """
    try:
        llm_client = get_llm_client()
        
        # Limit deals to top K
        top_deals = ranked_deals[:max_deals]
        
        # Format deals for prompt
        deals_block = _format_deals_for_prompt(top_deals)
        
        system_prompt, user_prompt = _resolve_prompts(prompt_version, query, deals_block)
        
        # Call LLM
        response = llm_client.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,  # Lower temperature for more consistent reasoning
            max_tokens=2500
        )
        
        # Parse and validate response
        reasoning_output = _parse_reasoning_output(response, top_deals)
        
        logging.info(
            f"Generated reasoning for query '{query[:50]}...' "
            f"with {len(reasoning_output.precedents)} precedents"
        )
        
        return reasoning_output
        
    except LLMClientError as e:
        raise ReasoningError(f"LLM reasoning failed: {e}")
    except Exception as e:
        raise ReasoningError(f"Unexpected reasoning error: {e}")


def analyze_deals_with_naive_prompt(
    query: str,
    ranked_deals: List[RankedDeal],
    max_deals: int = 10
) -> DealReasoningOutput:
    """
    Convenience function for using the naive prompt specifically.
    
    Args:
        query: User's search query
        ranked_deals: Ranked candidate deals
        max_deals: Maximum deals to analyze
        
    Returns:
        DealReasoningOutput
    """
    return deal_reasoner(query, ranked_deals, max_deals, prompt_version="v1")


def _resolve_prompts(prompt_version: str, query: str, deals_block: str) -> Tuple[str, str]:
    """Resolve the system and user prompts for the requested version."""
    version_arg: Optional[str] = None
    if prompt_version == "v1":
        system_prompt = _get_deal_reasoner_system_prompt()
        template = DEAL_REASONER_NAIVE_PROMPT
    else:
        version_arg = None if prompt_version == "latest" else prompt_version
        try:
            prompt_data = load_prompt("deal_reasoner", version_arg)
        except FileNotFoundError as exc:
            if prompt_version == "latest":
                system_prompt = _get_deal_reasoner_system_prompt()
                template = DEAL_REASONER_NAIVE_PROMPT
            else:
                raise ReasoningError(f"Prompt version '{prompt_version}' not found") from exc
        else:
            metadata = prompt_data["metadata"]
            system_prompt = getattr(metadata, "system_prompt", metadata.description)
            template = prompt_data["content"]
    
    return system_prompt, _format_prompt_template(template, query, deals_block)


def _format_deals_for_prompt(deals: List[RankedDeal]) -> str:
    """Format ranked deals into text block for LLM prompt."""
    if not deals:
        return "No candidate deals found."
    
    deals_text = []
    for i, candidate in enumerate(_iter_candidates(deals), 1):
        deal = candidate.deal
        snippets = candidate.snippets
        
        # Format deal information
        deal_info = [
            f"- Deal {i}: {deal.name}",
            f"  ID: {deal.id}",
            f"  Sector: {deal.sector_id}",
            f"  Region: {deal.region_id}",
            f"  Platform: {'Yes' if deal.is_platform else 'No'}",
            f"  Status: {deal.status}",
            f"  Description: {deal.description}",
            f"  Text Similarity: {candidate.text_similarity:.3f}",
            f"  Graph Features: {candidate.graph_features}",
        ]
        
        # Add snippets if available
        if snippets:
            snippet_texts = [f"'{snippet.text}' (source: {snippet.source})" for snippet in snippets[:2]]
            deal_info.append(f"  Key Snippets: {', '.join(snippet_texts)}")
        
        deals_text.append('\n'.join(deal_info))
    
    return '\n\n'.join(deals_text)


def _get_deal_reasoner_system_prompt() -> str:
    """Get the system prompt for deal reasoning."""
    return """You are a private-equity deal research assistant. Your expertise is in analyzing historical deals to identify precedents, extract playbook levers, and assess risk themes.

You will be given a new investment opportunity and a set of candidate historical deals that were retrieved based on similarity. Your task is to:

1. **Identify Precedents**: Select the most relevant historical deals that serve as good precedents for the new opportunity. Explain why each is a precedent.

2. **Extract Playbook Levers**: Identify common value-creation strategies, operational improvements, or strategic approaches used across the precedents.

3. **Analyze Risk Themes**: Extract common risk patterns, challenges, or mitigation approaches from the precedents.

4. **Generate Narrative**: Provide an executive-level summary that synthesizes the analysis into actionable insights.

Focus on patterns, strategies, and lessons that would be relevant for evaluating the new opportunity. Be specific and evidence-based in your analysis."""

def _format_prompt_template(template: str, query: str, deals_block: str) -> str:
    """Safely substitute the query and deals block into a prompt template."""
    return template.replace("{query}", query).replace("{deals_block}", deals_block)


def _parse_reasoning_output(
    response: Dict[str, Any],
    deals: List[RankedDeal]
) -> DealReasoningOutput:
    """Parse LLM response into DealReasoningOutput schema."""
    try:
        candidate_deals = list(_iter_candidates(deals))

        # Extract precedents
        precedents = []
        precedents_data = response.get("precedents", [])
        
        for precedent_data in precedents_data:
            if not isinstance(precedent_data, dict):
                continue
                
            deal_id = precedent_data.get("deal_id")
            name = precedent_data.get("name", "Unknown")
            similarity_reason = precedent_data.get("similarity_reason", "")
            
            # Validate that the deal_id exists in our ranked deals
            if any(candidate.deal.id == deal_id for candidate in candidate_deals):
                precedents.append(Precedent(
                    deal_id=deal_id,
                    name=name,
                    similarity_reason=similarity_reason
                ))
        
        # Extract playbook levers
        playbook_levers = []
        levers_data = response.get("playbook_levers", [])
        if isinstance(levers_data, list):
            playbook_levers = [str(lever) for lever in levers_data if lever]
        
        # Extract risk themes
        risk_themes = []
        risks_data = response.get("risk_themes", [])
        if isinstance(risks_data, list):
            risk_themes = [str(risk) for risk in risks_data if risk]
        
        # Extract narrative summary
        narrative_summary = str(response.get("narrative_summary", ""))
        
        # Validate required fields
        if not precedents:
            logging.warning("No valid precedents found in LLM response")
        if not playbook_levers:
            logging.warning("No playbook levers found in LLM response")
        if not risk_themes:
            logging.warning("No risk themes found in LLM response")
        if not narrative_summary:
            logging.warning("No narrative summary found in LLM response")
        
        return DealReasoningOutput(
            precedents=precedents,
            playbook_levers=playbook_levers,
            risk_themes=risk_themes,
            narrative_summary=narrative_summary
        )
        
    except Exception as e:
        logging.error(f"Failed to parse reasoning output: {e}")
        logging.error(f"Response data: {response}")
        raise ReasoningError(f"Invalid reasoning output format: {e}")


def _iter_candidates(deals: List[RankedDeal]) -> List[CandidateDeal]:
    """Yield CandidateDeal objects from ranked deals."""
    normalized: List[CandidateDeal] = []
    for deal in deals:
        if isinstance(deal, CandidateDeal):
            normalized.append(deal)
        elif hasattr(deal, "candidate"):
            normalized.append(deal.candidate)
        else:
            raise ReasoningError("Invalid deal entry supplied to reasoner")
    return normalized


def validate_reasoning_output(output: DealReasoningOutput) -> bool:
    """Validate that reasoning output has expected structure and content."""
    # Check required fields
    if not hasattr(output, 'precedents') or not isinstance(output.precedents, list):
        return False
    if not hasattr(output, 'playbook_levers') or not isinstance(output.playbook_levers, list):
        return False
    if not hasattr(output, 'risk_themes') or not isinstance(output.risk_themes, list):
        return False
    if not hasattr(output, 'narrative_summary') or not isinstance(output.narrative_summary, str):
        return False
    
    # Check that we have some content
    has_content = (
        len(output.precedents) > 0 or
        len(output.playbook_levers) > 0 or
        len(output.risk_themes) > 0 or
        len(output.narrative_summary.strip()) > 0
    )
    
    if not has_content:
        logging.warning("Reasoning output has no meaningful content")
        return False
    
    return True


def extract_reasoning_metrics(output: DealReasoningOutput) -> Dict[str, Any]:
    """Extract metrics from reasoning output for evaluation."""
    return {
        "num_precedents": len(output.precedents),
        "num_playbook_levers": len(output.playbook_levers),
        "num_risk_themes": len(output.risk_themes),
        "narrative_length": len(output.narrative_summary),
        "has_precedents": len(output.precedents) > 0,
        "has_playbook": len(output.playbook_levers) > 0,
        "has_risks": len(output.risk_themes) > 0,
        "has_narrative": len(output.narrative_summary.strip()) > 0
    }


# DSPy Integration Functions

def dspy_optimize_prompt(
    baseline_version: str = "v1",
    num_candidate_prompts: int = 10,
    max_evaluations: int = 100,
    min_improvement: float = 0.05
) -> Dict[str, Any]:
    """
    Optimize prompt using DSPy MIPRO algorithm.
    
    Args:
        baseline_version: Version of baseline prompt to optimize from
        num_candidate_prompts: Number of candidate prompts to generate
        max_evaluations: Maximum number of optimization evaluations  
        min_improvement: Minimum improvement required to adopt new prompt
        
    Returns:
        Dictionary with optimization results and metrics
        
    Raises:
        ReasoningError: If optimization fails
    """
    try:
        from .dspy import get_dspy_optimizer
        
        optimizer = get_dspy_optimizer()
        optimization_result = optimizer.optimize_deal_reasoner_prompt(
            baseline_prompt_version=baseline_version,
            num_candidate_prompts=num_candidate_prompts,
            max_evaluations=max_evaluations,
            min_improvement=min_improvement
        )
        
        # Save optimized prompt if it meets threshold
        if optimization_result.get("meets_threshold", False):
            optimizer.save_optimized_prompt(optimization_result)
            logging.info(
                f"DSPy optimization successful with {optimization_result['improvement_score']:.3f} improvement"
            )
        else:
            logging.warning(
                f"DSPy optimization below threshold ({optimization_result['improvement_score']:.3f} < {min_improvement})"
            )
        
        return optimization_result
        
    except Exception as e:
        raise ReasoningError(f"DSPy optimization failed: {e}")


def dspy_evaluate_performance(
    reasoning_output: Dict[str, Any],
    reference_data: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Evaluate reasoning output performance using DSPy evaluator.
    
    Args:
        reasoning_output: Output from deal reasoning
        reference_data: Optional reference data for ground truth
        
    Returns:
        Dictionary with performance metrics (0.0 to 1.0)
    """
    try:
        from .dspy import get_dspy_optimizer
        
        optimizer = get_dspy_optimizer()
        return optimizer.evaluator.evaluate_reasoning_output(reasoning_output, reference_data)
        
    except Exception as e:
        logging.warning(f"DSPy evaluation failed: {e}")
        return extract_reasoning_metrics(reasoning_output)


def compare_reasoning_prompts(
    query: str,
    ranked_deals: List[RankedDeal],
    prompt_versions: List[str] = ["v1", "v2"]
) -> Dict[str, Any]:
    """
    Compare reasoning performance between different prompt versions.
    
    Args:
        query: User's search query
        ranked_deals: Ranked candidate deals
        prompt_versions: List of prompt versions to compare
        
    Returns:
        Dictionary with comparison results
    """
    try:
        from .dspy import get_dspy_optimizer
        
        optimizer = get_dspy_optimizer()
        results = {}
        
        for version in prompt_versions:
            try:
                result = deal_reasoner(query, ranked_deals, prompt_version=version)
                results[version] = extract_reasoning_metrics(result)
            except Exception as e:
                results[version] = {"error": str(e)}
        
        # Calculate improvements
        baseline_version = prompt_versions[0]
        if baseline_version in results and "error" not in results[baseline_version]:
            baseline_metrics = results[baseline_version]
            
            for version in prompt_versions[1:]:
                if version in results and "error" not in results[version]:
                    current_metrics = results[version]
                    improvements = {}
                    
                    for metric, baseline_value in baseline_metrics.items():
                        if isinstance(baseline_value, (int, float)) and metric in current_metrics:
                            current_value = current_metrics[metric]
                            improvement = current_value - baseline_value
                            improvements[metric] = improvement
                    
                    results[f"{version}_improvements"] = improvements
        
        return results
        
    except Exception as e:
        raise ReasoningError(f"Prompt comparison failed: {e}")


def dspy_get_optimization_history() -> List[Dict[str, Any]]:
    """Get history of DSPy optimization attempts."""
    try:
        from .dspy import get_dspy_optimizer
        optimizer = get_dspy_optimizer()
        return optimizer.get_optimization_history()
    except Exception as e:
        logging.warning(f"Failed to get optimization history: {e}")
        return []


def dspy_rollback_to_baseline(baseline_version: str = "v1") -> bool:
    """Rollback to baseline prompt if optimization failed."""
    try:
        from .dspy import get_dspy_optimizer
        optimizer = get_dspy_optimizer()
        return optimizer.rollback_to_baseline(baseline_version)
    except Exception as e:
        logging.error(f"Rollback failed: {e}")
        return False
