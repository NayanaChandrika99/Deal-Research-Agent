# ABOUTME: DSPy signature and module definitions for deal reasoning workflows.
# ABOUTME: Encapsulates input/output fields so MIPRO optimization can compile prompts.

from __future__ import annotations

import dspy


class DealReasonerSignature(dspy.Signature):
    """Analyze historical PE deals to identify precedents and extract insights."""

    query: str = dspy.InputField(desc="New deal opportunity description from user")
    candidate_deals: str = dspy.InputField(desc="JSON-formatted candidate deals with metadata")

    precedents: str = dspy.OutputField(desc="JSON list of relevant precedent deals")
    playbook_levers: str = dspy.OutputField(desc="JSON list of common value-creation levers")
    risk_themes: str = dspy.OutputField(desc="JSON list of common risk themes")
    narrative_summary: str = dspy.OutputField(desc="Executive narrative summarizing insights")


class DealReasonerModule(dspy.Module):
    """DSPy module that wraps the reasoning signature with optional Chain-of-Thought."""

    def __init__(self):
        super().__init__()
        self.reason = dspy.ChainOfThought(DealReasonerSignature)

    def forward(self, query: str, candidate_deals: str):
        result = self.reason(query=query, candidate_deals=candidate_deals)
        return dspy.Prediction(
            precedents=result.precedents,
            playbook_levers=result.playbook_levers,
            risk_themes=result.risk_themes,
            narrative_summary=result.narrative_summary
        )
