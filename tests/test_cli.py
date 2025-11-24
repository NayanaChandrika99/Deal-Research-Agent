"""CLI tests for dealgraph-agent."""

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from dealgraph.cli.main import app
from dealgraph.data.schemas import DealReasoningOutput, Precedent

runner = CliRunner()


def _fake_log():
    return {
        "query": "software",
        "tool_calls": [],
        "reasoning_output": DealReasoningOutput(
            precedents=[Precedent(deal_id="deal_1", name="Deal 1", similarity_reason="match")],
            playbook_levers=["lever"],
            risk_themes=["risk"],
            narrative_summary="Narrative",
        ),
    }


def test_cli_query_prints_output(monkeypatch):
    fake_log = _fake_log()
    monkeypatch.setattr(
        "dealgraph.cli.main.run_agent",
        lambda *args, **kwargs: DealAgentLog(**fake_log),
    )
    result = runner.invoke(app, ["Test"])
    assert result.exit_code == 0
    assert "Narrative" in result.output


def test_cli_query_writes_output(tmp_path, monkeypatch):
    fake_log = _fake_log()
    monkeypatch.setattr(
        "dealgraph.cli.main.run_agent",
        lambda *args, **kwargs: DealAgentLog(**fake_log),
    )
    output_file = tmp_path / "agent.json"
    result = runner.invoke(app, ["Test", "--output", str(output_file)])
    assert result.exit_code == 0
    assert output_file.exists()
    data = json.loads(output_file.read_text())
    assert data["query"] == "software"


class DealAgentLog:
    def __init__(self, query, tool_calls, reasoning_output):
        self.query = query
        self.tool_calls = []
        self.reasoning_output = reasoning_output

    def model_dump(self):
        return {
            "query": self.query,
            "tool_calls": self.tool_calls,
            "reasoning_output": self.reasoning_output.model_dump(),
        }
