# ABOUTME: CLI entry point exposing run_agent functionality.
# ABOUTME: Provides `dealgraph-agent query "..."` command.

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..agent.orchestrator import run_agent, AgentOrchestratorError

app = typer.Typer(help="DealGraph Agent CLI")


@app.command()
def query(
    q: str = typer.Argument(..., help="Free-form query describing the deal scenario."),
    prompt_version: str = typer.Option("latest", help="Prompt version to use."),
    max_results: int = typer.Option(10, help="Maximum number of deals to retrieve."),
    output: Path = typer.Option(None, help="Optional file to write AgentLog as JSON."),
):
    """Run the DealGraph agent end-to-end."""
    try:
        log = run_agent(q, max_results=max_results, prompt_version=prompt_version)
    except AgentOrchestratorError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(log.reasoning_output.model_dump_json(indent=2))
    typer.echo("\n--- Narrative ---\n")
    typer.echo(log.reasoning_output.narrative_summary)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(log.model_dump(), indent=2))
        typer.echo(f"\nAgent log saved to {output}")


if __name__ == "__main__":
    app()
