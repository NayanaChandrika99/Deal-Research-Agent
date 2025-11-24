++ plans/execplan_5_2_cli_interface.md
# ExecPlan 5.2 – CLI Interface

This plan obeys `.agent/PLANS.md`. It stands alone as the reference for delivering Phase 5.2.

## Purpose / Big Picture

With `run_agent()` available, we want a CLI entrypoint (e.g., `dealgraph-agent query "..."`) that:
1. Accepts a free-text query and optional flags (prompt version, max results, JSON output file).
2. Calls the orchestrator.
3. Prints structured results (JSON + narrative) with friendly formatting.

After this plan, running `python -m dealgraph.cli.main query "US healthcare roll-up"` (or via console script) will execute the full pipeline and show the reasoning output.

## Progress

- [x] (2025-02-14 22:40Z) CLI module implemented (Typer root command).
- [x] (2025-02-14 22:42Z) Console script entry added to pyproject.
- [x] (2025-02-14 22:50Z) Tests covering CLI invocation (Typer CliRunner + mocks).

## Surprises & Discoveries

- Typer defaults to a single root command when there’s only one command defined; thus the CLI syntax is simply `dealgraph-agent "query text"` (no subcommand). Tests were adjusted accordingly.

## Decision Log

- Decision: Keep CLI as a root command rather than requiring a subcommand; simpler UX and aligns with Typer’s behavior when there is only one command.  
  Date/Author: 2025-02-14 / Claude

## Outcomes & Retrospective

- CLI is runnable (`python -m dealgraph.cli.main "query text"`) and installed as `dealgraph-agent`. It prints JSON + narrative and can optionally save the AgentLog; tests guarantee output formatting and error handling via mocks.

## Context and Orientation

- CLI modules typically live at `dealgraph/cli/main.py` per SPEC §7.
- We can use Typer (preferred) or argparse. Typer gives nice help output.
- Orchestrator function: `dealgraph.agent.orchestrator.run_agent`.
- Output format:
  - Print JSON representation of `reasoning_output` (pretty-printed).
  - Then print narrative summary.
- Flags:
  - `--prompt-version` (default `latest`).
  - `--max-results` (default 10).
  - `--output` (optional JSON file to write the AgentLog).
- Need to handle exceptions gracefully (display error message, exit code 1).

## Plan of Work

1. **CLI module**
   - Create `dealgraph/cli/main.py`:
     - Use Typer.
     - Command `query`: arguments `query: str`; options `--prompt-version`, `--max-results`, `--output`.
     - Within handler:
       - Call `run_agent`.
       - Print `reasoning_output` as JSON (preferrably using `.model_dump_json(indent=2)`).
       - Print narrative summary below (with separators).
       - If `--output` provided, write entire `AgentLog` (JSON).
     - Provide `if __name__ == "__main__": app()` hook.

2. **Entry point**
   - Update `pyproject.toml` `[project.scripts]` (or equivalent) to add `dealgraph-agent = "dealgraph.cli.main:app"`.

3. **Tests**
   - Add `tests/test_cli.py` (or extend existing) using Typer’s `CliRunner` (from `typer.testing`).
   - Mock `run_agent` to avoid hitting real pipeline.
   - Verify:
     - CLI prints JSON + narrative.
     - `--output` writes file.
     - Errors propagate (mock run_agent to raise).

4. **Docs & validation**
   - Mention CLI in README (if time; otherwise note as TODO).
   - Run `PYTHONPATH=. .venv/bin/pytest tests/test_cli.py`.
   - Manual smoke run (e.g., `python -m dealgraph.cli.main query "Software roll-up" --dry-run` using mocks if necessary).

## Concrete Steps

1. Implement CLI module as described.
2. Add script entry to pyproject.
3. Create tests with Typer CliRunner.
4. Run targeted pytest.
5. (Optional) update README to show usage snippet.

## Validation and Acceptance

- CLI command works interactively (mocked or real).
- Tests pass and cover success/error cases.
- `dealgraph-agent --help` shows expected options.

## Idempotence and Recovery

- CLI writes outputs only if the user passes `--output`; ensures no accidental overwrite unless intended.
- Tests isolate file writes to tmp directories.

## Interfaces and Dependencies

- `dealgraph.cli.main.app` (Typer).
- CLI command: `dealgraph-agent query "US healthcare roll-up" --prompt-version latest --max-results 5 --output results/run.json`.
