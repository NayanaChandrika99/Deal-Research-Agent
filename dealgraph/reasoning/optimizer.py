# ABOUTME: CLI entry point for running DSPy prompt optimization.
# ABOUTME: Wraps DSPyOptimizer utilities so engineers can update prompts offline.

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .dspy.optimizer import optimize_deal_reasoner

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Optimize deal reasoner prompts via DSPy.")
    parser.add_argument("--baseline", default="v1", help="Baseline prompt version to optimize from.")
    parser.add_argument("--num-candidates", type=int, default=10, help="Number of candidate prompts.")
    parser.add_argument("--max-evaluations", type=int, default=100, help="Maximum MIPRO evaluations.")
    parser.add_argument("--min-improvement", type=float, default=0.05, help="Required composite-score lift.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write optimization metadata JSON (defaults to stdout).",
    )
    parser.add_argument(
        "--prompt-output",
        type=Path,
        help="Override path for the saved prompt file (defaults to prompts/deal_reasoner/vX_optimized.json).",
    )
    args = parser.parse_args()

    result = optimize_deal_reasoner(
        baseline_version=args.baseline,
        num_candidate_prompts=args.num_candidates,
        max_evaluations=args.max_evaluations,
        output_path=str(args.prompt_output) if args.prompt_output else None,
        min_improvement=args.min_improvement,
    )

    serialized = json.dumps({k: v for k, v in result.items() if k != "optimized_program"}, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
        logger.info("Optimization metadata written to %s", args.output)
    else:
        print(serialized)


if __name__ == "__main__":
    main()
