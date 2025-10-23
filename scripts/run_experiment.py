#!/usr/bin/env python3
"""
CLI wrapper to execute context-engineering experiments and persist results.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from context_engineering.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run context-engineering experiment batch")
    parser.add_argument("profiles_dir", type=Path, help="Directory containing customer JSON profiles")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save results (CSV or Parquet)",
    )
    parser.add_argument("--max-profiles", type=int, help="Limit number of profiles to process")
    parser.add_argument("--run-number", type=int, default=1, help="Iteration/run identifier")
    parser.add_argument("--strategy-attempt", type=int, default=1, help="Strategy attempt id")
    parser.add_argument("--message-attempt", type=int, default=1, help="Message/prompt variant id")
    parser.add_argument("--tone", type=str, default="empático-directo", help="Default tone")
    parser.add_argument("--max-turns", type=int, default=3, help="Max conversation turns")
    parser.add_argument(
        "--end-trigger",
        action="append",
        dest="end_triggers",
        help="Register additional end triggers (can be set multiple times)",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-4.1-mini", help="Model for Judge agent")
    parser.add_argument("--planner-model", type=str, default="gpt-4.1", help="Model for Planner agent")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df, summary = run_experiment(
        profiles_dir=args.profiles_dir,
        output_path=args.output,
        max_profiles=args.max_profiles,
        run_number=args.run_number,
        strategy_attempt_id=args.strategy_attempt,
        message_attempt_id=args.message_attempt,
        tone=args.tone,
        max_turns=args.max_turns,
        end_triggers=args.end_triggers,
        judge_model=args.judge_model,
        planner_model=args.planner_model,
    )

    print("=== Experiment Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nPreview of results:")
    if df.empty:
        print("(No records)")
    else:
        print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
