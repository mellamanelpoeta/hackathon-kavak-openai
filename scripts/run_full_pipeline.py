#!/usr/bin/env python3
"""
End-to-end runner for the context-engineering experiment.

Usage:
    python scripts/run_full_pipeline.py \
        --profiles profiles \
        --output results/run1.json \
        --run-number 1 \
        --concurrency 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path when executed as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from context_engineering.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute full context-engineering pipeline")
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("personas_output"),
        help="Directory containing persona JSON files (supports nested folders)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/run1.json"),
        help="Path to store JSON results",
    )
    parser.add_argument("--max-profiles", type=int, help="Optional cap on number of profiles")
    parser.add_argument("--run-number", type=int, default=1, help="Learning iteration identifier")
    parser.add_argument("--strategy-attempt", type=int, default=1, help="Strategy attempt id")
    parser.add_argument("--message-attempt", type=int, default=1, help="Message/prompt attempt id")
    parser.add_argument("--tone", type=str, default="empático-directo", help="Default tone")
    parser.add_argument("--max-turns", type=int, default=3, help="Max conversation turns")
    parser.add_argument(
        "--end-trigger",
        action="append",
        dest="end_triggers",
        help="Extra end triggers (can be repeated)",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-4.1-mini", help="Judge model")
    parser.add_argument("--planner-model", type=str, default="gpt-4.1", help="Planner model")
    parser.add_argument("--concurrency", type=int, default=10, help="Parallel conversations")
    parser.add_argument("--seed", type=int, help="Random seed for persona sampling")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Process profiles sequentially without shuffling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df, summary = run_experiment(
        profiles_dir=args.profiles,
        output_path=None,
        max_profiles=args.max_profiles,
        run_number=args.run_number,
        strategy_attempt_id=args.strategy_attempt,
        message_attempt_id=args.message_attempt,
        tone=args.tone,
        max_turns=args.max_turns,
        end_triggers=args.end_triggers,
        judge_model=args.judge_model,
        planner_model=args.planner_model,
        concurrency=args.concurrency,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_payload = {
        "run_number": args.run_number,
        "summary": summary,
        "records": df.to_dict(orient="records"),
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results_payload['records'])} records to {output_path}")
    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
