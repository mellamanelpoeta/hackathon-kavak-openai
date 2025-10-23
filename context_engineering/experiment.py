"""
Batch experiment runner for context-engineering agents.

Provides utilities to execute multiple customer conversations,
aggregate results, and persist them for analytics/visualization.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .example_runner import load_profiles, run_iteration


def run_experiment(
    *,
    profiles_dir: Path,
    output_path: Optional[Path] = None,
    max_profiles: Optional[int] = None,
    run_number: int = 1,
    strategy_attempt_id: int = 1,
    message_attempt_id: int = 1,
    tone: str = "empático-directo",
    max_turns: int = 3,
    end_triggers: Optional[List[str]] = None,
    judge_model: str = "gpt-4.1-mini",
    planner_model: str = "gpt-4.1",
    api_key: Optional[str] = None,
    concurrency: int = 10,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Execute conversations for all profiles in the directory and return results as DataFrame.

    Args:
        profiles_dir: path to directory with JSON profiles.
        output_path: optional file path to persist results (CSV or Parquet).
        max_profiles: cap number of profiles processed (dense sampling).
        run_number: iteration number for online learning loop.
        strategy_attempt_id: identifier of current strategy experiment.
        message_attempt_id: identifier of current prompt variant.
        tone: default tone override (used if planner generates prompt without tone).
        max_turns: maximum number of turns per conversation.
        end_triggers: optional list of endings recognized by orchestrator.
        judge_model: model used by Judge agent.
        planner_model: model used by Planner agent.
        api_key: OpenAI API key (falls back to env if None).

    Returns:
        Tuple of (results DataFrame, summary dict)
    """
    profiles = load_profiles(Path(profiles_dir))
    if max_profiles:
        profiles = profiles[:max_profiles]

    records = run_iteration(
        profiles,
        tone=tone,
        max_turns=max_turns,
        end_triggers=end_triggers,
        judge_model=judge_model,
        run_number=run_number,
        strategy_attempt_id=strategy_attempt_id,
        message_attempt_id=message_attempt_id,
        api_key=api_key,
        concurrency=concurrency,
        planner_model=planner_model,
        verbose=verbose,
    )

    if not records:
        df = pd.DataFrame(columns=[
            "client_id",
            "nps_og",
            "vocal",
            "satisfecho",
            "cohort_label",
            "run_number",
            "estrategia_intentada",
            "mensaje_intentado",
            "NPS_final",
            "LTV_og",
            "LTV_final",
            "engagement",
            "resultado",
            "ganancia_LTV",
            "costo_estrategia",
            "reward",
            "strategy_name",
            "strategy_rationale",
            "issue_bucket",
            "mini_story",
            "channel_pref",
            "timestamp",
        ])
        summary = {"n_conversations": 0, "ltv_gain_avg": 0.0, "best_strategy": None}
        return df, summary

    df = pd.DataFrame(records)

    summary = _summarize_results(df)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() in {".parquet", ".pq"}:
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)

    return df, summary


def _summarize_results(df: pd.DataFrame) -> Dict:
    """Compute high-level summary metrics from experiment DataFrame."""
    avg_ltv_gain = float(df["ganancia_LTV"].mean())
    avg_reward = float(df["reward"].mean())
    best_strategy_name = None

    if "strategy_name" in df.columns and not df.empty:
        strategy_performance = df.groupby("strategy_name")["ganancia_LTV"].mean()
        if not strategy_performance.empty:
            best_strategy_name = strategy_performance.idxmax()

    cohort_perf = None
    if "cohort_label" in df.columns:
        cohort_perf = (
            df.groupby("cohort_label")["ganancia_LTV"]
            .mean()
            .to_dict()
        )

    summary = {
        "n_conversations": int(len(df)),
        "ltv_gain_avg": avg_ltv_gain,
        "reward_avg": avg_reward,
        "best_strategy": best_strategy_name,
        "cohort_ltv_gain": cohort_perf,
    }

    return summary


__all__ = ["run_experiment"]
