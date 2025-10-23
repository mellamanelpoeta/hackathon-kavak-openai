"""
Persistence utilities for experiment history, strategy insights, and prompt overrides.
"""
from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

RESULTS_DIR = Path("results")
HISTORY_FILE = RESULTS_DIR / "history.jsonl"
STRATEGY_FILE = RESULTS_DIR / "strategy_metrics.json"
PROMPT_OVERRIDES_FILE = RESULTS_DIR / "prompt_overrides.json"

DEFAULT_OVERRIDES = {
    "notes": "",
    "planner": {"global": [], "cohorts": {}},
    "initiative": {"global": [], "cohorts": {}},
    "history": [],
}

DEFAULT_STRATEGY_INSIGHTS = {
    "generated_at": None,
    "overall": {},
    "best_by_cohort": {},
    "strategy_stats": {},
}


def _ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def append_history_records(records: Iterable[Dict[str, Any]] | pd.DataFrame, source_path: Optional[str] = None) -> None:
    """
    Append conversation records to the persistent history file (JSONL).
    """
    if isinstance(records, pd.DataFrame):
        if records.empty:
            return
        rows = records.to_dict(orient="records")
    else:
        rows = list(records)
        if not rows:
            return

    _ensure_results_dir()
    timestamp = datetime.now().isoformat()
    with HISTORY_FILE.open("a", encoding="utf-8") as handle:
        for row in rows:
            payload = dict(row)
            payload.setdefault("recorded_at", timestamp)
            if source_path:
                payload.setdefault("source_file", source_path)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_history_df() -> pd.DataFrame:
    """
    Load the persisted history as a DataFrame.
    """
    if not HISTORY_FILE.exists():
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    with HISTORY_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if {"run_number", "client_id", "timestamp"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["run_number", "client_id", "timestamp"], keep="last")
    return df


def update_strategy_metrics() -> Dict[str, Any]:
    """
    Compute and persist strategy insights from the aggregated history.

    Returns the computed insights dictionary.
    """
    history_df = load_history_df()
    insights = deepcopy(DEFAULT_STRATEGY_INSIGHTS)
    insights["generated_at"] = datetime.now().isoformat()

    if history_df.empty or "strategy_name" not in history_df.columns:
        _ensure_results_dir()
        with STRATEGY_FILE.open("w", encoding="utf-8") as handle:
            json.dump(insights, handle, ensure_ascii=False, indent=2)
        return insights

    def _safe_mean(series: pd.Series) -> float:
        return float(series.mean()) if not series.empty else 0.0

    agg_columns = [col for col in ["ganancia_LTV", "reward", "costo_estrategia"] if col in history_df.columns]
    if agg_columns:
        strategy_stats = (
            history_df.groupby("strategy_name")[agg_columns]
            .mean()
            .reset_index()
        )
        if "ganancia_LTV" in strategy_stats.columns:
            strategy_stats = strategy_stats.rename(columns={"ganancia_LTV": "ltv_gain_avg"})
        if "reward" in strategy_stats.columns:
            strategy_stats = strategy_stats.rename(columns={"reward": "reward_avg"})
        if "costo_estrategia" in strategy_stats.columns:
            strategy_stats = strategy_stats.rename(columns={"costo_estrategia": "cost_avg"})
        strategy_stats["samples"] = history_df.groupby("strategy_name")["client_id"].count().values
        insights["strategy_stats"] = strategy_stats.set_index("strategy_name").to_dict(orient="index")

        if not strategy_stats.empty and "ltv_gain_avg" in strategy_stats.columns:
            best_row = strategy_stats.loc[strategy_stats["ltv_gain_avg"].idxmax()]
            insights["overall"] = {
                "strategy": best_row.name if hasattr(best_row, "name") else best_row["strategy_name"],
                "metrics": best_row.to_dict(),
            }

    if {"cohort_label", "strategy_name"}.issubset(history_df.columns):
        best_by_cohort: Dict[str, Dict[str, Any]] = {}
        for cohort, cohort_df in history_df.groupby("cohort_label"):
            if cohort_df.empty:
                continue
            cohort_stats = (
                cohort_df.groupby("strategy_name")[agg_columns]
                .mean()
                .reset_index()
            )
            if "ganancia_LTV" in cohort_stats.columns:
                cohort_stats = cohort_stats.rename(columns={"ganancia_LTV": "ltv_gain_avg"})
            if "reward" in cohort_stats.columns:
                cohort_stats = cohort_stats.rename(columns={"reward": "reward_avg"})
            if "costo_estrategia" in cohort_stats.columns:
                cohort_stats = cohort_stats.rename(columns={"costo_estrategia": "cost_avg"})
            cohort_stats["samples"] = cohort_df.groupby("strategy_name")["client_id"].count().values
            if not cohort_stats.empty and "ltv_gain_avg" in cohort_stats.columns:
                best_row = cohort_stats.loc[cohort_stats["ltv_gain_avg"].idxmax()]
                best_by_cohort[cohort] = {
                    "strategy": best_row["strategy_name"],
                    "metrics": best_row.to_dict(),
                }
        insights["best_by_cohort"] = best_by_cohort

    _ensure_results_dir()
    with STRATEGY_FILE.open("w", encoding="utf-8") as handle:
        json.dump(insights, handle, ensure_ascii=False, indent=2)
    return insights


def load_strategy_insights() -> Dict[str, Any]:
    if STRATEGY_FILE.exists():
        with STRATEGY_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data
    return deepcopy(DEFAULT_STRATEGY_INSIGHTS)


def load_prompt_overrides() -> Dict[str, Any]:
    if PROMPT_OVERRIDES_FILE.exists():
        with PROMPT_OVERRIDES_FILE.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return deepcopy(DEFAULT_OVERRIDES)


def save_prompt_overrides(overrides: Dict[str, Any]) -> None:
    _ensure_results_dir()
    with PROMPT_OVERRIDES_FILE.open("w", encoding="utf-8") as handle:
        json.dump(overrides, handle, ensure_ascii=False, indent=2)


def merge_prompt_guidance(overrides: Dict[str, Any], guidance: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge Prompt Tuner guidance into the stored overrides structure.
    """
    merged = deepcopy(DEFAULT_OVERRIDES)

    # Start from existing overrides
    def _extend_unique(target: List[str], items: Iterable[str]) -> None:
        for item in items:
            if item and item not in target:
                target.append(item)

    # Deep merge existing overrides first
    for key in ["notes", "history"]:
        if key in overrides:
            merged[key] = deepcopy(overrides[key])
    for section in ["planner", "initiative"]:
        merged[section]["global"] = list(overrides.get(section, {}).get("global", []))
        merged[section]["cohorts"] = deepcopy(overrides.get(section, {}).get("cohorts", {}))

    timestamp = datetime.now().isoformat()

    notes = guidance.get("notes")
    if notes:
        merged["notes"] = notes
        merged.setdefault("history", []).append({"timestamp": timestamp, "notes": notes})

    guidelines = guidance.get("prompt_guidelines") or []
    _extend_unique(merged["planner"]["global"], guidelines)
    _extend_unique(merged["initiative"]["global"], guidelines)

    adjustments = guidance.get("strategy_adjustments") or []
    for adj in adjustments:
        cohort = adj.get("cohort")
        strategy = adj.get("strategy")
        action = adj.get("action", "mantener")
        if not cohort or not strategy:
            continue
        text = f"{action.capitalize()} la estrategia {strategy}"
        cohort_planner = merged["planner"]["cohorts"].setdefault(cohort, [])
        cohort_initiative = merged["initiative"]["cohorts"].setdefault(cohort, [])
        _extend_unique(cohort_planner, [text])
        _extend_unique(cohort_initiative, [text])

    experiments = guidance.get("experiments") or []
    if experiments:
        merged.setdefault("history", []).append({"timestamp": timestamp, "experiments": experiments})

    return merged


__all__ = [
    "append_history_records",
    "load_history_df",
    "update_strategy_metrics",
    "load_strategy_insights",
    "load_prompt_overrides",
    "save_prompt_overrides",
    "merge_prompt_guidance",
]
