"""
Streamlit UI for launching context-engineering experiments and reviewing results.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from context_engineering.experiment import run_experiment
from context_engineering.prompt_tuner import PromptTunerAgent
from context_engineering.example_runner import load_profiles
from context_engineering.persistence import (
    load_history_df,
    load_strategy_insights,
    load_prompt_overrides,
    save_prompt_overrides,
    merge_prompt_guidance,
)


DEFAULT_PROFILES_DIR = Path("personas_output")
RESULTS_DIR = Path("results")


def init_session_state():
    st.session_state.setdefault("profiles", [])
    st.session_state.setdefault("results_df", pd.DataFrame())
    st.session_state.setdefault("summary", {})
    st.session_state.setdefault("history_notes", "")
    st.session_state.setdefault("last_output_path", None)
    st.session_state.setdefault("history_df", load_history_df())
    st.session_state.setdefault("strategy_insights", load_strategy_insights())
    st.session_state.setdefault("prompt_overrides", load_prompt_overrides())
    if not st.session_state.history_notes:
        st.session_state.history_notes = st.session_state.prompt_overrides.get("notes", "")


def ensure_api_key():
    if os.getenv("OPENAI_API_KEY"):
        return True
    api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        return True
    st.sidebar.warning("Ingresa tu OPENAI_API_KEY para ejecutar el experimento.")
    return False


def load_profiles_ui(profiles_path: Path):
    try:
        profiles = load_profiles(profiles_path)
        st.session_state.profiles = profiles
        st.success(f"{len(profiles)} perfiles cargados desde {profiles_path}")
    except FileNotFoundError:
        st.error(f"No se encontró el directorio {profiles_path}.")
    except Exception as exc:
        st.error(f"Error al cargar perfiles: {exc}")


def save_results(df: pd.DataFrame, summary: dict, run_number: int, output_name: str | None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_name.strip() if output_name else f"run_{run_number}_{timestamp}.json"
    if not filename.lower().endswith(".json"):
        filename += ".json"
    path = RESULTS_DIR / filename
    payload = {
        "run_number": run_number,
        "summary": summary,
        "records": df.to_dict(orient="records"),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def load_results_file(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        df = pd.DataFrame(payload.get("records", []))
        summary = payload.get("summary", {})
        st.session_state.results_df = df
        st.session_state.summary = summary
        st.session_state.last_output_path = path
        if not df.empty:
            df_with_source = df.copy()
            df_with_source["source_file"] = str(path)
            df_with_source["run_saved_at"] = datetime.now().isoformat()
            st.session_state.history_df = (
                pd.concat([st.session_state.history_df, df_with_source], ignore_index=True)
                .drop_duplicates(subset=["run_number", "client_id", "timestamp"], keep="last")
            )
        st.session_state.history_df = load_history_df()
        st.session_state.strategy_insights = load_strategy_insights()
        st.session_state.prompt_overrides = load_prompt_overrides()
        st.success(f"Resultados cargados desde {path}")
    except FileNotFoundError:
        st.error(f"No se encontró el archivo {path}")
    except Exception as exc:
        st.error(f"Error al cargar resultados: {exc}")


def run_experiment_ui(
    profiles_dir: Path,
    max_profiles: int,
    run_number: int,
    strategy_attempt: int,
    message_attempt: int,
    concurrency: int,
    max_turns: int,
    seed: int | None,
    output_name: str | None,
):
    with st.spinner("Ejecutando conversaciones..."):
        df, summary = run_experiment(
            profiles_dir=profiles_dir,
            max_profiles=max_profiles,
            run_number=run_number,
            strategy_attempt_id=strategy_attempt,
            message_attempt_id=message_attempt,
            concurrency=concurrency,
            verbose=False,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_turns=max_turns,
            seed=seed,
        )
    st.session_state.results_df = df
    st.session_state.summary = summary
    output_path = save_results(df, summary, run_number, output_name)
    st.session_state.last_output_path = output_path
    df_with_source = df.copy()
    df_with_source["source_file"] = str(output_path)
    df_with_source["run_saved_at"] = datetime.now().isoformat()
    st.session_state.history_df = (
        pd.concat([st.session_state.history_df, df_with_source], ignore_index=True)
        .drop_duplicates(subset=["run_number", "client_id", "timestamp"], keep="last")
    )
    st.session_state.history_df = load_history_df()
    st.session_state.strategy_insights = load_strategy_insights()
    st.session_state.prompt_overrides = load_prompt_overrides()
    st.success(
        f"Experimento completado: {summary.get('n_conversations', 0)} conversaciones. "
        f"Guardado en {output_path}"
    )


def show_profiles_table():
    if not st.session_state.profiles:
        st.info("Carga perfiles para visualizar.")
        return
    df = pd.json_normalize(st.session_state.profiles)
    display_cols = [
        "customer_id",
        "cohort.vocal",
        "cohort.satisfied",
        "persona.name",
        "persona.bio",
        "purchase.vehicle",
        "purchase.price",
        "risk_signals.churn_est",
        "risk_signals.ltv_apriori",
    ]
    available_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available_cols])


def show_results():
    current_df = st.session_state.results_df
    history_df = st.session_state.history_df

    if current_df.empty and history_df.empty:
        st.info("Ejecuta el experimento para ver resultados.")
        return

    options = ["Último run", "Historial acumulado"]
    dataset_option = st.radio("Dataset base", options, horizontal=True)
    if dataset_option == "Historial acumulado" and not history_df.empty:
        base_df = history_df
    else:
        base_df = current_df if not current_df.empty else history_df

    if base_df.empty:
        st.info("No hay datos para mostrar con el dataset seleccionado.")
        return

    filtered_df = render_filters(base_df)
    tabs = st.tabs(["Resumen", "Conversaciones", "Métricas", "Prompt Tuner"])

    with tabs[0]:
        show_summary_tab(filtered_df, st.session_state.summary)
    with tabs[1]:
        show_conversations_tab(filtered_df)
    with tabs[2]:
        show_metrics_tab(filtered_df)
    with tabs[3]:
        show_prompt_guidance(filtered_df)


def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("🎛️ Filtros")

    runs = sorted(df["run_number"].dropna().unique())
    cohorts = sorted(df.get("cohort_label", pd.Series(dtype=str)).dropna().unique())
    strategies = sorted(df.get("strategy_name", pd.Series(dtype=str)).dropna().unique())

    col1, col2, col3 = st.columns(3)
    selected_runs = col1.multiselect("Run number", runs, default=runs)
    selected_cohorts = col2.multiselect("Cohorte", cohorts, default=cohorts)
    selected_strategies = col3.multiselect("Estrategia", strategies, default=strategies)

    mask = pd.Series(True, index=df.index)
    if selected_runs:
        mask &= df["run_number"].isin(selected_runs)
    if selected_cohorts:
        mask &= df["cohort_label"].isin(selected_cohorts)
    if selected_strategies:
        mask &= df["strategy_name"].isin(selected_strategies)

    filtered = df[mask].copy()
    st.caption(f"Mostrando {len(filtered)} conversaciones filtradas")
    return filtered


def show_summary_tab(df: pd.DataFrame, overall_summary: Dict):
    if df.empty:
        st.info("No hay resultados para mostrar con los filtros seleccionados.")
        return

    cost_series = df["costo_estrategia"] if "costo_estrategia" in df.columns else pd.Series(0.0, index=df.index)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Conversaciones", len(df))
    col2.metric("Δ LTV promedio", f"${df['ganancia_LTV'].mean():.2f}")
    col3.metric("Reward promedio", f"{df['reward'].mean():.3f}")
    col4.metric("Costo promedio", f"${cost_series.mean():.2f}")

    st.markdown("**Tabla de resultados filtrados**")
    st.dataframe(df, use_container_width=True)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar CSV filtrado", data=csv_data, file_name="experiment_results.csv")

    if "strategy_name" in df.columns and "ganancia_LTV" in df.columns and not df.empty:
        strat_perf = df.groupby("strategy_name")["ganancia_LTV"].mean().sort_values(ascending=False)
        best_strategy = strat_perf.index[0]
        st.markdown(
            f"**Mejor estrategia (datos filtrados):** `{best_strategy}` — Δ LTV {strat_perf.iloc[0]:.2f}"
        )

        if "cohort_label" in df.columns:
            rows = []
            for cohort, cohort_df in df.groupby("cohort_label"):
                perf = cohort_df.groupby("strategy_name")["ganancia_LTV"].mean().sort_values(ascending=False)
                if not perf.empty:
                    rows.append(
                        {
                            "Cohorte": cohort,
                            "Estrategia": perf.index[0],
                            "Δ LTV": perf.iloc[0],
                        }
                    )
            if rows:
                st.markdown("**Mejor estrategia por cohorte (filtrado)**")
                st.table(pd.DataFrame(rows))

    if overall_summary:
        st.markdown("**Resumen general del último experimento**")
        st.json(overall_summary)
        if overall_summary.get("best_strategy"):
            st.markdown(
                f"**Mejor estrategia (último run):** `{overall_summary['best_strategy']}`"
            )
        cohort_best = overall_summary.get("best_strategy_by_cohort") or {}
        if cohort_best:
            st.markdown("**Mejor estrategia por cohorte (último run)**")
            st.table(pd.DataFrame([
                {"Cohorte": cohort, "Estrategia": strategy}
                for cohort, strategy in cohort_best.items()
            ]))

    insights = st.session_state.get("strategy_insights", {})
    if insights:
        overall_insight = insights.get("overall", {})
        if overall_insight.get("strategy"):
            st.markdown(
                "**Mejor estrategia global (histórico):** "
                f"`{overall_insight['strategy']}` — Δ LTV {overall_insight.get('metrics', {}).get('ltv_gain_avg', 0):.2f}"
            )
        cohort_history = insights.get("best_by_cohort", {})
        if cohort_history:
            st.markdown("**Mejor estrategia por cohorte (histórico)**")
            hist_rows = []
            for cohort, data in cohort_history.items():
                metrics = data.get("metrics", {})
                hist_rows.append(
                    {
                        "Cohorte": cohort,
                        "Estrategia": data.get("strategy"),
                        "Δ LTV": metrics.get("ltv_gain_avg", 0),
                        "Reward": metrics.get("reward_avg", 0),
                    }
                )
            st.table(pd.DataFrame(hist_rows))


def show_conversations_tab(df: pd.DataFrame):
    if df.empty:
        st.info("Sin conversaciones para mostrar.")
        return

    st.markdown("**Conversaciones**")
    sorted_df = df.sort_values(["run_number", "client_id"]).reset_index(drop=True)
    for _, row in sorted_df.iterrows():
        transcript = row.get("transcript")
        if not isinstance(transcript, list):
            continue
        header = (
            f"Run {row['run_number']} · {row['client_id']} · {row['strategy_name']} · "
            f"Reward {row['reward']:.3f}"
        )
        with st.expander(header):
            st.markdown(
                f"**Cohorte:** {row.get('cohort_label', 'N/D')} | "
                f"**Costo:** ${row.get('costo_estrategia', 0):.0f} | "
                f"**NPS reportado:** {row.get('nps_score_reported', 'N/D')}"
            )
            for turn in transcript:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role == "context":
                    st.caption(f"Contexto inicial: {content}")
                    continue
                role_label = role.capitalize()
                st.markdown(f"**{role_label}:** {content}")
            if row.get("NPS_comment"):
                st.caption(f"Comentario NPS: {row['NPS_comment']}")


def show_metrics_tab(df: pd.DataFrame):
    if df.empty:
        st.info("Sin métricas para mostrar.")
        return

    st.markdown("**Evolución por run**")
    available_cols = [col for col in ["ganancia_LTV", "reward", "costo_estrategia"] if col in df.columns]
    agg = df.groupby("run_number")[available_cols].mean().reset_index()
    rename_map = {
        "ganancia_LTV": "Δ LTV promedio",
        "reward": "Reward promedio",
        "costo_estrategia": "Costo promedio",
    }
    agg.rename(columns=rename_map, inplace=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["run_number"],
        y=agg["Δ LTV promedio"],
        mode="lines+markers",
        name="Δ LTV promedio",
    ))
    fig.add_trace(go.Scatter(
        x=agg["run_number"],
        y=agg["Reward promedio"],
        mode="lines+markers",
        name="Reward promedio",
        yaxis="y2",
    ))
    if "Costo promedio" in agg.columns:
        fig.add_trace(go.Bar(
            x=agg["run_number"],
            y=agg["Costo promedio"],
            name="Costo promedio",
            opacity=0.3,
            yaxis="y3",
        ))
    fig.update_layout(
        xaxis_title="Run",
        yaxis=dict(title="Δ LTV"),
        yaxis2=dict(title="Reward", overlaying="y", side="right"),
        yaxis3=dict(title="Costo", overlaying="y", side="left", position=0.05, showgrid=False),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Desempeño por estrategia**")
    strat_cols = [col for col in ["ganancia_LTV", "reward", "costo_estrategia"] if col in df.columns]
    strat = (
        df.groupby("strategy_name")[strat_cols]
        .mean()
        .reset_index()
        .sort_values(by="ganancia_LTV", ascending=False)
    )
    st.dataframe(strat, use_container_width=True)

    insights = st.session_state.get("strategy_insights", {})
    stats = insights.get("strategy_stats", {})
    if stats:
        st.markdown("**Histórico acumulado por estrategia**")
        st.dataframe(pd.DataFrame.from_dict(stats, orient="index"), use_container_width=True)

    cohort_history = insights.get("best_by_cohort", {})
    if cohort_history:
        st.markdown("**Estrategia recomendada por cohorte (histórico)**")
        hist_rows = []
        for cohort, data in cohort_history.items():
            metrics = data.get("metrics", {})
            hist_rows.append(
                {
                    "Cohorte": cohort,
                    "Estrategia": data.get("strategy"),
                    "Δ LTV": metrics.get("ltv_gain_avg", 0),
                    "Reward": metrics.get("reward_avg", 0),
                }
            )
        st.table(pd.DataFrame(hist_rows))


def show_prompt_guidance(df: pd.DataFrame):
    if df.empty:
        st.info("Ejecuta un experimento para generar recomendaciones.")
        return

    prompt_overrides = st.session_state.get("prompt_overrides", load_prompt_overrides())

    if st.button("🔁 Generar recomendaciones de prompt"):
        tuner = PromptTunerAgent(api_key=os.getenv("OPENAI_API_KEY"))
        guidance = tuner.run(
            run_records=df.to_dict(orient="records"),
            current_prompt_notes=st.session_state.history_notes,
        )
        prompt_overrides = merge_prompt_guidance(prompt_overrides, guidance)
        save_prompt_overrides(prompt_overrides)
        st.session_state.prompt_overrides = prompt_overrides
        st.session_state.history_notes = prompt_overrides.get("notes", "")
        st.session_state.strategy_insights = load_strategy_insights()
        st.session_state.history_df = load_history_df()
        st.json(guidance)

    if st.session_state.history_notes:
        st.info(f"Notas actuales del prompt: {st.session_state.history_notes}")

    st.markdown("**Overrides vigentes**")
    st.json(st.session_state.get("prompt_overrides", {}))


def main():
    st.set_page_config(page_title="Context Engineering Demo", layout="wide")
    init_session_state()

    st.title("🚀 Kavak Context Engineering Demo")
    st.caption("Simula conversaciones proactivas, mide LTV y optimiza prompts con multi-agentes.")

    profiles_dir = Path(
        st.sidebar.text_input("Directorio de perfiles", str(DEFAULT_PROFILES_DIR))
    )
    max_profiles = st.sidebar.slider("Clientes a procesar", 1, 100, 20)
    run_number = st.sidebar.number_input("Run number", min_value=1, value=1)
    strategy_attempt = st.sidebar.number_input("Strategy attempt", min_value=1, value=1)
    message_attempt = st.sidebar.number_input("Message attempt", min_value=1, value=1)
    max_turns = st.sidebar.slider("Turnos del agente proactivo", 3, 10, 4)
    concurrency = st.sidebar.slider("Conversaciones en paralelo", 1, 20, 8)
    seed_input = st.sidebar.text_input("Seed aleatoria (opcional)", value="")
    try:
        seed_value = int(seed_input) if seed_input.strip() else None
    except ValueError:
        seed_value = None
        st.sidebar.warning("Seed inválida; se usará aleatoriedad por defecto.")
    output_name = st.sidebar.text_input("Nombre de archivo de salida (opcional)", "")

    st.sidebar.divider()
    api_key_ok = ensure_api_key()

    if st.sidebar.button("📂 Cargar perfiles"):
        load_profiles_ui(profiles_dir)

    saved_files = sorted(RESULTS_DIR.glob("*.json"))
    options = ["(Selecciona archivo)"] + [f.name for f in saved_files]
    selected_file = st.sidebar.selectbox("Resultados guardados", options)
    if st.sidebar.button("📥 Cargar resultados guardados", disabled=selected_file == "(Selecciona archivo)"):
        load_results_file(RESULTS_DIR / selected_file)

    if st.sidebar.button("▶️ Ejecutar experimento", disabled=not api_key_ok):
        run_experiment_ui(
            profiles_dir=profiles_dir,
            max_profiles=max_profiles,
            run_number=run_number,
            strategy_attempt=strategy_attempt,
            message_attempt=message_attempt,
            concurrency=concurrency,
            max_turns=max_turns,
            seed=seed_value,
            output_name=output_name,
        )

    st.header("Perfiles simulados")
    show_profiles_table()

    st.header("Resultados")
    show_results()

    if st.session_state.last_output_path:
        st.caption(f"Último archivo guardado: {st.session_state.last_output_path}")


if __name__ == "__main__":
    main()
