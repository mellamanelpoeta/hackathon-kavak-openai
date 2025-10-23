"""
Minimal Streamlit UI for context-engineering experiments.
Allows running batches of proactive conversations and inspecting metrics.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

from context_engineering.experiment import run_experiment
from context_engineering.prompt_tuner import PromptTunerAgent
from context_engineering.example_runner import load_profiles


DEFAULT_PROFILES_DIR = Path("profiles")


def init_session_state():
    st.session_state.setdefault("profiles", [])
    st.session_state.setdefault("results_df", pd.DataFrame())
    st.session_state.setdefault("summary", {})
    st.session_state.setdefault("history_notes", "")


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


def run_experiment_ui(
    profiles_dir: Path,
    max_profiles: int,
    run_number: int,
    strategy_attempt: int,
    message_attempt: int,
):
    with st.spinner("Ejecutando conversaciones..."):
        df, summary = run_experiment(
            profiles_dir=profiles_dir,
            max_profiles=max_profiles,
            run_number=run_number,
            strategy_attempt_id=strategy_attempt,
            message_attempt_id=message_attempt,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    st.session_state.results_df = df
    st.session_state.summary = summary
    st.success(f"Experimento completado: {summary.get('n_conversations', 0)} conversaciones")


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
    df = st.session_state.results_df
    if df.empty:
        st.info("Ejecuta el experimento para ver resultados.")
        return

    summary = st.session_state.summary
    st.subheader("Resumen del experimento")
    col1, col2, col3 = st.columns(3)
    col1.metric("Conversaciones", summary.get("n_conversations", 0))
    col2.metric("Δ LTV promedio", f"${summary.get('ltv_gain_avg', 0):.2f}")
    col3.metric("Reward promedio", f"{summary.get('reward_avg', 0):.3f}")

    st.markdown("**Tabla de resultados**")
    st.dataframe(df)

    st.markdown("**Ganancia de LTV por cliente**")
    fig = px.bar(
        df,
        x="client_id",
        y="ganancia_LTV",
        color="strategy_name",
        title="Ganancia LTV por estrategia",
        labels={"ganancia_LTV": "Δ LTV", "client_id": "Cliente"},
    )
    st.plotly_chart(fig, use_container_width=True)

    if "cohort_ltv_gain" in summary and summary["cohort_ltv_gain"]:
        st.markdown("**Ganancia promedio por cohorte**")
        coh_df = pd.DataFrame(
            [{"cohorte": k, "ganancia_LTV": v} for k, v in summary["cohort_ltv_gain"].items()]
        )
        st.bar_chart(coh_df.set_index("cohorte"))

    st.markdown("**Distribución por cohorte**")
    cohort_fig = px.pie(
        df,
        names="cohort_label",
        values="ganancia_LTV",
        title="Participación de ganancia LTV por cohorte",
    )
    st.plotly_chart(cohort_fig, use_container_width=True)


def show_prompt_guidance():
    df = st.session_state.results_df
    if df.empty:
        return

    if st.button("🔁 Generar recomendaciones de prompt"):
        tuner = PromptTunerAgent(api_key=os.getenv("OPENAI_API_KEY"))
        guidance = tuner.run(
            run_records=df.to_dict(orient="records"),
            current_prompt_notes=st.session_state.history_notes,
        )
        st.session_state.history_notes = guidance.get("notes", "")
        st.json(guidance)

    if st.session_state.history_notes:
        st.info(f"Notas actuales del prompt: {st.session_state.history_notes}")


def main():
    st.set_page_config(page_title="Context Engineering Demo", layout="wide")
    init_session_state()

    st.title("🚀 Kavak Context Engineering Demo")
    st.caption("Simula conversaciones proactivas, mide LTV y optimiza prompts con multi-agentes.")

    profiles_dir = Path(
        st.sidebar.text_input("Directorio de perfiles", str(DEFAULT_PROFILES_DIR))
    )
    max_profiles = st.sidebar.slider("Clientes a procesar", 1, 50, 10)
    run_number = st.sidebar.number_input("Run number", min_value=1, value=1)
    strategy_attempt = st.sidebar.number_input("Strategy attempt", min_value=1, value=1)
    message_attempt = st.sidebar.number_input("Message attempt", min_value=1, value=1)

    st.sidebar.divider()
    api_key_ok = ensure_api_key()

    if st.sidebar.button("📂 Cargar perfiles"):
        load_profiles_ui(profiles_dir)

    if st.sidebar.button("▶️ Ejecutar experimento", disabled=not api_key_ok):
        run_experiment_ui(
            profiles_dir=profiles_dir,
            max_profiles=max_profiles,
            run_number=run_number,
            strategy_attempt=strategy_attempt,
            message_attempt=message_attempt,
        )

    st.header("Perfiles simulados")
    show_profiles_table()

    st.header("Resultados")
    show_results()
    show_prompt_guidance()


if __name__ == "__main__":
    main()
