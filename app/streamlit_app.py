"""
Streamlit UI for Kavak Customer Service Demo.
MVP with bandits + LLM evaluators.
"""
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List

# Import factories
from app.factories.persona_forge import PersonaForge
from app.factories.state_builder import StateBuilder
from app.factories.template_factory import TemplateFactory
from app.factories.responders import ResponderAgent, OutreachAgent
from app.factories.judge import Judge
from app.factories.policy_learner import ThompsonBandit, EpsilonGreedyBandit
from app.factories.prioritizer import Prioritizer
from app.factories.metrics import MetricsAggregator
from app.factories.safety import SafetyChecker, ToneValidator

from app.models import Customer, InteractionLog
from app.templates import get_template_ids
from app.scoring import compute_reward


# Page config
st.set_page_config(
    page_title="Kavak Demo - Bandits + LLM",
    page_icon="🚗",
    layout="wide"
)

# Initialize session state
if 'customers' not in st.session_state:
    st.session_state.customers = []
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'policy' not in st.session_state:
    st.session_state.policy = None
if 'iteration' not in st.session_state:
    st.session_state.iteration = 0
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False


def check_api_key():
    """Check if API key is set."""
    if not os.getenv("OPENAI_API_KEY"):
        st.sidebar.error("⚠️ OPENAI_API_KEY no configurada")
        api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.session_state.api_key_set = True
            st.rerun()
        return False
    st.session_state.api_key_set = True
    return True


def main():
    """Main Streamlit app."""
    st.title("🚗 Kavak - Demo Bandits + LLM Evaluators")
    st.markdown("**MVP**: Agentes conversacionales con aprendizaje online y evaluación automática")

    # Check API key
    if not check_api_key():
        st.stop()

    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuración")

        # Dataset size
        n_customers = st.number_input(
            "Clientes a generar",
            min_value=10,
            max_value=500,
            value=100,
            step=10
        )

        # Bandit type
        bandit_type = st.selectbox(
            "Algoritmo Bandit",
            ["Thompson Sampling", "ε-greedy"]
        )

        # Outreach top N
        outreach_n = st.number_input(
            "Top N para Outreach",
            min_value=5,
            max_value=100,
            value=20,
            step=5
        )

        # Model selection
        st.subheader("Modelos")
        available_responder_models = ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"]
        responder_model = st.selectbox(
            "Modelo Responder",
            available_responder_models,
            index=0
        )
        judge_model = st.selectbox(
            "Modelo Judge",
            ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1"],
            index=0
        )

        st.divider()

        # Actions
        if st.button("🎲 Generar Dataset", use_container_width=True):
            with st.spinner("Generando clientes..."):
                forge = PersonaForge(seed=42)
                customers = forge.generate(n=n_customers)
                st.session_state.customers = customers
                st.session_state.logs = []
                st.session_state.iteration = 0

                # Initialize policy
                template_ids = get_template_ids()
                if bandit_type == "Thompson Sampling":
                    st.session_state.policy = ThompsonBandit(arms=template_ids)
                else:
                    st.session_state.policy = EpsilonGreedyBandit(arms=template_ids, epsilon=0.1)

                st.success(f"✅ {len(customers)} clientes generados")
                st.rerun()

        if st.button("▶️ Ejecutar Iteración", use_container_width=True, disabled=len(st.session_state.customers) == 0):
            run_iteration(
                responder_model=responder_model,
                judge_model=judge_model,
                outreach_n=outreach_n
            )

        st.divider()
        st.caption(f"Iteración actual: {st.session_state.iteration}")
        st.caption(f"Total logs: {len(st.session_state.logs)}")

    # Main content - Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "💬 Conversaciones",
        "📤 Outreach",
        "📈 Métricas",
        "🎯 Recomendaciones"
    ])

    with tab1:
        show_dashboard()

    with tab2:
        show_conversations()

    with tab3:
        show_outreach()

    with tab4:
        show_metrics()

    with tab5:
        show_recommendations()


def run_iteration(responder_model: str, judge_model: str, outreach_n: int):
    """Run one iteration of the bandit algorithm."""
    with st.spinner("Ejecutando iteración..."):
        customers = st.session_state.customers
        policy = st.session_state.policy
        logs = st.session_state.logs

        # Increment iteration
        st.session_state.iteration += 1
        current_iteration = st.session_state.iteration

        # Initialize agents
        responder = ResponderAgent(model=responder_model)
        outreach = OutreachAgent(model=responder_model)
        judge = Judge(model=judge_model)
        state_builder = StateBuilder()
        template_factory = TemplateFactory()
        prioritizer = Prioritizer()
        safety = SafetyChecker()

        # Process vocal customers
        vocal_customers = [c for c in customers if c.is_vocal]
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_tasks = len(vocal_customers) + outreach_n
        completed = 0

        for customer in vocal_customers:
            status_text.text(f"Procesando vocal: {customer.customer_id}...")

            # Build context
            ctx = state_builder.build(customer)

            # Select template (arm)
            context_key = (customer.segment, customer.issue_bucket or "atencion")
            arm = policy.select(context=context_key)

            # Fill template
            draft = template_factory.fill(arm, ctx)

            # Generate response
            message = responder.run(ctx, draft)

            # Safety check
            is_safe, reason = safety.check(message)
            if not is_safe:
                # Fallback to safe message
                message = responder._fallback_message(ctx)

            # Evaluate
            score = judge.run(ctx, message)

            # Compute reward
            reward = compute_reward(score)

            # Update policy
            policy.update(arm, reward, context=context_key)

            # Log
            log = InteractionLog(
                customer_id=customer.customer_id,
                segment=customer.segment,
                issue_bucket=customer.issue_bucket or "atencion",
                arm=arm,
                message=message,
                score=score,
                reward=reward,
                iteration=current_iteration,
                interaction_type="vocal"
            )
            logs.append(log)

            completed += 1
            progress_bar.progress(completed / total_tasks)

        # Process non-vocal (outreach)
        non_vocal = [c for c in customers if not c.is_vocal]
        ranked = prioritizer.rank(non_vocal, top_n=outreach_n)

        for customer in ranked:
            status_text.text(f"Procesando outreach: {customer.customer_id}...")

            ctx = state_builder.build(customer)
            context_key = (customer.segment, customer.issue_bucket or "atencion")
            arm = policy.select(context=context_key)

            draft = template_factory.fill(arm, ctx)
            message = outreach.run(ctx, draft)

            # Safety check
            is_safe, reason = safety.check(message)
            if not is_safe:
                message = outreach._fallback_message(ctx)

            # Evaluate
            score = judge.run(ctx, message)
            reward = compute_reward(score)

            # Update policy
            policy.update(arm, reward, context=context_key)

            # Log
            log = InteractionLog(
                customer_id=customer.customer_id,
                segment=customer.segment,
                issue_bucket=customer.issue_bucket or "atencion",
                arm=arm,
                message=message,
                score=score,
                reward=reward,
                iteration=current_iteration,
                interaction_type="outreach"
            )
            logs.append(log)

            completed += 1
            progress_bar.progress(completed / total_tasks)

        progress_bar.empty()
        status_text.empty()

        st.session_state.logs = logs
        st.success(f"✅ Iteración {current_iteration} completada!")
        st.rerun()


def show_dashboard():
    """Show main dashboard."""
    st.header("📊 Dashboard Principal")

    if not st.session_state.logs:
        st.info("👆 Genera un dataset y ejecuta una iteración para ver resultados")
        return

    logs = st.session_state.logs
    metrics = MetricsAggregator.aggregate(logs)

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Interacciones", metrics['n_interactions'])
    with col2:
        st.metric("Reward Promedio", f"{metrics['avg_reward']:.3f}")
    with col3:
        st.metric("NPS Esperado", f"{metrics['avg_NPS']:.2f}")
    with col4:
        st.metric("Engagement", f"{metrics['avg_engagement']:.1%}")

    st.divider()

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Reward por Plantilla")
        if logs:
            df = pd.DataFrame([{
                'arm': log.arm,
                'reward': log.reward
            } for log in logs])
            fig = px.box(df, x='arm', y='reward', color='arm')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribución de Selección")
        arm_dist = metrics['arm_distribution']
        if arm_dist:
            fig = px.pie(
                values=list(arm_dist.values()),
                names=list(arm_dist.keys()),
                title="% Uso de cada plantilla"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Evolution over iterations
    if 'by_iteration' in metrics and metrics['by_iteration']:
        st.subheader("Evolución por Iteración")
        iter_df = pd.DataFrame(metrics['by_iteration']).T
        iter_df.reset_index(inplace=True)
        iter_df.rename(columns={'index': 'iteration'}, inplace=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=iter_df['iteration'],
            y=iter_df['reward'],
            mode='lines+markers',
            name='Reward',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=iter_df['iteration'],
            y=iter_df['NPS_expected'] / 10,  # Normalize to [0,1]
            mode='lines+markers',
            name='NPS (norm)',
            line=dict(color='green')
        ))
        fig.update_layout(
            title="Reward y NPS por Iteración",
            xaxis_title="Iteración",
            yaxis_title="Valor"
        )
        st.plotly_chart(fig, use_container_width=True)


def show_conversations():
    """Show conversation logs (vocal customers)."""
    st.header("💬 Conversaciones (Clientes Vocales)")

    if not st.session_state.logs:
        st.info("No hay conversaciones aún")
        return

    vocal_logs = [log for log in st.session_state.logs if log.interaction_type == "vocal"]

    if not vocal_logs:
        st.info("No hay conversaciones vocales aún")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        segments = list(set(log.segment for log in vocal_logs))
        selected_segment = st.selectbox("Segmento", ["Todos"] + segments)

    with col2:
        arms = list(set(log.arm for log in vocal_logs))
        selected_arm = st.selectbox("Plantilla", ["Todas"] + arms)

    with col3:
        issues = list(set(log.issue_bucket for log in vocal_logs))
        selected_issue = st.selectbox("Issue", ["Todos"] + issues)

    # Filter logs
    filtered_logs = vocal_logs
    if selected_segment != "Todos":
        filtered_logs = [log for log in filtered_logs if log.segment == selected_segment]
    if selected_arm != "Todas":
        filtered_logs = [log for log in filtered_logs if log.arm == selected_arm]
    if selected_issue != "Todos":
        filtered_logs = [log for log in filtered_logs if log.issue_bucket == selected_issue]

    st.caption(f"Mostrando {len(filtered_logs)} conversaciones")

    # Display conversations
    for log in filtered_logs[-20:]:  # Show last 20
        with st.expander(f"{log.customer_id} - {log.arm} - Reward: {log.reward:.3f}"):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("**Mensaje generado:**")
                st.write(log.message)

            with col2:
                st.markdown("**Scores:**")
                st.metric("NPS", f"{log.score.NPS_expected:.1f}/10")
                st.metric("Engagement", f"{log.score.EngagementProb:.1%}")
                st.metric("Churn", f"{log.score.ChurnProb:.1%}")
                st.caption(f"**Rationale:** {log.score.rationale}")


def show_outreach():
    """Show outreach logs (non-vocal customers)."""
    st.header("📤 Outreach Proactivo")

    if not st.session_state.logs:
        st.info("No hay outreach aún")
        return

    outreach_logs = [log for log in st.session_state.logs if log.interaction_type == "outreach"]

    if not outreach_logs:
        st.info("No hay outreach aún")
        return

    st.caption(f"Total outreach: {len(outreach_logs)}")

    # Display outreach messages
    for log in outreach_logs[-15:]:  # Show last 15
        with st.expander(f"{log.customer_id} - {log.arm} - Reward: {log.reward:.3f}"):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("**Mensaje de outreach:**")
                st.write(log.message)

            with col2:
                st.markdown("**Scores:**")
                st.metric("NPS", f"{log.score.NPS_expected:.1f}/10")
                st.metric("Engagement", f"{log.score.EngagementProb:.1%}")
                st.metric("Churn", f"{log.score.ChurnProb:.1%}")


def show_metrics():
    """Show detailed metrics."""
    st.header("📈 Métricas Detalladas")

    if not st.session_state.logs:
        st.info("No hay métricas aún")
        return

    metrics = MetricsAggregator.aggregate(st.session_state.logs)

    # By segment
    st.subheader("Por Segmento")
    if metrics['by_segment']:
        segment_df = pd.DataFrame(metrics['by_segment']).T
        st.dataframe(segment_df, use_container_width=True)

    # By template
    st.subheader("Por Plantilla (Arm)")
    if metrics['by_arm']:
        arm_df = pd.DataFrame(metrics['by_arm']).T
        st.dataframe(arm_df, use_container_width=True)

    # By issue
    st.subheader("Por Tipo de Issue")
    if metrics['by_issue']:
        issue_df = pd.DataFrame(metrics['by_issue']).T
        st.dataframe(issue_df, use_container_width=True)

    # Policy statistics
    st.subheader("Estadísticas de la Política")
    policy = st.session_state.policy
    if policy:
        stats = policy.get_statistics()
        stats_df = pd.DataFrame(stats).T
        st.dataframe(stats_df, use_container_width=True)


def show_recommendations():
    """Show insights and recommendations."""
    st.header("🎯 Recomendaciones e Insights")

    if not st.session_state.logs:
        st.info("No hay recomendaciones aún")
        return

    metrics = MetricsAggregator.aggregate(st.session_state.logs)

    st.subheader("📌 Top Insights")
    insights = metrics.get('insights', [])

    if insights:
        for insight in insights:
            st.info(insight)
    else:
        st.write("Ejecuta más iteraciones para generar insights")

    # Contextual recommendations
    st.subheader("🎯 Recomendaciones Contextuales")

    policy = st.session_state.policy
    if policy and hasattr(policy, 'get_all_contexts'):
        contexts = policy.get_all_contexts()

        if contexts:
            st.write(f"Se han explorado {len(contexts)} combinaciones de (segmento, issue_bucket)")

            # Show best arm per context
            for context in contexts[:10]:  # Show top 10
                segment, issue = context
                stats = policy.get_statistics(context=context)

                # Find best arm
                best_arm = max(stats, key=lambda k: stats[k]['mean'])
                best_mean = stats[best_arm]['mean']

                st.markdown(f"**{segment} + {issue}**: usar `{best_arm}` (reward medio: {best_mean:.3f})")


if __name__ == "__main__":
    main()
