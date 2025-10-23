# Context Engineering Demo — Kavak Hackathon

## 🧱 Componentes principales

- **Personas sintéticas** (`profiles/sample_*.json`): 10 perfiles cubren las 4 cohortes (vocal feliz / vocal enojado / no vocal feliz / no vocal enojado) con historias, tickets y señales transaccionales.
- **Agentes LLM**:
  - `context_engineering/planner.py`: PlannerAgent selecciona estrategia (plantilla, tono, objetivos, triggers).
  - `context_engineering/conversation.py`: ProactiveConversationAgent orquesta diálogo 1:1 con el persona agent.
  - `context_engineering/agents_factory.py`: genera agentes-persona a partir de cada JSON.
  - `app/factories/judge.py`: Judge evalúa {NPS_expected, EngagementProb, ChurnProb, AspectSentiment}.
  - `context_engineering/ltv.py`: calcula `LTV_final = LTV_og + (0.7*NPS + 0.3*Engagement) – costo_estrategia`.
  - `context_engineering/prompt_tuner.py`: meta-agente que ajusta el prompt del planner según las métricas.
- **Catálogo de estrategias** (`context_engineering/strategies.py`): 7 estrategias (Sin_Accion, Cupon_Descuento, … Upsell_Personalizado) con costo y razonamiento.
- **Experiment runner**:
  - `context_engineering/example_runner.py`: corre una iteración → produce registros `{client_id, NPS_final, LTV_final, ganancia_LTV, …}`.
  - `context_engineering/experiment.py`: ejecuta batches y devuelve DataFrame + summary.
  - `scripts/run_experiment.py`: CLI para procesar directorios, guardar CSV/Parquet y mostrar resumen.
- **UI mínima** (`app/streamlit_context.py`): muestra perfiles, ejecuta experimento, grafica ΔLTV por cliente/estrategia y llama al prompt tuner.
- **Infra**:
  - `Dockerfile` y `.dockerignore`.
  - `requirements.txt` incluye Streamlit, pandas, numpy, pydantic, openai, plotly, scipy, pyarrow.
  - `AGENTS.md` y `CONTEXT_ENGINEERING.md` documentan arquitectura y ejecución.

## 🚀 Correr el proyecto (local o VM)

1. **Variables de entorno**
   ```bash
   export OPENAI_API_KEY="sk-proj-..."
   ```

2. **Instalación local**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   export OPENAI_API_KEY="sk-proj-..."
   ```

3. **Ejecutar experimento por CLI (CSV)**
   ```bash
   python scripts/run_experiment.py profiles --output results/iter1.csv --concurrency 10
   ```
   Resultado: CSV con el schema solicitado + resumen en consola.

4. **Pipeline completo a JSON**
   ```bash
   python scripts/run_full_pipeline.py --profiles profiles --output results/run1.json --concurrency 10
   ```
   Genera `run1.json` con `{run_number, summary, records}` listo para demos o dashboards.

5. **Streamlit demo**
   ```bash
   streamlit run app/streamlit_context.py
   ```
   Abrir en `http://localhost:8501`. Ingresar la API key en el sidebar, cargar perfiles y ejecutar experimento.

## 🐳 Docker

1. **Build**
   ```bash
   docker build -t kavak-agents .
   ```

2. **Batch experiment (CLI)**
   ```bash
   docker run --rm \
     -e OPENAI_API_KEY="sk-proj-..." \
     -v $(pwd)/profiles:/app/profiles \
     -v $(pwd)/results:/app/results \
     kavak-agents \
     python scripts/run_experiment.py profiles --output results/iter1.csv --concurrency 10
   ```

   Salida:
   - `results/iter1.csv` con las columnas:
     ```
     client_id, nps_og, vocal, run_number, estrategia_intentada, mensaje_intentado,
     NPS_final, LTV_og, LTV_final, engagement, resultado, ganancia_LTV,
     costo_estrategia, reward, strategy_name, strategy_rationale, timestamp
     ```
   - Resumen en consola: número de conversaciones, ΔLTV promedio, reward promedio, mejor estrategia, etc.

3. **Streamlit (UI)**
   ```bash
   docker run --rm -p 8501:8501 \
     -e OPENAI_API_KEY="sk-proj-..." \
     -v $(pwd)/profiles:/app/profiles \
     kavak-agents \
     streamlit run app/streamlit_context.py
   ```
   Abrir `http://localhost:8501`. La app permite:
   - Visualizar perfiles simulados.
   - Ejecutar el batch (mismo pipeline que CLI).
   - Ver tabla y gráficos de ΔLTV por cliente/estrategia.
   - Generar recomendaciones del Prompt Tuner (JSON con guidelines para el planner).

## 📊 Loop de aprendizaje

1. `PlannerAgent` usa cohort summary + historial para seleccionar `StrategyPlan`.
2. `ProactiveConversationAgent` conversa con cada `CustomerAgent` (persona) hasta trigger `END`.
3. `Judge` evalúa y retorna `Score`.
4. `evaluate_conversation` computa `LTV_final`, `ganancia_LTV`, `reward`.
5. Se genera registro `{client_id, NPS_final, LTV_final, engagement, ...}`.
6. `PromptTunerAgent` consume historial de registros y sugiere ajustes (prompt_guidelines, strategy_adjustments, experiments, notes).
7. Repetir → mostrar mejora en gráfica (Δ LTV, reward, participación de cada estrategia).

## 📦 Archivos clave

```
profiles/               # JSONs de perfiles
context_engineering/
  agents_factory.py      # Crea agentes persona
  conversation.py        # Orquestador proactivo
  planner.py             # PlannerAgent con prompts
  prompt_tuner.py        # Meta agente
  prompts.py             # Prompts sistémicos
  experiment.py          # Runner batch (DataFrame)
app/
  streamlit_context.py   # UI demo
scripts/
  run_experiment.py      # CLI experimento (CSV)
  run_full_pipeline.py   # Ejecuta batch completo y guarda JSON
Dockerfile, .dockerignore
requirements.txt
```

## ✅ Estado actual

- ✅ Personas sintéticas para tests.
- ✅ Pipeline completo (planner → conversación → judge → LTV → registro).
- ✅ Outputs listos para gráficas/insights.
- ✅ Prompt tuner operativo.
- ✅ Dockerizado y documentado para VM/cloud.
- ⚙️ UI sencilla (Streamlit) para demos; se puede extender con tabs/persistencia.

Ya está listo para deploy rápido: levanta la imagen, corre el batch o abre el Streamlit para mostrar aprendizaje y ΔLTV en vivo. ¡Listo para demo de hackathon! 🚗💥
