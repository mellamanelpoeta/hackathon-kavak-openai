# Kavak Hackathon — Bandits + LLM Agents (Agents API)

**Meta**: Demo en <6 h que auto-aprende (bandits + evaluador LLM) y muestra lift en NPS simulado, churn/engagement y qué estrategia (plantilla) conviene por cohorte. Todo in-memory sobre Streamlit y OpenAI Agents API.

---

## 0) Plan por fases (6 h disponibles)

| Fase | Duración | Objetivo | Entregables clave |
| --- | --- | --- | --- |
| **F0: Kickoff & setup** | 0:00–0:30 | Alinear demo, preparar entorno, definir owners | `requirements` instalados, API key seteada, backlog priorizado |
| **F1: Datos & estructura** | 0:30–2:00 | Simulador + modelos + session state + UI esqueleto | `PersonaForge`, `StateBuilder`, `SessionState`, layout Streamlit con tabs/botones |
| **F2: Agents & bandits** | 2:00–3:30 | Integrar OpenAI Agents (Responder/Outreach/Judge) + policy learner | `AgentsRunner`, prompts finales, loop vocal/outreach con reward |
| **F3: Métricas & dashboard** | 3:30–4:30 | Métricas agregadas, charts, insights, safety guardrails | Dashboard con KPIs, tablas por cohorte, safety en producción |
| **F4: Hardening & demo script** | 4:30–5:15 | QA rápida, fallback flows, baseline compare, guion de demo | Dataset baseline, pantalla “caso individual”, script narrativo |
| **F5: Infra & Polish** | 5:15–6:00 | Dockerizar, artefactos demo-ready, slides | Dockerfile, script batch, 5 slides cristalinas, dry-run completo |

**Paralelización sugerida**  
1. **Persona & datos**: mantiene `PersonaForge`, `Prioritizer`, datasets.  
2. **Agents & prompts**: define `Responder/Outreach/Judge`, safety.  
3. **Bandits & métricas**: policy learner + KPIs/plots.  
4. **Streamlit owner**: integra piezas y cuida UX.

---

## 0) MVP express (🎯 listo en 60 min)

### Objetivo
1. Simular 4 perfiles: vocal-feliz, vocal-enojado, no-vocal-feliz, no-vocal-enojado (50 c/u → 200 clientes).
2. Para vocales: agente Responder + bandit (Thompson o ε-greedy) elige 1 de 5 plantillas y aprende con evaluador LLM `{NPS_expected, EngagementProb, ChurnProb}`.
3. Para no-vocales: agente Outreach prioriza top-30 vía heurística + bandit y aprende igual.
4. Streamlit muestra métricas: lift NPS, churn/engagement, top plantillas x segmento/issue, logs de mensajes y racionales.

### Supuestos
- Dataset sintético generado por PersonaForge (mini historia, señales transaccionales, mensaje inicial si es vocal).
- Recompensa: `R = clip0_1(0.6*NPS + 0.3*(EngProb*10) - 0.3*(ChurnProb*10))`.
- Evaluador (`Judge`) retorna JSON estricto sin humanos en el loop.
- Política contextual: `context = (segmento, issue_bucket)` → actualiza priors online.
- Sin persistencia, todo en session state.

### Arquitectura mínima
```
PersonaForge → StateBuilder → TemplateFactory
                 ↓                        ↓
            Policy Learner -------→ Responder / Outreach (Agents API)
                                          ↓
                                   Judge (Agents API JSON)
                                          ↓
                             compute_reward → policy.update
```

Tabs Streamlit: Simulación | Conversaciones | Outreach | Métricas | Recomendaciones.

### Flujo para demo (5 mins)
1. Click “Generar dataset sintético”.
2. Click “Ejecutar iteración” → procesa vocales + outreach top-30.
3. Revisar Dashboard (KPIs, gráficos, insights) y Conversaciones/Outreach (mensajes + scores).
4. Repetir 1–3 veces para mostrar aprendizaje real-time.

---

## 1) Caso de negocio (pitch rápido)

**Problema**  
- Vocals concentran señales; non-vocals = zona ciega.  
- Outreach wrong-time desperdicia el “one-shot”.  
- Imposible A/B manual a escala.

**Solución (MVP)**  
Agentes que simulan clientes + agentes operativos (Responder/Outreach) que aprenden online con bandits, guiados por un evaluador LLM (Judge) que estima NPS/churn/engagement y aspect sentiment.

**KPIs demo**  
- Δ NPS esperado vs baseline (plantilla uniforme).  
- Churn esperado ↓.  
- Engagement esperado ↑.  
- Precisión de selección (expected engagement) en no-vocales top-N.

**Impacto esperado (fase real)**  
- +2–5 pts NPS en 4–8 semanas.  
- −10–20 % churn conversacional.  
- Menos retrabajo vía rutas correctas (mecánico/finanzas).

**Costos & tiempo**  
- Hoy: prototipo in-memory sin integración.  
- Pronto: conectar CRM/tickets (2–4 semanas).  
- Coste variable = tokens (hackathon: ilimitado). Caching/batching luego.

**Riesgos & mitigación**  
- Sesgo auto-selección → logging + estratificación + holdouts.  
- Sobre-ajuste plantillas → Thompson/ε mantiene exploración.  
- Tono inadecuado → guardrails (Safety Checker) + blacklist.

---

## 2) Hoja de ruta (post-MVP)

**Semana 1–2**  
- Plug CRM histórico (NPS, tickets, pagos).  
- Features adicionales (canal, respuesta previa, taxonomía de issues).  
- Plantillas con micro-acciones (cupón %, derivación explícita, SLA).

**Semana 3–4**  
- Aspect-Based Sentiment (ABS) propio por {finanzas, mecánica, logística, atención}.  
- Off-policy eval (IPS/DR) para testear nuevas políticas sin full deployment.  
- Personalización via embeddings (RAG liviano por cliente).

**Semana 5–8**  
- Contextual bandits (LinUCB/NeuralUCB).  
- Multi-objetivo con restricciones (cost-to-serve).  
- Causal uplift para outreach en no-vocales.  
- Medir ΔLTV/cohort + costo de beneficios.

---

## 3) Streamlit (estructura + hooks)

**Tabs & contenido**
- **Simulación**: sliders (n clientes, %enojados, %vocales), botón dataset.  
- **Conversaciones**: tabla expandible (mensaje, score Judge, plantilla, rationale).  
- **Outreach**: ranking top-30 no-vocales (score + mensaje).  
- **Métricas**:  
  - Línea: reward/NPS por iteración.  
  - Barras: churn/engagement por segmento.  
  - Dataframes: resumen por arm/segment/issue.  
- **Recomendaciones**: insights auto-generados (“Para NVE-mecánica → Empático + SLA 48h”).

**Session state**: `customers`, `logs`, `policy`, `iteration`, `baseline_logs?`.

**Key buttons**: Generar dataset | Ejecutar iteración (procesa vocal+outreach).  
Opcional: botones separados (solo vocales / solo outreach) si se quiere granularidad.

---

## 4) Plantillas & Prompts (Agents API)

**Templates (máx 5)**
1. `empatiko` — Empático breve  
2. `tecnico` — Técnico solución  
3. `cupon` — Compensación/cupón  
4. `escalar` — Escalación interna  
5. `seguimiento` — Seguimiento proactivo  

Slots: `[saludo, reconocimiento_issue/diagnóstico, accion/accion_tecnica, next_step/sla/next_touch, firma]`.  
Guardrails: no culpar, disculpa breve cuando aplique, SLA claro, sin promesas imposibles, límite de compensación.

**System prompts**  
- **Responder**: “Eres agente Kavak… usa UNA plantilla, rellena slots con contexto, no culpes, ofrece next step claro, escalar cuando aplique, máx 150 palabras, responde solo texto final.”  
- **Outreach**: “Agente outreach one-shot… personaliza con recencia/monto/issue, CTA claro, máx 120 palabras, solo texto.”  
- **Judge**: “Evalúa mensaje dado contexto, devuelve JSON estricto `{NPS_expected 0-10, EngagementProb 0-1, ChurnProb 0-1, AspectSentiment (finanzas/mecanica/logistica/atencion -1..1), rationale <=280 chars}`; penaliza frialdad, valora claridad + SLA.”

---

## 5) Datos sintéticos (CSV mínimo)

```
customer_id, segment{VF,VE,NVF,NVE}, is_vocal, last_purchase_days,
price, issues_flag{0/1}, past_NPS, first_message, channel_pref,
churn_risk_est, issue_bucket, mini_story
```

Opcionales úteis: `messages[]`, `sentiment_hist[]` (para versiones extendidas).  
PersonaForge genera: mini bio, historia, primer mensaje (si vocal), issue bucket, banderas de enojo.

Prioritizer heurístico:  
`S = 0.5*issues_flag + 0.3*normalize(price) + 0.2*normalize(1/last_purchase_days)`  
→ rank top-N no-vocales.

---

## 6) Loop de aprendizaje (pseudo)

```python
for customer in vocal_customers:
    ctx = state_builder.build(customer)
    arm = policy.select((ctx.segment, ctx.issue_bucket))
    draft = template_factory.fill(arm, ctx)
    msg = responder.run(ctx, draft)              # Agents API → texto
    is_safe, _ = safety.check(msg)
    if not is_safe:
        msg = responder._fallback_message(ctx)
    score = judge.run(ctx, msg)                  # Agents API → JSON (Score)
    reward = compute_reward(score)               # clip 0..1
    policy.update(arm, reward, (ctx.segment, ctx.issue_bucket))
    logs.append(InteractionLog(...))

non_vocals = prioritizer.rank(no_vocal_customers)[:30]
repetir mismo patrón con OutreachAgent
```

---

## 7) 5 Slides “cristalinos”
1. **Problema** – Vocals = mucha señal; non-vocals = ciegos. Outreach y soporte pierden potencia.  
2. **Solución** – Simulación + agentes operativos con bandits + evaluador LLM → loop prueba-mide-aprende.  
3. **Demo & Métricas** – NPS↑, churn↓, engagement↑; gráfico evolución + tabla mejores plantillas.  
4. **Valor Kavak** – Priorización inteligente, tono perfecto desde el primer mensaje, rutas correctas → menos retrabajo/costo.  
5. **Next Steps & ROI** – De bandits simples a contextuales, ABS, uplift. Meta: +2–5 NPS, −10–20% churn en 4–8 semanas.

---

## 8) Checklist técnico (Agents API, ~60 min)

> Tips: 3 personas en paralelo → (1) Streamlit, (2) prompts/agents, (3) bandit + scoring.  
> Set `OPENAI_API_KEY` una sola vez (clave de hackathon).

### A. Entorno (5 min)
- `python -m venv venv && source venv/bin/activate`
- `pip install -r requirements.txt` (incluye `openai>=1.16`)
- `export OPENAI_API_KEY=...`

### B. Datos & modelos (10 min)
- Implementar `PersonaForge` (listo).  
- `StateBuilder`, `TemplateFactory`, `Score`, `InteractionLog` (listo).  
- `data_sim.generate_customers` → DataFrame/CSV (listo).

### C. Bandit (10 min)
- `ThompsonBandit` por contexto `(segmento, issue)` (ya implementado).  
- Reward clipping in `scoring.compute_reward`.  
- Opcional `EpsilonGreedy` con `epsilon=0.1`.

### D. Agents (15 min)
- `AgentsRunner` (Responses API).  
- `ResponderAgent`, `OutreachAgent` → `run_text`.  
- `Judge` → `run_json` + `Score` validation.  
- Safety guardrails (`SafetyChecker`).

### E. Streamlit (15 min)
- Sidebar: config + acciones.  
- Buttons → disparan loops (ya implementado en `streamlit_app.py`).  
- Tabs con gráficos (plotly) y tablas.  
- `st.session_state` para persistir dataset / logs / policy / iteration.

### F. Testing rápido (5 min)
- Ejecutar `streamlit run app/streamlit_app.py`.  
- Generar dataset, correr iteración, verificar logs y métricas.  
- Confirmar JSON Judge válido (ver consola si fallback).  
- Ajustar tokens/temperatura si hay cortes.

### G. Demo script
1. Mostrar un cliente enojado antes/después (mensaje + scores).  
2. Mostrar gráfico de evolución (reward/NPS).  
3. Destacar insight: “Para NVE mecánica → Escalar + Empático”.  
4. Mencionar roadmap y ROI.

---

## 9) “Modo BASIC” (si hay poco tiempo)
- Plantillas: sólo `empatiko` y `cupon`.  
- Segmentos: `VE` (vocal enojado) y `NVE` (no vocal enojado).  
- Iteraciones: 2 loops.  
- Sigue mostrando aprendizaje (bandit converge).

---

## 10) Apéndice — Repositorio esperado

```
app/
  streamlit_app.py
  data_sim.py
  templates.py
  scoring.py
  models.py
  factories/
    persona_forge.py
    state_builder.py
    template_factory.py
    responders.py
    judge.py
    policy_learner.py
    prioritizer.py
    metrics.py
    safety.py
    agents_runner.py
requirements.txt
README.md
AGENTS.md
Dockerfile
.dockerignore
scripts/run_experiment.py
context_engineering/
```

## 11) Infra rápida & ejecución en nube

- **Docker**: `docker build -t kavak-agents .`  
  Ejecutar experimento:  
  `docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY -v $(pwd)/profiles:/app/profiles kavak-agents python scripts/run_experiment.py profiles --output results/iter1.csv`
- **Resultados**: `scripts/run_experiment.py` genera DataFrame con `{client_id, NPS_final, LTV_final, ganancia_LTV, ...}` y summary; ideal para conectarlo a Streamlit/Gráficas.
- **Loop completo**: Planner → Conversación 1:1 → Judge → LTV Evaluator → Registro → Prompt Tuner (reutiliza summaries para evolucionar prompt).

Esto deja todo listo para montar una UI (Streamlit) en la VM sin tocar la lógica.

Listo para hackathon: “auto-aprende”, tokens ilimitados, valor claro en minutos. !*** End Patch
