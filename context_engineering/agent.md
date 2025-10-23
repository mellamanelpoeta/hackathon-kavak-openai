# Context Engineering Loop — Proactive Outreach Optimizer

Sistema multi-agente (OpenAI Agents SDK) que:
1. Lee un folder con `N` perfiles en formato JSON y levanta un agente “cliente” por archivo.
2. Usa un agente “Planner” para diseñar el mensaje proactivo inicial (context engineering).
3. Ejecuta un agente “Orchestrator” que conversa, uno por uno, con todos los agentes cliente.
4. Califica cada conversación, calcula métricas (incl. LTV estimado por cohorte vocal/no vocal, satisfecho/no satisfecho).
5. Llama a un agente “Prompt Tuner” que ajusta el prompt-base del Planner según los aprendizajes de la iteración.
6. Repite el ciclo hasta converger, guardando evolución de prompts, métricas y selección de estrategias.

> Enfoque: **context engineering + online learning**. Todo debe correr in-memory (sin BBDD) y en < 30 min por iteración en demo.

---

## 1. Estructura de entrada (`profiles/*.json`)

Cada JSON describe un cliente simulado:
```json
{
  "customer_id": "C0001",
  "cohort": {"vocal": true, "satisfied": false},
  "persona": {
    "name": "Karen Ruiz",
    "age": 36,
    "location": "CDMX",
    "bio": "Compró SUV seminueva hace 2 meses. Enojada por retraso en papeles."
  },
  "purchase": {
    "vehicle": "SUV 2021",
    "price": 365000,
    "last_purchase_days": 65,
    "channel": "online"
  },
  "history": {
    "past_nps": 3,
    "tickets": [
      {"issue": "documentación", "sentiment": -0.6, "last_touch_days": 10}
    ],
    "messages": [
      {"role": "customer", "content": "Ya pasaron dos semanas y nadie me entrega los papeles."}
    ]
  },
  "risk_signals": {
    "churn_est": 0.45,
    "value_segment": "VE"
  }
}
```

---

## 2. Componentes (Agents)

| Módulo | Descripción | Entradas | Salidas |
| --- | --- | --- | --- |
| **ProfileLoader** | Lee folder, valida esquemas, crea `CustomerProfile` | rutas JSON | lista `CustomerProfile` |
| **CustomerAgentFactory** | Instancia agente OpenAI por perfil (role=“customer-sim”) con prompt personalizado | profile | `customer_agent_id` |
| **ContextPlanner (Planner Agent)** | Diseña mensaje proactivo base + estrategia (plantilla, tono, CTA) | cohort agregada + historial prompt | `Plan`: {prompt, estrategia, objetivos} |
| **ProactiveAgent (Orchestrator)** | Usa plan para interactuar con cada `customer_agent` (máx 3 turnos) | Plan + customer_agent | transcript + feedback |
| **JudgeAgent** | Evalúa conversación final (`NPS`, `Engagement`, `Churn`, `Sentiment`, `CTA clarity`) | transcript + profile | score JSON |
| **CohortScorer** | Calcula LTV esperado por cohorte, métricas agregadas, historial | scores + profiles | `CohortReport` |
| **PromptTuner (Meta Agent)** | Ajusta prompt del Planner usando reporte + históricos | Plan + CohortReport + prompt_hist | nuevo prompt, rationale |
| **RunLogger** | Persistencia ligera (memoria / CSV opcional) | todos los artefactos | timeline iteraciones |

Todos los agentes usan **OpenAI Responses API** con Models `gpt-4.1` (planner/meta) y `gpt-4.1-mini` (customer/judge/orchestrator).

---

## 3. Ciclo completo (por iteración)

1. **Load Profiles**  
   - `ProfileLoader.scan("profiles/")` → lista de perfiles validada.  
   - Mapear cohortes (Vocal+Satisfied, Vocal+Unsatisfied, etc.).

2. **Instantiate Customer Agents**  
   - `CustomerAgentFactory.create(profile)` → `agent_id` + `context` (primer mensaje si es vocal).  
   - Guardar mapping `profile_id → agent_id`.

3. **Plan Outreach**  
   - `ContextPlanner.run(cohort_stats, last_prompt)` produce:  
     ```json
     {
       "prompt_seed": "...",
       "tone": "empático-directo",
       "template": "escalar",
       "objectives": ["clarify SLA", "reduce churn"],
       "max_turns": 3
     }
     ```  
   - Registrar `prompt_seed` en histórico.

4. **Interact (ProactiveAgent)**  
   - Para cada `customer_agent`:  
     1. `proactive_agent.run(agent_id, prompt_seed, profile_context)`  
     2. Guarda transcript (`messages[]`, tokens, turnos).  
     3. Detener si `max_turns` o cliente responde negativamente continuo.  

5. **Evaluate & Score**  
   - `JudgeAgent` consume `profile + transcript` → JSON  
     ```json
     {
       "NPS_expected": 7.2,
       "EngagementProb": 0.63,
       "ChurnProb": 0.22,
       "LTV_delta": 0.08,
       "AspectSentiment": {"finanzas": -0.1, ...},
       "rationale": "Mensaje reconoce error, da SLA 48h."
     }
     ```  
   - `CohortScorer` agrega por cohorte:  
     - Promedio NPS esperado.  
     - Prob. churn, engagement.  
     - LTV esperado (`baseline_ltv + delta`).  
     - Plantilla/tono más efectivo.

6. **Tune Prompt (PromptTuner)**  
   - Input: `Plan`, `CohortReport`, `history`.  
   - Output: nuevo `prompt_seed` + reglas por cohorte.  
   - Guardar razonamiento (`what worked`, `cohorts needing change`, `experimentos propuestos`).

7. **Loop / Stopping**  
   - Mantener `iteration += 1`.  
   - Condiciones de parada:  
     - LTV delta < threshold (converged) ó  
     - Máx iteraciones (p.ej., 5) ó  
     - Tiempo agotado (hackathon).  
   - Exportar `RunLogger` → `runs/iteration_X.json`.

---

## 4. Contratos de datos

### 4.1 `CustomerProfile` (Pydantic)
```python
class CustomerProfile(BaseModel):
    customer_id: str
    cohort: Literal["vocal_satisfecho", "vocal_insatisfecho", "no_vocal_satisfecho", "no_vocal_insatisfecho"]
    persona: Persona
    purchase: PurchaseInfo
    history: History
    risk_signals: RiskSignals
```

### 4.2 Transcript
```json
{
  "customer_id": "C0001",
  "turns": [
    {"role": "agent", "content": "Hola Karen..."},
    {"role": "customer", "content": "Gracias, pero aún no tengo papeles."}
  ],
  "metadata": {"tokens": 118, "duration_sec": 4.2}
}
```

### 4.3 CohortReport
```json
{
  "iteration": 2,
  "cohorts": {
    "vocal_insatisfecho": {
      "n": 45,
      "avg_nps": 7.1,
      "avg_churn": 0.24,
      "avg_engagement": 0.66,
      "avg_ltv": 1.08,
      "best_strategy": "escalar + SLA 48h",
      "notes": "Responder rápido en el primer turno reduce churn"
    }
  },
  "global": {
    "avg_ltv": 1.04,
    "reward": 0.72,
    "tokens_used": 12045
  }
}
```

---

## 5. Métricas clave

- **Reward** (misma fórmula MVP): `0.6*NPS + 0.3*(Eng*10) - 0.3*(Churn*10)` normalizada.  
- **Δ LTV vs baseline**: `expected_ltv / baseline_ltv`.  
- **Mensaje ganador por cohorte**: top estrategia (plantilla+tono+CTA).  
- **Costo tokens** vs. LTV delta.  
- **Experimentos pendientes**: generados por PromptTuner.

---

## 6. Prompt Engineering Cycle

1. **Prompt seed inicial** (guardrails + objetivo general).  
2. **PromptTuner** agrega instrucciones condicionadas por cohorte:  
   - “Si cohort = vocal_insatisfecho: enfatiza SLA y disculpa breve”.  
   - “Si cohort = no_vocal_satisfecho: ofrecer beneficio sorpresa sólo si LTV < 1.05”.  
3. Añadir “experimentos” a probar en próxima iteración.  
4. Registrar cambios en `prompt_history/iter_X.json`.

---

## 7. Roadmap (sistema a escalar después del hackathon)

### Día 1–2 (hackathon)
- Implementar pipeline descrito (in-memory).  
- Mostrar 2 iteraciones en vivo.  
- Colectar sample output para slides (antes/después).

### Semana 1
- Persistencia en SQLite/Parquet para análisis posterior.  
- Evaluación offline con baseline random.  
- Introducir A/B (50% plantillas baseline).

### Semana 2–3
- Añadir features reales (CRM, tickets, pagos).  
- Contextual bandits/LinUCB con embeddings.  
- Métrica multi-objetivo (LTV, costo ticket, SLA).

### Semana 4+
- Integrar a workflow real (aprobación humana en loop).  
- Batch scoring + caching para reducir tokens.  
- Dashboard productivo (Metabase/Superset).

---

## 8. Checklist de desarrollo

1. **Setup**  
   - `pip install openai pydantic pandas streamlit`  
   - `export OPENAI_API_KEY=...`  
   - Crear carpetas: `profiles/`, `runs/`, `prompt_history/`.

2. **Profile loader**  
   - Validar schema (Pydantic).  
   - Log de perfiles cargados por cohorte.

3. **Customer Agents**  
   - Prompt base: describir persona, historial, tono esperado.  
   - Guardar `agent_id` y seeds.

4. **Planner**  
   - Context: KPIs target, restricciones (máx tokens, SLA requerido).  
   - Output JSON validado.

5. **Orchestrator**  
   - Pipeline: plan → message → interaction loop (max 3 turnos).  
   - Registro: transcripts, tokens, fallbacks si error.

6. **Judge & Cohort scoring**  
   - Judge con `response_format=json_object`.  
   - Cohort metrics + baseline comparativa.

7. **Prompt tuner**  
   - Input: plan + report + history.  
   - Output: nuevo prompt + `experiments`.

8. **Iteration runner**  
   - CLI `python run_iteration.py --profiles profiles --iter 1`.  
   - Guarda `runs/iter_1.json`, `prompt_history/iter_1.txt`.

9. **Streamlit (opcional)**  
   - Visualizar cohort metrics, transcripts, evolución prompt.  
   - Botón “Run iteration”.

10. **QA**  
    - Probar con 4 perfiles (1 por cohorte) antes de escalar.  
    - Validar JSONs Judge y Planner.  
    - Revisar escalabilidad con 50 perfiles.

---

## 9. Demo flow (5 min)

1. Mostrar perfiles (`profiles/`) y cohorte resumen.  
2. Correr iteración en vivo → ver transcripts destacados.  
3. Mostrar CohortReport (LTV ↑ / churn ↓).  
4. Enseñar prompt history (antes/después).  
5. Concluir con roadmap (contextual bandits, integración real).

---

Listo: carpeta `context_engineering/` contiene este `agent.md` para guiar implementación paso a paso. Utilizar como blueprint del sistema de context engineering auto-adaptativo.***
