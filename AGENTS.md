# agents.md — MVP bandits + LLM evaluators (Kavak demo)

> **Objetivo**: Documento operativo para implementar, en < 1 hora, un demo funcional en Streamlit con agentes (OpenAI Agents API o equivalente), bandits online y evaluador LLM. Incluye prompts, contratos I/O, flujos, KPIs y stubs de código.

---

## 0) Arquitectura lógica

**Factories/Agentes**

* **PersonaForge (Simulador)**: genera dataset sintético y primer mensaje si el cliente es vocal.
* **State Builder**: empaqueta features + mini-historia cliente.
* **Template Factory**: plantillas parametrizadas + guardrails de estilo.
* **Responder Agent**: responde a *vocales* usando plantilla elegida por la política (bandit).
* **Outreach Agent**: redacta *one-shot* a *no‑vocales* (usando plantilla elegida por policy).
* **Judge (Evaluator)**: puntúa `{NPS_expected, EngagementProb, ChurnProb, AspectSentiment}` + rationale.
* **Policy Learner (Bandits)**: Thompson/ε-greedy por (segmento, issue_bucket) → elige plantilla y actualiza con reward.
* **Prioritizer**: rankea no‑vocales para outreach (top‑N).
* **Metrics**: cálculos agregados para dashboard.
* **Safety**: PII/tox/toño/claims guardrails.

**Tabs Streamlit**: Simulación | Conversaciones | Outreach | Métricas | Recomendaciones

---

## 1) Esquemas y contratos (I/O)

### 1.1 CSV mínimo de clientes

```
customer_id,segment{VF,VE,NVF,NVE},is_vocal,last_purchase_days,price,issues_flag{0/1},past_NPS,first_message,channel_pref,churn_risk_est
```

### 1.2 Contexto (State Builder → Agents)

```json
{
  "customer_id": "C001",
  "segment": "VE",
  "is_vocal": true,
  "last_purchase_days": 17,
  "price": 215000,
  "issues_flag": 1,
  "past_NPS": 4,
  "channel_pref": "whatsapp",
  "churn_risk_est": 0.42,
  "mini_story": "Cliente reportó ruido en motor tras compra hace 3 semanas.",
  "first_message": "Estoy muy molesto, llevo días sin respuesta sobre el ruido del motor.",
  "issue_bucket": "mecanica"
}
```

### 1.3 Score del **Judge** (JSON estricto)

```json
{
  "NPS_expected": 0.0,
  "EngagementProb": 0.0,
  "ChurnProb": 0.0,
  "AspectSentiment": {"finanzas":0.0, "mecanica":0.0, "logistica":0.0, "atencion":0.0},
  "rationale": "<=280 chars"
}
```

### 1.4 Recompensa (reward)

```
R_raw = 0.6*NPS_expected + 0.3*(EngagementProb*10) - 0.3*(ChurnProb*10)
R = clip(rescale_0_1(R_raw), 0, 1)
```

### 1.5 Plantillas (máx 5)

```yaml
- id: empatiko
  name: Empático breve
  slots: [saludo, reconocimiento_issue, accion, next_step, firma]
  guardrails:
    - no culpar al cliente
    - ofrece disculpa breve
    - claridad en próxima acción
- id: tecnico
  name: Técnico solución
  slots: [saludo, diagnostico, accion_tecnica, next_step, firma]
- id: cupon
  name: Compensación/cupón
  slots: [saludo, reconocimiento_issue, oferta_compensacion, next_step, firma]
- id: escalar
  name: Escalación interna
  slots: [saludo, reconocimiento_issue, derivacion_area, sla, firma]
- id: seguimiento
  name: Seguimiento proactivo
  slots: [saludo, resumen, next_touch, canal, firma]
```

---

## 2) Prompts (sistémicos) resumidos

### 2.1 **Responder Agent** (vocales)

```
Eres un agente de atención de Kavak. Objetivo: maximizar NPS esperado, reducir churn y aumentar engagement.
Usa UNA de las plantillas disponibles y rellena los slots con el contexto del cliente.
Reglas: tono respetuoso, no culpar, no promesas legales, no instrucciones técnicas imposibles.
Si corresponde, sugiere escalar a mecánica/finanzas/logística con un SLA concreto.
Devuelve SOLO el texto final para el cliente (sin JSON, sin metadatos).
```

**Entrada**: `{context, template_id}`

**Salida**: `FinalMessage (string)`

---

### 2.2 **Outreach Agent** (no‑vocales)

```
Eres un agente de outreach de Kavak con una oportunidad única de contacto (one‑shot).
Elige UNA plantilla para maximizar apertura y respuesta. Personaliza con recencia, monto y issue.
Tono profesional, claro, y orientado a siguiente paso.
Devuelve SOLO el texto a enviar (sin JSON).
```

**Entrada**: `{context, template_id}`

**Salida**: `FinalMessage (string)`

---

### 2.3 **Judge (Evaluator)**

```
Evalúas un mensaje dado el perfil y el historial del cliente.
Devuelve JSON ESTRICTO con:
{ "NPS_expected": 0-10, "EngagementProb": 0-1, "ChurnProb": 0-1,
  "AspectSentiment": {"finanzas":-1..1, "mecanica":-1..1, "logistica":-1..1, "atencion":-1..1},
  "rationale": "<=280 chars" }
Criterios: penaliza frialdad con enojados; valora claridad, acción concreta y SLA cuando aplique.
```

**Entrada**: `{context, message}`

**Salida**: `ScoreJSON`

---

## 3) Flujo E2E (pseudocódigo)

```python
# 1) Dataset sintético
customers = persona_forge.generate(n=200, segments=["VF","VE","NVF","NVE"], p_angry=0.5, p_vocal=0.5)

# 2) Respuesta a vocales
for c in customers if c.is_vocal:
    ctx = state_builder.build(c)
    arm = policy.select(context=(c.segment, c.issue_bucket))  # plantilla
    draft = template_factory.fill(arm, ctx)
    msg = responder.run(context=ctx, template_text=draft)
    score = judge.run(context=ctx, message=msg)  # JSON estricto
    reward = compute_reward(score)
    policy.update(arm, reward, context=(c.segment, c.issue_bucket))
    logs.append({"customer_id": c.id, "arm": arm, "score": score, "reward": reward, "message": msg})

# 3) Outreach a no‑vocales (top‑N)
nv = [c for c in customers if not c.is_vocal]
ranked = prioritizer.rank(nv)[:30]
for c in ranked:
    ctx = state_builder.build(c)
    arm = policy.select(context=(c.segment, c.issue_bucket))
    draft = template_factory.fill(arm, ctx)
    msg = outreach.run(context=ctx, template_text=draft)
    score = judge.run(context=ctx, message=msg)
    reward = compute_reward(score)
    policy.update(arm, reward, context=(c.segment, c.issue_bucket))
    logs.append({"customer_id": c.id, "arm": arm, "score": score, "reward": reward, "message": msg})

# 4) Métricas
dashboard = metrics.aggregate(logs)
```

---

## 4) Política (Bandits) — API mínima

```python
class ThompsonBandit:
    def __init__(self, arms: list[str]):
        self.state = {arm: {"alpha": 1.0, "beta": 1.0} for arm in arms}

    def select(self, context) -> str:
        # Context can be used to index a per-context dict in v2
        samples = {arm: np.random.beta(s["alpha"], s["beta"]) for arm, s in self.state.items()}
        return max(samples, key=samples.get)

    def update(self, arm: str, reward: float, context=None):
        r = np.clip(reward, 0, 1)
        self.state[arm]["alpha"] += r
        self.state[arm]["beta"] += 1 - r
```

> **Nota**: Para **ε-greedy**, mantener `counts` y `means`; explorar con probabilidad ε.

---

## 5) Integración con Agents API (patrón agnóstico)

> Puedes usar OpenAI Agents API u orquestador equivalente. Mantén este contrato:

```python
# Responder / Outreach
final_text = agents.run(
  name="Responder",               # o "Outreach"
  system_prompt=RESPONDER_SYSTEM,  # prompt de la sección 2
  input={"context": ctx, "template_text": draft}
)

# Judge (debe devolver JSON estricto)
score_json = agents.run(
  name="Judge",
  system_prompt=JUDGE_SYSTEM,
  input={"context": ctx, "message": final_text},
  output_format="json"  # forzar JSON válido
)
```

**Requisitos mínimos**

* Control de *temperature* bajo (p.ej., 0.2–0.5) para Responder/Outreach.
* `max_tokens` suficientes (<= 300 para mensajes; <= 200 para Judge).
* Retries con backoff y validación de JSON (Judge).

---

## 6) Prioritizer (no‑vocales)

```python
S = 0.5*issues_flag + 0.3*normalize(price) + 0.2*normalize(1/last_purchase_days)
ranked = sorted(no_vocales, key=lambda c: S(c), reverse=True)
```

---

## 7) Métricas de demo (dashboard)

* **Lift NPS esperado** vs baseline (plantilla uniforme).
* **Recompensa media** por plantilla y por segmento.
* **% selección por arm** (exploración/explotación).
* **Top insights** (texto): “Para NVE mecánico, Empático+Escalación domina”.

---

## 8) Safety & Compliance (mínimo viable)

* Bloqueo de PII/claims, lista de términos prohibidos (jurídicos/garantías imposibles).
* Tono: evitar culpas; siempre ofrecer siguiente paso claro.
* Límite de compensación (p.ej., cupón <= X%).

```python
def safety_check(text: str) -> tuple[bool, str]:
    # return (is_safe, reason)
    if contains_pii(text) or violates_policy(text):
        return False, "blocked: policy"
    return True, "ok"
```

---

## 9) Estructura de repo

```
/app
  streamlit_app.py
  factories/
    persona_forge.py
    state_builder.py
    template_factory.py
    responders.py      # Responder / Outreach wrappers
    judge.py
    policy_learner.py
    prioritizer.py
    metrics.py
    safety.py
  scoring.py           # compute_reward
  data_sim.py
  templates.py
requirements.txt
```

---

## 10) `requirements.txt` sugerido

```
streamlit
pandas
numpy
pydantic
# openai (o sdk equivalente del proveedor de Agents)
```

---

## 11) Hooks de Streamlit (UI → Factories)

* **Botones**: `Generar dataset`, `Correr respuestas`, `Correr outreach`, `Iterar aprendizaje`.
* **Session state**: `customers`, `logs`, `policy_state`, `iteration`.
* **Charts**: línea (NPS esperado por iteración), barras (reward medio por plantilla/segmento), tabla (mensajes & rationale truncado).

---

## 12) KPIs por factory (resumen)

| Factory            | KPI principal                      |
| ------------------ | ---------------------------------- |
| PersonaForge       | Cobertura segmentos, diversidad    |
| Responder/Outreach | Score Judge ↑, latencia p95        |
| Judge              | % JSON válido, estabilidad         |
| Policy             | Regret ↓, reward ↑                 |
| Prioritizer        | Lift engagement esperado vs random |
| Safety             | Violaciones = 0                    |
| Metrics            | Freshness, cobertura               |

---

## 13) Pruebas rápidas (checklist 15’)

* [ ] Judge devuelve JSON válido con 5 campos.
* [ ] `compute_reward()` clipping [0,1].
* [ ] Bandit actualiza alpha/beta y cambia la distribución de selección.
* [ ] Baseline vs policy: diferencia visible en la 2ª–3ª iteración.
* [ ] Safety bloquea términos prohibidos.
* [ ] Streamlit refresca métricas sin reiniciar sesión.

---

## 14) “Modo BASIC” (ultra‑recortado)

* 2 plantillas (Empático vs Cupón)
* 2 segmentos (VE vs NVE)
* 2 iteraciones
* Aún se observa aprendizaje online.

---

## 15) Anexos — Stubs

### 15.1 `compute_reward`

```python
def compute_reward(score: dict) -> float:
    nps = float(score.get("NPS_expected", 0.0))
    eng = float(score.get("EngagementProb", 0.0))
    chrn = float(score.get("ChurnProb", 0.0))
    r_raw = 0.6*nps + 0.3*(eng*10) - 0.3*(chrn*10)
    # reescalar 0-10 → 0-1 con min-max simple (asumiendo rango [0,10])
    r = max(0.0, min(1.0, r_raw/10.0))
    return r
```

### 15.2 Validación estricta del JSON del Judge

```python
from pydantic import BaseModel, Field
from typing import Dict

class Score(BaseModel):
    NPS_expected: float = Field(ge=0, le=10)
    EngagementProb: float = Field(ge=0, le=1)
    ChurnProb: float = Field(ge=0, le=1)
    AspectSentiment: Dict[str, float]
    rationale: str
```

---

## 16) Variables de entorno

```
AGENTS_PROVIDER=openai
OPENAI_API_KEY=...
MODEL_RESPONDER=...   # p.ej., gpt-4.1-mini / equivalente
MODEL_OUTREACH=...
MODEL_JUDGE=...
```

> Cambia nombres según proveedor/modelos disponibles.

---

## 17) Riesgos & mitigación (demo)

* **Sesgo de auto-selección**: estratificar por segmento & mantener exploración (>0) siempre.
* **Sobre‑ajuste a plantillas**: rotación/expansión gradual y evaluación off‑policy en fase 2.
* **Tono inapropiado**: Safety + palabras prohibidas + SLA explícito.

---

## 18) Roadmap inmediato (1–2 semanas)

* Conectar CRM para históricos reales (tickets, NPS).
* Extender plantillas a **micro‑acciones** (cupón%, escalar a mecánico/finanzas, promesa SLA).
* Contextual bandits (LinUCB) → usar features.
* ABS por aspectos para insights operativos.

---

