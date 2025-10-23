# Kavak Demo - Bandits + LLM Evaluators

MVP demo de agentes conversacionales con aprendizaje online (multi-armed bandits) y evaluación automática mediante LLM para optimizar atención al cliente en Kavak.

## 🎯 Objetivo

Demostrar cómo los agentes LLM pueden:
1. **Personalizar** respuestas usando plantillas optimizadas por contexto
2. **Aprender online** qué plantillas funcionan mejor para cada segmento/issue
3. **Evaluar automáticamente** la calidad de las respuestas con métricas objetivas
4. **Priorizar** outreach proactivo a clientes de alto valor

## 🏗️ Arquitectura

```
PersonaForge → State Builder → Policy (Bandit) → Template Factory
                                    ↓
                            Responder/Outreach Agent
                                    ↓
                              Judge (Evaluator)
                                    ↓
                            Reward → Update Policy
```

### Componentes Principales

- **PersonaForge**: Genera dataset sintético de clientes con perfiles diversos
- **State Builder**: Empaqueta features del cliente en contexto estructurado
- **Template Factory**: 5 plantillas parametrizadas (empático, técnico, cupón, escalación, seguimiento)
- **Policy Learner**: Thompson Sampling o ε-greedy para selección de plantillas
- **Responder Agent**: Genera respuestas para clientes vocales (con mensaje inicial)
- **Outreach Agent**: Genera mensajes proactivos para clientes no-vocales
- **Judge**: Evalúa mensajes con LLM → {NPS, Engagement, Churn, AspectSentiment}
- **Prioritizer**: Rankea clientes para outreach basado en valor/riesgo
- **Safety**: Filtros de PII, términos prohibidos, tono inapropiado

## 🚀 Quick Start

### 1. Instalación

```bash
# Clonar repo
git clone <repo-url>
cd hackathon-kavak-openai

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configuración

```bash
# Copiar template de variables de entorno
cp .env.example .env

# Editar .env y agregar tu API key
OPENAI_API_KEY=sk-...
```

### 3. Ejecutar Demo

```bash
# Iniciar Streamlit
streamlit run app/streamlit_app.py
```

La app abrirá en `http://localhost:8501`

## 📊 Uso

### Workflow Básico

1. **Generar Dataset**: Click en "🎲 Generar Dataset" (sidebar)
   - Configura número de clientes (default: 100)
   - Genera perfiles sintéticos con segmentos, issues, NPS, etc.

2. **Ejecutar Iteración**: Click en "▶️ Ejecutar Iteración"
   - Procesa clientes vocales (respuestas)
   - Genera outreach proactivo (top-N no-vocales)
   - Evalúa cada mensaje con Judge
   - Actualiza política del bandit

3. **Revisar Resultados** en los tabs:
   - **Dashboard**: KPIs, gráficos de reward, distribución de plantillas
   - **Conversaciones**: Mensajes generados para vocales + scores
   - **Outreach**: Mensajes proactivos + scores
   - **Métricas**: Desglose por segmento/plantilla/issue
   - **Recomendaciones**: Insights automáticos y mejores combinaciones

### Configuración Avanzada

**Sidebar options:**
- Número de clientes: 10-500
- Algoritmo: Thompson Sampling o ε-greedy
- Top N outreach: 5-100
- Modelos: gpt-4o, gpt-4o-mini, gpt-3.5-turbo

**Iteraciones múltiples:**
- Ejecuta 2-3 iteraciones para ver aprendizaje online
- El bandit converge hacia mejores plantillas por contexto

## 📁 Estructura del Proyecto

```
hackathon-kavak-openai/
├── app/
│   ├── streamlit_app.py          # UI principal
│   ├── models.py                 # Pydantic models
│   ├── templates.py              # Definiciones de plantillas
│   ├── scoring.py                # Cálculo de reward
│   ├── factories/
│   │   ├── persona_forge.py      # Generador de clientes
│   │   ├── state_builder.py      # Constructor de contexto
│   │   ├── template_factory.py   # Relleno de plantillas
│   │   ├── responders.py         # Agents (Responder/Outreach)
│   │   ├── judge.py              # Evaluador LLM
│   │   ├── policy_learner.py     # Bandits (Thompson/ε-greedy)
│   │   ├── prioritizer.py        # Ranker de outreach
│   │   ├── metrics.py            # Agregación de KPIs
│   │   └── safety.py             # Filtros de contenido
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── AGENTS.md                     # Especificación técnica completa
```

## 🎯 Métricas y KPIs

### Reward Function

```python
R = 0.6 * NPS_expected + 0.3 * (Engagement * 10) - 0.3 * (Churn * 10)
Normalizado a [0, 1]
```

### Judge Scores

- **NPS_expected** (0-10): NPS anticipado del cliente tras recibir el mensaje
- **EngagementProb** (0-1): Probabilidad de respuesta/acción del cliente
- **ChurnProb** (0-1): Probabilidad de abandono/churn
- **AspectSentiment** (-1 a 1): Sentimiento por aspecto (finanzas, mecánica, logística, atención)
- **Rationale** (≤280 chars): Explicación de la evaluación

## 🧪 Testing Rápido

Checklist de 15 minutos:

- [ ] Judge devuelve JSON válido con 5 campos
- [ ] Reward en rango [0, 1]
- [ ] Bandit actualiza distribución tras rewards
- [ ] Baseline vs policy: diferencia visible en 2ª-3ª iteración
- [ ] Safety bloquea términos prohibidos
- [ ] Streamlit refresca sin reiniciar sesión

## 🔒 Safety & Compliance

**Guardrails automáticos:**
- Detección de PII (RFC, tarjetas, SSN)
- Términos prohibidos (garantías absolutas, claims legales)
- Límites de compensación (máx 20%)
- Validación de tono (no culpar cliente, incluir next step)

**Fallbacks:**
- Mensajes seguros si Judge falla
- Mensaje genérico si Safety bloquea contenido

## 🚦 Roadmap

### Fase 1 (Actual - MVP)
- [x] Dataset sintético con PersonaForge
- [x] 5 plantillas base
- [x] Thompson Sampling + ε-greedy
- [x] Judge con OpenAI
- [x] UI Streamlit con 5 tabs
- [x] Safety básico

### Fase 2 (1-2 semanas)
- [ ] Conectar CRM real (históricos, tickets)
- [ ] Micro-acciones (cupón %, SLA específico)
- [ ] Contextual bandits (LinUCB con features)
- [ ] A/B testing framework
- [ ] Aspect-based sentiment (ABS) granular

### Fase 3 (1 mes)
- [ ] Deployment productivo
- [ ] Monitoreo real-time
- [ ] Human-in-the-loop para casos edge
- [ ] Expansion de plantillas automática

## 🛠️ Troubleshooting

**Error: "OpenAI API key required"**
- Verifica que `.env` tenga `OPENAI_API_KEY=sk-...`
- O ingresa la key en el sidebar de la UI

**Error: "Judge evaluation failed"**
- Revisa quota/límites de la API
- El sistema usa fallback score por defecto

**Mensajes bloqueados por Safety**
- Revisa logs en terminal
- Ajusta `PROHIBITED_TERMS` en `app/factories/safety.py`

**Bandit no aprende (reward flat)**
- Ejecuta más iteraciones (mínimo 2-3)
- Aumenta número de clientes
- Verifica que Judge esté devolviendo scores diversos

## 📚 Referencias

- [AGENTS.md](./AGENTS.md) - Especificación técnica completa
- [OpenAI Agents API](https://platform.openai.com/docs/guides/agents)
- [Thompson Sampling](https://en.wikipedia.org/wiki/Thompson_sampling)
- [Multi-Armed Bandits](https://en.wikipedia.org/wiki/Multi-armed_bandit)

## 👥 Equipo

Kavak Hackathon - OpenAI Agents Track

## 📄 Licencia

MIT
