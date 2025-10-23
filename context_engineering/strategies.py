"""
Strategy catalog for proactive outreach conversations.

Each strategy entry provides:
  - nombre_estrategia (identifier)
  - razonamiento_estrategia
  - accion_puntual
  - costo (MXN)
  - LTV base estimado (para referencia)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class StrategyDefinition:
    nombre_estrategia: str
    razonamiento_estrategia: str
    accion_puntual: str
    costo: float
    ltv_base: float


STRATEGY_DEFINITIONS: Dict[str, StrategyDefinition] = {
    "Sin_Accion": StrategyDefinition(
        nombre_estrategia="Sin_Accion",
        razonamiento_estrategia="Mantener comunicación amable sin incentivos; refuerza confianza sin costos adicionales.",
        accion_puntual="Enviar mensaje de seguimiento o actualización general sin ofrecer beneficios.",
        costo=0.0,
        ltv_base=15000.0,
    ),
    "Cupon_Descuento": StrategyDefinition(
        nombre_estrategia="Cupon_Descuento",
        razonamiento_estrategia="Incentivar recompra o mantenimiento con un cupón de $500–$1,000 MXN (1–2% del ticket promedio).",
        accion_puntual="Enviar cupón de descuento o crédito aplicable a servicios o próxima compra.",
        costo=800.0,
        ltv_base=18500.0,
    ),
    "Compensacion_Devolucion": StrategyDefinition(
        nombre_estrategia="Compensacion_Devolucion",
        razonamiento_estrategia="Resolver inconformidades serias mediante devolución parcial o bonificación; alto costo pero fuerte efecto en lealtad.",
        accion_puntual="Ofrecer reembolso parcial o bonificación directa en cuenta Kavak.",
        costo=2500.0,
        ltv_base=21000.0,
    ),
    "Canalizar_Area": StrategyDefinition(
        nombre_estrategia="Canalizar_Area",
        razonamiento_estrategia="Derivar al área correcta (Finanzas, Legal o Atención avanzada) para mejorar resolución y tiempos de respuesta.",
        accion_puntual="Registrar ticket interno y canalizar el caso al área adecuada con seguimiento confirmado.",
        costo=500.0,
        ltv_base=17000.0,
    ),
    "Escalar_Mecanico": StrategyDefinition(
        nombre_estrategia="Escalar_Mecanico",
        razonamiento_estrategia="Brindar atención técnica o revisión en taller; útil para quejas de desempeño o garantía.",
        accion_puntual="Agendar cita con taller certificado Kavak y dar seguimiento post-servicio.",
        costo=1200.0,
        ltv_base=20000.0,
    ),
    "Upsell_Personalizado": StrategyDefinition(
        nombre_estrategia="Upsell_Personalizado",
        razonamiento_estrategia="Incrementar valor de clientes satisfechos ofreciendo upgrades, accesorios o recompra.",
        accion_puntual="Proponer plan de recompra o mejora de vehículo con condiciones de crédito personalizadas.",
        costo=-1000.0,
        ltv_base=23000.0,
    ),
    "Seguimiento_Concreto": StrategyDefinition(
        nombre_estrategia="Seguimiento_Concreto",
        razonamiento_estrategia="Aumentar confianza cumpliendo promesas claras (SLA de contacto o cita) sin costo alto.",
        accion_puntual="Comprometer seguimiento específico (llamada o correo en 48 h) y ejecutarlo.",
        costo=300.0,
        ltv_base=17500.0,
    ),
}


def get_strategy(strategy_id: str) -> StrategyDefinition:
    if strategy_id not in STRATEGY_DEFINITIONS:
        raise KeyError(f"Estrategia '{strategy_id}' no encontrada")
    return STRATEGY_DEFINITIONS[strategy_id]


__all__ = ["StrategyDefinition", "STRATEGY_DEFINITIONS", "get_strategy"]
