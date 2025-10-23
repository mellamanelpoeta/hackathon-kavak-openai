"""Pydantic schemas for HumanSimulacra personas."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, model_validator


class RegistroVocalidad(BaseModel):
    """Interaction record visible to the CX team."""

    canal: Literal["app", "email", "telefono", "whatsapp", "taller", "otro"]
    resumen: str = Field(
        min_length=10,
        description="Resumen conciso del evento que CX conoce.",
    )
    nps: int | None = Field(
        default=None,
        ge=0,
        le=10,
        description="Puntaje NPS si fue capturado durante la interaccion.",
    )


class Persona(BaseModel):
    """
    Persona utilizada para simulaciones de conversaciones con agentes de CX.

    - `historia_oculta`: Detalles personales desconocidos por CX que explican motivaciones.
    - `historia_revelada`: Informacion disponible en los sistemas de Kavak.
    - `es_vocal`: Indica si CX tiene senales recientes del cliente.
    - `satisfaccion`: Solo se conoce cuando la persona es vocal; en caso contrario debe ser `None`.
    - `historial_vocalidad`: Interacciones visibles para CX (vacio si no es vocal).
    - `problema`: Presente cuando la persona esta insatisfecha (puede ser `None` si no hay problema).
    - `expectativa_solucion`: Deseo intimo de la persona para resolver la relacion con Kavak.
    - `prompt_conversacional`: Instruccion final para role-play con un agente proactivo.
    """

    nombre: str = Field(description="Nombre de pila que usara la persona durante la simulacion.")
    edad: PositiveInt = Field(description="Edad aproximada de la persona.")
    ciudad: str = Field(description="Ciudad principal asociada al journey con Kavak.")
    ocupacion: str = Field(description="Ocupacion o rol profesional relevante.")
    relacion_kavak: Literal["compra", "venta", "compra_venta"] = Field(
        description="Tipo de relacion principal con Kavak."
    )
    ltv: PositiveFloat = Field(description="Valor presente de la relacion estimada por Kavak.")
    es_vocal: bool = Field(description="Indica si existen senales directas recientes para CX.")
    satisfaccion: Literal["Satisfecho", "Insatisfecho"] | None = Field(
        default=None,
        description="Estado de satisfaccion conocido por CX; debe ser None cuando no es vocal.",
    )
    historia_oculta: str = Field(
        description="Narrativa completa que explica contexto socioeconomico y emocional.",
    )
    historia_revelada: str = Field(
        description="Informacion visible para CX sobre la operacion con Kavak.",
    )
    historial_vocalidad: list[RegistroVocalidad] = Field(
        description="Eventos reportados que sustentan la vocalidad de la persona.",
    )
    problema: str | None = Field(
        default=None,
        description="Dolor principal actual respecto a Kavak, presente si esta insatisfecho.",
    )
    expectativa_solucion: str = Field(
        min_length=20,
        description="Resultado deseado por la persona para mejorar su relacion con Kavak.",
    )
    prompt_conversacional: str = Field(
        min_length=50,
        description="Instruccion para que el modelo interprete a la persona frente al agente de CX.",
    )

    @model_validator(mode="after")
    def _validar_coherencia_vocalidad(self) -> "Persona":
        if not self.es_vocal:
            if self.satisfaccion is not None:
                raise ValueError("La satisfaccion debe ser None cuando la persona no es vocal.")
            if self.historial_vocalidad:
                raise ValueError(
                    "Una persona no vocal no debe tener historial de interacciones visibles."
                )
        else:
            if self.satisfaccion is None:
                raise ValueError(
                    "Las personas vocales deben exponer explicitamente su estado de satisfaccion."
                )
            if not self.historial_vocalidad:
                raise ValueError(
                    "Las personas vocales requieren al menos un registro en el historial de vocalidad."
                )
            if self.satisfaccion == "Insatisfecho" and not self.problema:
                raise ValueError(
                    "Una persona vocal e insatisfecha debe describir su problema actual."
                )
        return self
