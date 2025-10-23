"""
PersonaForge: Synthetic customer data generator.
Generates diverse customer profiles with realistic attributes.
"""
import random
import numpy as np
from typing import Literal
from app.models import Customer


ISSUE_MESSAGES = {
    "mecanica": [
        "Estoy muy molesto, llevo días sin respuesta sobre el ruido del motor.",
        "El auto tiene problemas con la transmisión desde que lo compré.",
        "Los frenos hacen un ruido extraño, necesito que lo revisen urgente.",
        "El motor se sobrecalienta constantemente, esto es inaceptable.",
    ],
    "finanzas": [
        "No entiendo los cargos adicionales en mi factura.",
        "Me prometieron una tasa de interés diferente, esto no es lo acordado.",
        "Quiero cancelar el financiamiento pero nadie me responde.",
        "Los pagos no están reflejados correctamente en mi cuenta.",
    ],
    "logistica": [
        "Llevo 2 semanas esperando la entrega del vehículo.",
        "Me cambiaron la fecha de entrega 3 veces sin aviso previo.",
        "No puedo contactar a nadie sobre el estatus de mi pedido.",
        "La documentación está incompleta y no me la han entregado.",
    ],
    "atencion": [
        "Nadie me devuelve las llamadas, el servicio es pésimo.",
        "He llamado 5 veces y cada persona me dice algo diferente.",
        "El vendedor desapareció después de la compra.",
        "Llevo días esperando una respuesta por WhatsApp.",
    ]
}

MINI_STORIES = {
    "mecanica": [
        "Cliente reportó ruido en motor tras compra hace {} semanas.",
        "Problemas de transmisión detectados {} días después de la entrega.",
        "Frenos con sonidos anormales, requiere inspección técnica urgente.",
        "Sobrecalentamiento del motor reportado en múltiples ocasiones.",
    ],
    "finanzas": [
        "Discrepancia en la tasa de interés ofrecida vs facturada.",
        "Cargos no explicados aparecen en el estado de cuenta del mes {}.",
        "Cliente solicita cancelación de crédito sin respuesta hace {} días.",
        "Pagos realizados no aparecen reflejados en el sistema.",
    ],
    "logistica": [
        "Retraso en entrega de {} semanas sin comunicación proactiva.",
        "Fechas de entrega reprogramadas {} veces en el último mes.",
        "Documentación del vehículo incompleta tras {} días de la compra.",
        "Cliente no puede rastrear el estatus de su pedido.",
    ],
    "atencion": [
        "Múltiples intentos de contacto sin respuesta hace {} días.",
        "Información contradictoria de diferentes agentes de servicio.",
        "Vendedor asignado no responde tras cierre de venta.",
        "Canal preferido (WhatsApp) con tiempos de respuesta >48h.",
    ]
}


class PersonaForge:
    """Generates synthetic customer data for simulation."""

    def __init__(self, seed: int = 42):
        """
        Initialize PersonaForge.

        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)

    def generate(
        self,
        n: int = 200,
        segments: list[Literal["VF", "VE", "NVF", "NVE"]] = None,
        p_angry: float = 0.5,
        p_vocal: float = 0.5
    ) -> list[Customer]:
        """
        Generate synthetic customer dataset.

        Args:
            n: Number of customers to generate
            segments: List of segments to sample from
            p_angry: Probability of having issues (angry customer)
            p_vocal: Probability of being vocal (proactive contact)

        Returns:
            List of Customer objects
        """
        if segments is None:
            segments = ["VF", "VE", "NVF", "NVE"]

        customers = []

        for i in range(n):
            segment = random.choice(segments)
            is_vocal = random.random() < p_vocal
            has_issues = random.random() < p_angry

            # Price distribution by segment
            if segment == "VF":
                price = np.random.uniform(300000, 600000)
            elif segment == "VE":
                price = np.random.uniform(150000, 350000)
            elif segment == "NVF":
                price = np.random.uniform(250000, 500000)
            else:  # NVE
                price = np.random.uniform(100000, 300000)

            # Recency (days since last purchase)
            last_purchase_days = int(np.random.exponential(30)) + 1

            # Past NPS (lower if has issues)
            if has_issues:
                past_NPS = random.randint(0, 6)
            else:
                past_NPS = random.randint(6, 10)

            # Churn risk (higher if has issues and low NPS)
            churn_risk_est = min(1.0, (10 - past_NPS) / 10 + (0.3 if has_issues else 0))
            churn_risk_est = round(churn_risk_est, 2)

            # Issue bucket and messages (if applicable)
            issue_bucket = None
            first_message = None
            mini_story = None

            if has_issues:
                issue_bucket = random.choice(["mecanica", "finanzas", "logistica", "atencion"])

                if is_vocal:
                    first_message = random.choice(ISSUE_MESSAGES[issue_bucket])

                # Generate mini story with time context
                story_template = random.choice(MINI_STORIES[issue_bucket])
                if "{}" in story_template:
                    time_value = random.randint(1, 4)
                    mini_story = story_template.format(time_value)
                else:
                    mini_story = story_template
            else:
                mini_story = f"Cliente satisfecho, compra reciente hace {last_purchase_days} días."

            channel_pref = random.choice(["whatsapp", "email", "phone", "sms"])

            customer = Customer(
                customer_id=f"C{str(i+1).zfill(4)}",
                segment=segment,
                is_vocal=is_vocal,
                last_purchase_days=last_purchase_days,
                price=round(price, 2),
                issues_flag=1 if has_issues else 0,
                past_NPS=past_NPS,
                first_message=first_message,
                channel_pref=channel_pref,
                churn_risk_est=churn_risk_est,
                issue_bucket=issue_bucket,
                mini_story=mini_story
            )

            customers.append(customer)

        return customers
