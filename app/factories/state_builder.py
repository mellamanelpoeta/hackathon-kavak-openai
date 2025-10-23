"""
State Builder: Converts Customer objects to Context for agents.
Packages features and customer story for agent consumption.
"""
from app.models import Customer, Context


class StateBuilder:
    """Builds context objects from customer data."""

    @staticmethod
    def build(customer: Customer) -> Context:
        """
        Build context from customer data.

        Args:
            customer: Customer object with all attributes

        Returns:
            Context object ready for agent consumption
        """
        return Context(
            customer_id=customer.customer_id,
            segment=customer.segment,
            is_vocal=customer.is_vocal,
            last_purchase_days=customer.last_purchase_days,
            price=customer.price,
            issues_flag=customer.issues_flag,
            past_NPS=customer.past_NPS,
            channel_pref=customer.channel_pref,
            churn_risk_est=customer.churn_risk_est,
            mini_story=customer.mini_story or "",
            first_message=customer.first_message,
            issue_bucket=customer.issue_bucket or "atencion"
        )
