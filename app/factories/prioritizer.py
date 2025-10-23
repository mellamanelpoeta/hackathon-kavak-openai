"""
Prioritizer: Ranks non-vocal customers for proactive outreach.
Uses a scoring function based on issues, price, and recency.
"""
import numpy as np
from typing import List
from app.models import Customer


class Prioritizer:
    """Ranks customers for outreach based on engagement potential."""

    def __init__(
        self,
        w_issues: float = 0.5,
        w_price: float = 0.3,
        w_recency: float = 0.2
    ):
        """
        Initialize Prioritizer.

        Args:
            w_issues: Weight for issues flag
            w_price: Weight for price/value
            w_recency: Weight for recency (recent = higher priority)
        """
        self.w_issues = w_issues
        self.w_price = w_price
        self.w_recency = w_recency

    def rank(self, customers: List[Customer], top_n: int = None) -> List[Customer]:
        """
        Rank customers by outreach priority.

        Args:
            customers: List of customers to rank
            top_n: Return only top N (None for all)

        Returns:
            Sorted list of customers (highest priority first)
        """
        if not customers:
            return []

        # Compute scores
        scored = []
        prices = [c.price for c in customers]
        max_price = max(prices) if prices else 1.0
        recencies = [c.last_purchase_days for c in customers]
        max_recency = max(recencies) if recencies else 1.0

        for customer in customers:
            score = self._compute_score(customer, max_price, max_recency)
            scored.append((customer, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top N
        if top_n:
            scored = scored[:top_n]

        return [c for c, s in scored]

    def _compute_score(
        self,
        customer: Customer,
        max_price: float,
        max_recency: float
    ) -> float:
        """
        Compute priority score for a customer.

        Formula: S = w_issues*issues + w_price*norm(price) + w_recency*norm(1/recency)

        Args:
            customer: Customer to score
            max_price: Max price for normalization
            max_recency: Max recency for normalization

        Returns:
            Priority score
        """
        # Issues component
        issues_score = float(customer.issues_flag)

        # Price component (normalized)
        price_score = customer.price / max_price if max_price > 0 else 0.0

        # Recency component (inverse: more recent = higher score)
        # Add 1 to avoid division by zero
        recency_score = 1.0 / (customer.last_purchase_days + 1)
        recency_score = recency_score / (1.0 / (max_recency + 1)) if max_recency > 0 else 0.0

        # Weighted sum
        score = (
            self.w_issues * issues_score +
            self.w_price * price_score +
            self.w_recency * recency_score
        )

        return score
