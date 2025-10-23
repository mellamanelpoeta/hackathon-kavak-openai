"""
Policy Learner: Multi-armed bandit with Thompson Sampling.
Maintains per-context arm selection and updates based on rewards.
"""
import numpy as np
from typing import Optional, Tuple


class ThompsonBandit:
    """
    Thompson Sampling bandit for template selection.
    Maintains Beta distributions for each arm (template).
    Supports contextual bandits via context-specific states.
    """

    def __init__(self, arms: list[str], alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """
        Initialize Thompson Sampling bandit.

        Args:
            arms: List of arm IDs (template IDs)
            alpha_prior: Prior alpha parameter for Beta distribution
            beta_prior: Prior beta parameter for Beta distribution
        """
        self.arms = arms
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

        # Global state (used when no context provided)
        self.global_state = {
            arm: {"alpha": alpha_prior, "beta": beta_prior}
            for arm in arms
        }

        # Contextual state: dict[(segment, issue_bucket)] -> arm_state
        self.contextual_state = {}

    def _get_state(self, context: Optional[Tuple[str, str]] = None) -> dict:
        """
        Get state dict for given context.

        Args:
            context: (segment, issue_bucket) tuple or None for global

        Returns:
            State dictionary for the context
        """
        if context is None:
            return self.global_state

        if context not in self.contextual_state:
            # Initialize new context with priors
            self.contextual_state[context] = {
                arm: {"alpha": self.alpha_prior, "beta": self.beta_prior}
                for arm in self.arms
            }

        return self.contextual_state[context]

    def select(self, context: Optional[Tuple[str, str]] = None) -> str:
        """
        Select arm using Thompson Sampling.

        Args:
            context: Optional (segment, issue_bucket) tuple for contextual selection

        Returns:
            Selected arm ID (template ID)
        """
        state = self._get_state(context)

        # Sample from Beta distribution for each arm
        samples = {
            arm: np.random.beta(s["alpha"], s["beta"])
            for arm, s in state.items()
        }

        # Return arm with highest sample
        return max(samples, key=samples.get)

    def update(self, arm: str, reward: float, context: Optional[Tuple[str, str]] = None):
        """
        Update arm statistics with observed reward.

        Args:
            arm: Arm ID that was selected
            reward: Observed reward in [0, 1]
            context: Optional context tuple
        """
        # Clip reward to [0, 1]
        r = np.clip(reward, 0, 1)

        state = self._get_state(context)

        # Update Beta distribution parameters
        state[arm]["alpha"] += r
        state[arm]["beta"] += (1 - r)

    def get_statistics(self, context: Optional[Tuple[str, str]] = None) -> dict:
        """
        Get current statistics for all arms.

        Args:
            context: Optional context tuple

        Returns:
            Dictionary with mean and std for each arm
        """
        state = self._get_state(context)

        stats = {}
        for arm, s in state.items():
            alpha = s["alpha"]
            beta = s["beta"]
            mean = alpha / (alpha + beta)
            variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
            std = np.sqrt(variance)

            stats[arm] = {
                "mean": mean,
                "std": std,
                "alpha": alpha,
                "beta": beta,
                "n_pulls": alpha + beta - 2  # subtract priors
            }

        return stats

    def get_all_contexts(self) -> list[Tuple[str, str]]:
        """Get list of all contexts seen so far."""
        return list(self.contextual_state.keys())


class EpsilonGreedyBandit:
    """
    Epsilon-greedy bandit for template selection.
    Simpler alternative to Thompson Sampling.
    """

    def __init__(self, arms: list[str], epsilon: float = 0.1):
        """
        Initialize epsilon-greedy bandit.

        Args:
            arms: List of arm IDs
            epsilon: Exploration probability
        """
        self.arms = arms
        self.epsilon = epsilon

        # Global state
        self.global_state = {
            arm: {"total_reward": 0.0, "count": 0}
            for arm in arms
        }

        # Contextual state
        self.contextual_state = {}

    def _get_state(self, context: Optional[Tuple[str, str]] = None) -> dict:
        """Get state dict for given context."""
        if context is None:
            return self.global_state

        if context not in self.contextual_state:
            self.contextual_state[context] = {
                arm: {"total_reward": 0.0, "count": 0}
                for arm in self.arms
            }

        return self.contextual_state[context]

    def select(self, context: Optional[Tuple[str, str]] = None) -> str:
        """Select arm using epsilon-greedy strategy."""
        state = self._get_state(context)

        # Explore with probability epsilon
        if np.random.random() < self.epsilon:
            return np.random.choice(self.arms)

        # Exploit: choose arm with highest mean reward
        means = {
            arm: (s["total_reward"] / s["count"] if s["count"] > 0 else 0.0)
            for arm, s in state.items()
        }

        return max(means, key=means.get)

    def update(self, arm: str, reward: float, context: Optional[Tuple[str, str]] = None):
        """Update arm statistics."""
        r = np.clip(reward, 0, 1)
        state = self._get_state(context)

        state[arm]["total_reward"] += r
        state[arm]["count"] += 1

    def get_statistics(self, context: Optional[Tuple[str, str]] = None) -> dict:
        """Get current statistics for all arms."""
        state = self._get_state(context)

        stats = {}
        for arm, s in state.items():
            count = s["count"]
            mean = s["total_reward"] / count if count > 0 else 0.0

            stats[arm] = {
                "mean": mean,
                "total_reward": s["total_reward"],
                "n_pulls": count
            }

        return stats

    def get_all_contexts(self) -> list[Tuple[str, str]]:
        """Get list of all contexts seen so far."""
        return list(self.contextual_state.keys())
