"""
Metrics: Aggregates and computes KPIs from interaction logs.
Provides dashboard data and insights.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from app.models import InteractionLog


class MetricsAggregator:
    """Computes aggregate metrics from interaction logs."""

    @staticmethod
    def aggregate(logs: List[InteractionLog]) -> Dict[str, Any]:
        """
        Aggregate metrics from interaction logs.

        Args:
            logs: List of interaction logs

        Returns:
            Dictionary with aggregated metrics
        """
        if not logs:
            return MetricsAggregator._empty_metrics()

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([log.model_dump() for log in logs])

        # Extract score fields
        df['NPS_expected'] = df['score'].apply(lambda x: x['NPS_expected'])
        df['EngagementProb'] = df['score'].apply(lambda x: x['EngagementProb'])
        df['ChurnProb'] = df['score'].apply(lambda x: x['ChurnProb'])

        # Overall metrics
        metrics = {
            'n_interactions': len(df),
            'avg_reward': df['reward'].mean(),
            'avg_NPS': df['NPS_expected'].mean(),
            'avg_engagement': df['EngagementProb'].mean(),
            'avg_churn': df['ChurnProb'].mean(),
        }

        # By template (arm)
        arm_stats = df.groupby('arm').agg({
            'reward': ['mean', 'count'],
            'NPS_expected': 'mean',
            'EngagementProb': 'mean',
            'ChurnProb': 'mean'
        }).round(3)
        metrics['by_arm'] = arm_stats.to_dict()

        # By segment
        segment_stats = df.groupby('segment').agg({
            'reward': ['mean', 'count'],
            'NPS_expected': 'mean',
            'EngagementProb': 'mean',
            'ChurnProb': 'mean'
        }).round(3)
        metrics['by_segment'] = segment_stats.to_dict()

        # By issue bucket
        issue_stats = df.groupby('issue_bucket').agg({
            'reward': ['mean', 'count'],
            'NPS_expected': 'mean',
            'EngagementProb': 'mean',
            'ChurnProb': 'mean'
        }).round(3)
        metrics['by_issue'] = issue_stats.to_dict()

        # By interaction type
        type_stats = df.groupby('interaction_type').agg({
            'reward': ['mean', 'count'],
            'NPS_expected': 'mean',
            'EngagementProb': 'mean',
            'ChurnProb': 'mean'
        }).round(3)
        metrics['by_type'] = type_stats.to_dict()

        # Arm selection distribution
        arm_counts = df['arm'].value_counts().to_dict()
        total = len(df)
        arm_distribution = {arm: count/total for arm, count in arm_counts.items()}
        metrics['arm_distribution'] = arm_distribution

        # Evolution over iterations (if available)
        if 'iteration' in df.columns:
            iteration_stats = df.groupby('iteration').agg({
                'reward': 'mean',
                'NPS_expected': 'mean',
                'EngagementProb': 'mean',
                'ChurnProb': 'mean'
            }).round(3)
            metrics['by_iteration'] = iteration_stats.to_dict()

        # Top insights
        metrics['insights'] = MetricsAggregator._generate_insights(df)

        return metrics

    @staticmethod
    def _empty_metrics() -> Dict[str, Any]:
        """Return empty metrics structure."""
        return {
            'n_interactions': 0,
            'avg_reward': 0.0,
            'avg_NPS': 0.0,
            'avg_engagement': 0.0,
            'avg_churn': 0.0,
            'by_arm': {},
            'by_segment': {},
            'by_issue': {},
            'by_type': {},
            'arm_distribution': {},
            'by_iteration': {},
            'insights': []
        }

    @staticmethod
    def _generate_insights(df: pd.DataFrame) -> List[str]:
        """
        Generate textual insights from data.

        Args:
            df: DataFrame with interaction logs

        Returns:
            List of insight strings
        """
        insights = []

        # Best performing template overall
        arm_rewards = df.groupby('arm')['reward'].mean()
        if len(arm_rewards) > 0:
            best_arm = arm_rewards.idxmax()
            best_reward = arm_rewards.max()
            insights.append(
                f"Plantilla '{best_arm}' tiene el mejor desempeño global (reward: {best_reward:.3f})"
            )

        # Best segment-template combination
        if len(df) > 10:  # Only if enough data
            segment_arm = df.groupby(['segment', 'arm'])['reward'].mean()
            if len(segment_arm) > 0:
                best_combo = segment_arm.idxmax()
                best_combo_reward = segment_arm.max()
                insights.append(
                    f"Mejor combinación: {best_combo[0]} + {best_combo[1]} (reward: {best_combo_reward:.3f})"
                )

        # Issue bucket insights
        if 'issue_bucket' in df.columns:
            issue_arm = df.groupby(['issue_bucket', 'arm'])['reward'].mean()
            if len(issue_arm) > 0:
                best_issue_combo = issue_arm.idxmax()
                insights.append(
                    f"Para {best_issue_combo[0]}, usar '{best_issue_combo[1]}'"
                )

        # Churn risk insight
        avg_churn = df['ChurnProb'].mean()
        if avg_churn > 0.5:
            insights.append(
                f"⚠️ Riesgo de churn promedio alto ({avg_churn:.1%}), revisar estrategias de retención"
            )

        # Engagement insight
        avg_engagement = df['EngagementProb'].mean()
        if avg_engagement < 0.5:
            insights.append(
                f"💡 Engagement bajo ({avg_engagement:.1%}), considerar mensajes más accionables"
            )

        return insights[:5]  # Max 5 insights

    @staticmethod
    def compute_lift(
        treatment_logs: List[InteractionLog],
        baseline_logs: List[InteractionLog],
        metric: str = 'NPS_expected'
    ) -> float:
        """
        Compute lift vs baseline.

        Args:
            treatment_logs: Logs from bandit policy
            baseline_logs: Logs from baseline (e.g., uniform random)
            metric: Metric to compare

        Returns:
            Percentage lift
        """
        if not treatment_logs or not baseline_logs:
            return 0.0

        if metric == 'reward':
            treatment_val = np.mean([log.reward for log in treatment_logs])
            baseline_val = np.mean([log.reward for log in baseline_logs])
        elif metric == 'NPS_expected':
            treatment_val = np.mean([log.score.NPS_expected for log in treatment_logs])
            baseline_val = np.mean([log.score.NPS_expected for log in baseline_logs])
        else:
            return 0.0

        if baseline_val == 0:
            return 0.0

        lift = (treatment_val - baseline_val) / baseline_val
        return lift * 100  # percentage
