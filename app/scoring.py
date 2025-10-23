"""
Reward computation for bandit algorithm.
Formula: R = 0.6*NPS + 0.3*(Engagement*10) - 0.3*(Churn*10)
Normalized to [0, 1] range.
"""
from app.models import Score


def compute_reward(score: Score) -> float:
    """
    Compute reward from Judge score.

    Args:
        score: Score object from Judge evaluation

    Returns:
        Reward value in [0, 1] range
    """
    nps = float(score.NPS_expected)
    eng = float(score.EngagementProb)
    chrn = float(score.ChurnProb)

    # Raw reward formula
    r_raw = 0.6 * nps + 0.3 * (eng * 10) - 0.3 * (chrn * 10)

    # Normalize to [0, 1] range
    # Assuming r_raw range is approximately [0, 10]
    r = max(0.0, min(1.0, r_raw / 10.0))

    return r
