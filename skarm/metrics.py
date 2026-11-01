"""Scikit-learn compatible unsupervised scoring metrics for association rule estimators."""

from typing import Any, Optional


def mean_confidence_scorer(estimator: Any, X: Any = None, y: Optional[Any] = None) -> float:
    """Scorer calculating the mean confidence of discovered association rules."""
    if not hasattr(estimator, "rules_") or estimator.rules_.empty:
        return 0.0
    return float(estimator.rules_["confidence"].mean())


def mean_lift_scorer(estimator: Any, X: Any = None, y: Optional[Any] = None) -> float:
    """Scorer calculating the mean lift of discovered association rules."""
    if not hasattr(estimator, "rules_") or estimator.rules_.empty:
        return 0.0
    return float(estimator.rules_["lift"].mean())


def total_support_scorer(estimator: Any, X: Any = None, y: Optional[Any] = None) -> float:
    """Scorer calculating the total cumulative support of frequent itemsets."""
    if not hasattr(estimator, "frequent_itemsets_") or not estimator.frequent_itemsets_:
        return 0.0
    return float(sum(estimator.frequent_itemsets_.values()))


def rule_count_scorer(estimator: Any, X: Any = None, y: Optional[Any] = None) -> float:
    """Scorer calculating the count of discovered association rules."""
    if not hasattr(estimator, "rules_"):
        return 0.0
    return float(len(estimator.rules_))


__all__ = [
    "mean_confidence_scorer",
    "mean_lift_scorer",
    "total_support_scorer",
    "rule_count_scorer",
]
