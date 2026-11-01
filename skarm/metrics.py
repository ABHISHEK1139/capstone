"""Scikit-learn compatible unsupervised scoring metrics for association rule estimators."""

from typing import Any, Optional
import numpy as np


def mean_confidence_scorer(estimator: Any, X: Any = None, y: Optional[Any] = None) -> float:
    """Scorer calculating the mean confidence of discovered association rules.

    When held-out validation data X is provided, evaluates rules learned during fit()
    against X by computing:
        conf_val(X => Y) = supp_val(X U Y) / supp_val(X)
    Handles zero validation antecedent support explicitly by assigning 0.0 confidence.
    """
    if not hasattr(estimator, "rules_") or estimator.rules_.empty:
        return 0.0
    if X is None:
        return float(estimator.rules_["confidence"].mean())

    tx_sets = estimator._to_tx_sets(X)
    n_samples = len(tx_sets)
    if n_samples == 0:
        return 0.0

    confidences = []
    for _, row in estimator.rules_.iterrows():
        ante = set(row["antecedent"])
        cons = set(row["consequent"])
        union = ante | cons

        ante_count = sum(1 for tx in tx_sets if ante.issubset(tx))
        if ante_count == 0:
            confidences.append(0.0)
        else:
            union_count = sum(1 for tx in tx_sets if union.issubset(tx))
            confidences.append(union_count / ante_count)

    return float(np.mean(confidences)) if confidences else 0.0


def mean_lift_scorer(estimator: Any, X: Any = None, y: Optional[Any] = None) -> float:
    """Scorer calculating the mean lift of discovered association rules on held-out transactions X.

    When X is provided, computes:
        lift_val(X => Y) = supp_val(X U Y) / (supp_val(X) * supp_val(Y))
    Handles zero validation antecedent or consequent support explicitly by assigning 0.0.
    """
    if not hasattr(estimator, "rules_") or estimator.rules_.empty:
        return 0.0
    if X is None:
        return float(estimator.rules_["lift"].mean())

    tx_sets = estimator._to_tx_sets(X)
    n_samples = len(tx_sets)
    if n_samples == 0:
        return 0.0

    lifts = []
    for _, row in estimator.rules_.iterrows():
        ante = set(row["antecedent"])
        cons = set(row["consequent"])
        union = ante | cons

        ante_count = sum(1 for tx in tx_sets if ante.issubset(tx))
        cons_count = sum(1 for tx in tx_sets if cons.issubset(tx))
        if ante_count == 0 or cons_count == 0:
            lifts.append(0.0)
        else:
            union_count = sum(1 for tx in tx_sets if union.issubset(tx))
            lift_val = (union_count * n_samples) / (ante_count * cons_count)
            lifts.append(lift_val)

    return float(np.mean(lifts)) if lifts else 0.0


def total_support_scorer(estimator: Any, X: Any = None, y: Optional[Any] = None) -> float:
    """Scorer calculating total cumulative support of frequent itemsets on held-out transactions X."""
    if not hasattr(estimator, "frequent_itemsets_") or not estimator.frequent_itemsets_:
        return 0.0
    if X is None:
        return float(sum(estimator.frequent_itemsets_.values()))

    tx_sets = estimator._to_tx_sets(X)
    n_samples = len(tx_sets)
    if n_samples == 0:
        return 0.0

    total_supp = 0.0
    for itemset in estimator.frequent_itemsets_.keys():
        match_count = sum(1 for tx in tx_sets if itemset.issubset(tx))
        total_supp += match_count / n_samples

    return float(total_supp)


def rule_count_scorer(estimator: Any, X: Any = None, y: Optional[Any] = None) -> float:
    """Scorer calculating the count of discovered association rules that fire on validation data X."""
    if not hasattr(estimator, "rules_") or estimator.rules_.empty:
        return 0.0
    if X is None:
        return float(len(estimator.rules_))

    tx_sets = estimator._to_tx_sets(X)
    if not tx_sets:
        return 0.0

    active_rules = 0
    for _, row in estimator.rules_.iterrows():
        ante = set(row["antecedent"])
        if any(ante.issubset(tx) for tx in tx_sets):
            active_rules += 1

    return float(active_rules)


__all__ = [
    "mean_confidence_scorer",
    "mean_lift_scorer",
    "total_support_scorer",
    "rule_count_scorer",
]
