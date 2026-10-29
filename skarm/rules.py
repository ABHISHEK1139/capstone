"""Association rule generation and evaluation metrics."""

import math
from itertools import combinations
from typing import Dict, List, FrozenSet, Optional, Union
import pandas as pd
import numpy as np


def compute_leverage(supp_ac: float, supp_a: float, supp_c: float) -> float:
    """Compute leverage = supp(A U C) - (supp(A) * supp(C))."""
    return float(supp_ac - (supp_a * supp_c))


def compute_conviction(supp_c: float, confidence: float) -> float:
    """Compute conviction = (1 - supp(C)) / (1 - confidence)."""
    if math.isclose(confidence, 1.0, abs_tol=1e-12) or confidence >= 1.0:
        return float("inf")
    denom = 1.0 - confidence
    if denom <= 0.0:
        return float("inf")
    return float((1.0 - supp_c) / denom)


def compute_zhangs_metric(supp_ac: float, supp_a: float, supp_c: float) -> float:
    """Compute Zhang's association metric (-1 to +1)."""
    num = supp_ac - (supp_a * supp_c)
    if math.isclose(num, 0.0, abs_tol=1e-12):
        return 0.0

    if num > 0:
        denom = max(
            supp_ac * (1.0 - supp_a),
            supp_a * (supp_c - supp_ac),
        )
    else:
        denom = max(
            supp_ac * (supp_a - supp_ac),
            supp_a * (1.0 - supp_c),
        )

    if denom <= 1e-12:
        return 0.0
    return float(num / denom)


def association_rules(
    frequent_itemsets: Dict[FrozenSet[int], float],
    metric: str = "confidence",
    min_threshold: float = 0.5,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Generate association rules from frequent itemsets.

    Parameters
    ----------
    frequent_itemsets : dict
        Mapping from frozenset of item indices to support values.
    metric : str, default='confidence'
        Metric used for rule filtering ('confidence', 'lift', 'support', 'leverage', 'conviction').
    min_threshold : float, default=0.5
        Minimum metric threshold to keep a rule.
    feature_names : list of str, optional
        Names of features/items for human-readable rule representations.

    Returns
    -------
    rules : pd.DataFrame
        DataFrame with columns:
        - antecedent
        - consequent
        - antecedent_support
        - consequent_support
        - support
        - confidence
        - lift
        - leverage
        - conviction
        - zhangs_metric
    """
    valid_metrics = {"confidence", "lift", "support", "leverage", "conviction", "zhangs_metric"}
    if metric not in valid_metrics:
        raise ValueError(f"Unknown metric '{metric}'. Choose from {valid_metrics}")

    records = []

    # Filter itemsets with size >= 2
    multi_itemsets = [
        (itemset, supp)
        for itemset, supp in frequent_itemsets.items()
        if len(itemset) >= 2
    ]

    # Deterministic processing (NT-5)
    multi_itemsets.sort(key=lambda x: (len(x[0]), sorted(list(x[0]))))

    for itemset, supp_ac in multi_itemsets:
        items = sorted(list(itemset))
        k = len(items)

        # Generate all non-empty proper subsets as antecedents
        for r in range(1, k):
            for ante_combo in combinations(items, r):
                ante = frozenset(ante_combo)
                cons = itemset - ante

                supp_a = frequent_itemsets.get(ante)
                supp_c = frequent_itemsets.get(cons)

                # Skip if support of antecedent or consequent is not in frequent itemsets
                if supp_a is None or supp_c is None:
                    continue

                if supp_a <= 0.0:
                    continue

                conf = supp_ac / supp_a
                lift = conf / supp_c if supp_c > 0.0 else 0.0
                lev = compute_leverage(supp_ac, supp_a, supp_c)
                conv = compute_conviction(supp_c, conf)
                zhang = compute_zhangs_metric(supp_ac, supp_a, supp_c)

                # Metric filtering
                metric_val = {
                    "confidence": conf,
                    "lift": lift,
                    "support": supp_ac,
                    "leverage": lev,
                    "conviction": conv,
                    "zhangs_metric": zhang,
                }[metric]

                if metric_val >= min_threshold - 1e-12:
                    records.append({
                        "antecedent": ante,
                        "consequent": cons,
                        "antecedent_support": float(supp_a),
                        "consequent_support": float(supp_c),
                        "support": float(supp_ac),
                        "confidence": float(conf),
                        "lift": float(lift),
                        "leverage": float(lev),
                        "conviction": float(conv),
                        "zhangs_metric": float(zhang),
                    })

    if not records:
        df = pd.DataFrame(columns=[
            "antecedent", "consequent", "antecedent_support", "consequent_support",
            "support", "confidence", "lift", "leverage", "conviction", "zhangs_metric"
        ])
        return df

    df = pd.DataFrame(records)

    # Deterministic sorting (NT-5): sort by metric descending, then support descending, then antecedent/consequent
    df.sort_values(
        by=[metric, "support"],
        ascending=[False, False],
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)

    # Add named antecedent/consequent if feature_names provided
    if feature_names is not None:
        def to_names(f_set):
            return frozenset(feature_names[i] for i in f_set)

        df["antecedent_names"] = df["antecedent"].apply(to_names)
        df["consequent_names"] = df["consequent"].apply(to_names)

    return df
