"""Tests for association rule generation and metric formulas."""

import math
import pytest
from skarm.rules import association_rules
from tests.fixtures.oracle_data import (
    ORACLE_FREQUENT_ITEMSETS_0_4,
    ORACLE_RULES_0_4_CONF_0_6,
)


def test_association_rules_metrics_against_oracle():
    # Convert item name oracle itemsets to numeric indices for rules function
    items = sorted(list({"A", "B", "C", "D", "E", "F"}))
    name_to_idx = {name: idx for idx, name in enumerate(items)}

    idx_itemsets = {
        frozenset(name_to_idx[n] for n in k): supp
        for k, supp in ORACLE_FREQUENT_ITEMSETS_0_4.items()
    }

    rules_df = association_rules(
        idx_itemsets,
        metric="confidence",
        min_threshold=0.6,
        feature_names=items,
    )

    # Convert DataFrame rules to dictionary key (frozenset(antecedent_names), frozenset(consequent_names))
    generated_rules = {}
    for _, row in rules_df.iterrows():
        key = (row["antecedent_names"], row["consequent_names"])
        generated_rules[key] = row.to_dict()

    # Verify all oracle rules exist
    assert set(generated_rules.keys()) == set(ORACLE_RULES_0_4_CONF_0_6.keys()), (
        f"Mismatch: generated {set(generated_rules.keys())} vs oracle {set(ORACLE_RULES_0_4_CONF_0_6.keys())}"
    )

    # Validate each metric to precision < 1e-9
    for (ante, cons), expected in ORACLE_RULES_0_4_CONF_0_6.items():
        actual = generated_rules[(ante, cons)]
        for metric_name, exp_val in expected.items():
            act_val = actual[metric_name]
            if math.isinf(exp_val):
                assert math.isinf(act_val), f"Rule {ante}->{cons} {metric_name} should be inf"
            else:
                assert math.isclose(act_val, exp_val, abs_tol=1e-9), (
                    f"Rule {ante}->{cons} {metric_name} expected {exp_val}, got {act_val}"
                )


def test_association_rules_lift_filtering():
    items = ["A", "B", "C"]
    frequent = {
        frozenset([0]): 0.5,
        frozenset([1]): 0.5,
        frozenset([0, 1]): 0.5,
    }
    # A->B: conf = 1.0, lift = 1.0 / 0.5 = 2.0
    rules_df = association_rules(frequent, metric="lift", min_threshold=1.5, feature_names=items)
    assert len(rules_df) == 2
    assert all(rules_df["lift"] >= 1.5)
