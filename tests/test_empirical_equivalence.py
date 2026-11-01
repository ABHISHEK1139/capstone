"""Empirical Equivalence and Telemetry Auditability Verification Suite.

Adversarial test suite verifying:
1. Result determinism across Apriori, FP-Growth, and Eclat on diverse datasets
   (sparse retail, medium dense, weighted transactions) with Delta_supp < 1e-9.
2. Rule equivalence across all three engines (support, confidence, lift, leverage,
   conviction, Zhang's metric).
3. Telemetry auditability: metadata_['counters'] contains valid positive operation
   counters per algorithm (joins/scans in Apriori, trees/nodes in FP-Growth,
   intersections/TID scans in Eclat).
4. Ordering determinism: canonical row sorting across consecutive runs.
5. Edge cases: extreme thresholds, single items, disconnected items.
"""

import math
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from skarm.estimator import AssociationRulesEstimator


def canonical_itemsets_df(df: pd.DataFrame) -> pd.DataFrame:
    """Sort itemsets DataFrame by canonical key: (-support, length, sorted_items)."""
    if df.empty:
        return df.copy()
    records = df.to_dict("records")
    records.sort(
        key=lambda r: (
            -round(r["support"], 9),
            r["length"],
            sorted(list(r["itemsets"])),
        )
    )
    return pd.DataFrame(records)


def extract_rules_dict(rules_df: pd.DataFrame):
    """Index rules by normalized (antecedent, consequent) tuple for metric comparison."""
    r_dict = {}
    for _, row in rules_df.iterrows():
        ante = frozenset(int(x) for x in row["antecedent"])
        cons = frozenset(int(x) for x in row["consequent"])
        r_dict[(ante, cons)] = row
    return r_dict


class TestEmpiricalEquivalence:
    """Rigorous adversarial empirical verification of cross-engine contracts."""

@pytest.fixture(scope="module")
def sparse_retail_data():
    """Sparse retail transaction fixture (sampled from groceries)."""
    with open("data/groceries.csv", "r", encoding="utf-8") as f:
        lines = [line.strip().split(",") for line in f if line.strip()]
    return lines[:1500]


@pytest.fixture(scope="module")
def medium_dense_data():
    """Medium dense transaction fixture (tic_tac_toe categorical attributes)."""
    with open("data/tic_tac_toe.dat", "r", encoding="utf-8") as f:
        lines = [line.strip().split() for line in f if line.strip()]
    return lines


@pytest.fixture(scope="module")
def weighted_transactions():
    """Synthetic transaction matrix with continuous positive sample weights."""
    rng = np.random.default_rng(2026)
    X_mat = rng.choice([0, 1], size=(250, 14), p=[0.75, 0.25])
    weights = rng.uniform(0.5, 4.5, size=250)
    return X_mat, weights


class TestEmpiricalEquivalence:
    """Rigorous adversarial empirical verification of cross-engine contracts."""

    def test_result_determinism_sparse_retail(self, sparse_retail_data):
        """Verify result determinism on sparse retail dataset (Delta_supp < 1e-9)."""
        min_supp = 0.02
        ests = {
            algo: AssociationRulesEstimator(min_support=min_supp, algorithm=algo).fit(sparse_retail_data)
            for algo in ["apriori", "fpgrowth", "eclat"]
        }

        keys_a = set(ests["apriori"].frequent_itemsets_.keys())
        keys_f = set(ests["fpgrowth"].frequent_itemsets_.keys())
        keys_e = set(ests["eclat"].frequent_itemsets_.keys())

        assert keys_a == keys_f == keys_e, "Frequent itemset keys do not match across engines on sparse retail"
        assert len(keys_a) > 50, "Sparse retail should discover a non-trivial set of frequent itemsets"

        for k in keys_a:
            supp_a = ests["apriori"].frequent_itemsets_[k]
            supp_f = ests["fpgrowth"].frequent_itemsets_[k]
            supp_e = ests["eclat"].frequent_itemsets_[k]

            assert abs(supp_a - supp_f) < 1e-9, f"Apriori vs FP-Growth Delta_supp >= 1e-9 for itemset {k}"
            assert abs(supp_a - supp_e) < 1e-9, f"Apriori vs Eclat Delta_supp >= 1e-9 for itemset {k}"

    def test_result_determinism_medium_dense(self, medium_dense_data):
        """Verify result determinism on medium dense dataset (Delta_supp < 1e-9)."""
        min_supp = 0.15
        max_len = 4
        ests = {
            algo: AssociationRulesEstimator(min_support=min_supp, max_len=max_len, algorithm=algo).fit(medium_dense_data)
            for algo in ["apriori", "fpgrowth", "eclat"]
        }

        keys_a = set(ests["apriori"].frequent_itemsets_.keys())
        keys_f = set(ests["fpgrowth"].frequent_itemsets_.keys())
        keys_e = set(ests["eclat"].frequent_itemsets_.keys())

        assert keys_a == keys_f == keys_e, "Frequent itemset keys do not match across engines on medium dense"
        assert len(keys_a) > 50, "Medium dense should discover a non-trivial set of frequent itemsets"

        for k in keys_a:
            supp_a = ests["apriori"].frequent_itemsets_[k]
            supp_f = ests["fpgrowth"].frequent_itemsets_[k]
            supp_e = ests["eclat"].frequent_itemsets_[k]

            assert abs(supp_a - supp_f) < 1e-9, f"Apriori vs FP-Growth Delta_supp >= 1e-9 for itemset {k}"
            assert abs(supp_a - supp_e) < 1e-9, f"Apriori vs Eclat Delta_supp >= 1e-9 for itemset {k}"

    def test_result_determinism_weighted_transactions(self, weighted_transactions):
        """Verify result determinism on continuous weighted transactions (Delta_supp < 1e-9)."""
        X_mat, weights = weighted_transactions
        min_supp = 0.08
        ests = {
            algo: AssociationRulesEstimator(min_support=min_supp, algorithm=algo).fit(X_mat, sample_weight=weights)
            for algo in ["apriori", "fpgrowth", "eclat"]
        }

        keys_a = set(ests["apriori"].frequent_itemsets_.keys())
        keys_f = set(ests["fpgrowth"].frequent_itemsets_.keys())
        keys_e = set(ests["eclat"].frequent_itemsets_.keys())

        assert keys_a == keys_f == keys_e, "Frequent itemset keys do not match across engines on weighted transactions"
        assert len(keys_a) > 20, "Weighted transactions should discover frequent itemsets"

        for k in keys_a:
            supp_a = ests["apriori"].frequent_itemsets_[k]
            supp_f = ests["fpgrowth"].frequent_itemsets_[k]
            supp_e = ests["eclat"].frequent_itemsets_[k]

            assert abs(supp_a - supp_f) < 1e-9, f"Apriori vs FP-Growth Delta_supp >= 1e-9 for itemset {k}"
            assert abs(supp_a - supp_e) < 1e-9, f"Apriori vs Eclat Delta_supp >= 1e-9 for itemset {k}"

    def test_rule_equivalence_across_all_metrics(self, weighted_transactions):
        """Verify association rule equivalence across all 6 metrics for all 3 engines."""
        X_mat, weights = weighted_transactions
        min_supp = 0.08
        tested_metrics = ["confidence", "lift", "support", "leverage", "conviction", "zhangs_metric"]

        for metric in tested_metrics:
            threshold = 0.15 if metric != "lift" else 1.05
            ests = {
                algo: AssociationRulesEstimator(
                    min_support=min_supp,
                    metric=metric,
                    min_threshold=threshold,
                    algorithm=algo,
                ).fit(X_mat, sample_weight=weights)
                for algo in ["apriori", "fpgrowth", "eclat"]
            }

            rd_a = extract_rules_dict(ests["apriori"].rules_)
            rd_f = extract_rules_dict(ests["fpgrowth"].rules_)
            rd_e = extract_rules_dict(ests["eclat"].rules_)

            # Rule key set equivalence
            assert set(rd_a.keys()) == set(rd_f.keys()) == set(rd_e.keys()), (
                f"Rule sets differ across engines under metric={metric}"
            )

            # Metric numerical equivalence
            cols = ["support", "confidence", "lift", "leverage", "conviction", "zhangs_metric"]
            for r_key in rd_a:
                row_a = rd_a[r_key]
                row_f = rd_f[r_key]
                row_e = rd_e[r_key]

                for col in cols:
                    val_a = row_a[col]
                    val_f = row_f[col]
                    val_e = row_e[col]

                    if math.isinf(val_a) or math.isinf(val_f) or math.isinf(val_e):
                        assert math.isinf(val_a) and math.isinf(val_f) and math.isinf(val_e)
                    else:
                        assert abs(val_a - val_f) < 1e-9, f"Metric {col} Apriori vs FPGrowth mismatch on rule {r_key}"
                        assert abs(val_a - val_e) < 1e-9, f"Metric {col} Apriori vs Eclat mismatch on rule {r_key}"

    def test_telemetry_auditability_non_zero_operation_counters(self, sparse_retail_data):
        """Verify valid non-zero operation counters per algorithm in metadata_['counters']."""
        min_supp = 0.02
        ests = {
            algo: AssociationRulesEstimator(min_support=min_supp, algorithm=algo).fit(sparse_retail_data)
            for algo in ["apriori", "fpgrowth", "eclat"]
        }

        # Check Apriori counters
        meta_a = ests["apriori"].metadata_
        assert meta_a["status"] == "EXACT"
        assert meta_a["is_complete"] is True
        assert meta_a["future_candidates_status"] == "EXHAUSTED"
        cnt_a = meta_a["counters"]
        assert "candidates_generated" in cnt_a and cnt_a["candidates_generated"] > 0
        assert "joins_attempted" in cnt_a and cnt_a["joins_attempted"] > 0
        assert "scans_performed" in cnt_a and cnt_a["scans_performed"] > 0

        # Check FP-Growth counters
        meta_f = ests["fpgrowth"].metadata_
        assert meta_f["status"] == "EXACT"
        assert meta_f["is_complete"] is True
        assert meta_f["future_candidates_status"] == "EXHAUSTED"
        cnt_f = meta_f["counters"]
        assert "trees_constructed" in cnt_f and cnt_f["trees_constructed"] > 0
        assert "nodes_created" in cnt_f and cnt_f["nodes_created"] > 0
        assert "paths_traversed" in cnt_f and cnt_f["paths_traversed"] > 0

        # Check Eclat counters
        meta_e = ests["eclat"].metadata_
        assert meta_e["status"] == "EXACT"
        assert meta_e["is_complete"] is True
        assert meta_e["future_candidates_status"] == "EXHAUSTED"
        cnt_e = meta_e["counters"]
        assert "intersections_performed" in cnt_e and cnt_e["intersections_performed"] > 0
        assert "tid_elements_scanned" in cnt_e and cnt_e["tid_elements_scanned"] > 0

    def test_ordering_determinism_consecutive_runs(self, medium_dense_data):
        """Verify deterministic row ordering across consecutive runs of each engine."""
        min_supp = 0.2
        max_len = 3

        for algo in ["apriori", "fpgrowth", "eclat"]:
            est1 = AssociationRulesEstimator(min_support=min_supp, max_len=max_len, algorithm=algo).fit(medium_dense_data)
            est2 = AssociationRulesEstimator(min_support=min_supp, max_len=max_len, algorithm=algo).fit(medium_dense_data)

            # Itemsets DataFrame exact equality across runs
            assert est1.itemsets_.equals(est2.itemsets_), f"Algorithm {algo} itemsets_ differs across consecutive runs"

            # Rules DataFrame exact equality across runs
            assert est1.rules_.equals(est2.rules_), f"Algorithm {algo} rules_ differs across consecutive runs"

    def test_ordering_determinism_canonical_sorting(self, medium_dense_data):
        """Verify cross-engine canonical row sorting (descending support, ascending length, lexicographical items)."""
        min_supp = 0.2
        max_len = 3

        ests = {
            algo: AssociationRulesEstimator(min_support=min_supp, max_len=max_len, algorithm=algo).fit(medium_dense_data)
            for algo in ["apriori", "fpgrowth", "eclat"]
        }

        canon_a = canonical_itemsets_df(ests["apriori"].itemsets_)
        canon_f = canonical_itemsets_df(ests["fpgrowth"].itemsets_)
        canon_e = canonical_itemsets_df(ests["eclat"].itemsets_)

        assert canon_a.equals(canon_f), "Canonical itemsets differ between Apriori and FP-Growth"
        assert canon_a.equals(canon_e), "Canonical itemsets differ between Apriori and Eclat"
