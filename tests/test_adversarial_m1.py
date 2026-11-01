"""Adversarial stress-testing and edge-case verification harness for Milestone M1.

Evaluates:
1. Boundary transactions: empty, 1-item, all-items, 10,000 identical transactions.
2. Adversarial inputs: negative values, continuous floats, NaNs, Infs, bad weights.
3. Resource budget enforcement: candidate caps, runtime budgets, degraded metadata.
4. Zhang's metric bounds: extreme negative associations, massive grid search.
"""

import time
import math
import warnings
import pytest
import numpy as np
import pandas as pd
from scipy import sparse

from skarm.estimator import AssociationRulesEstimator
from skarm.preprocessing import TransactionEncoder
from skarm.rules import compute_zhangs_metric, association_rules
from skarm.apriori import apriori
from skarm.fpgrowth import fpgrowth
from skarm.eclat import eclat
from skarm.pruning import (
    PruningPolicy,
    CandidateExplosionError,
    BoundaryConditionError,
    DegradedModeWarning,
)


# =====================================================================
# 1. Extreme Boundary Transactions
# =====================================================================

def test_adversarial_empty_transaction_list():
    """Verify empty transaction list raises ValueError cleanly."""
    est = AssociationRulesEstimator()
    with pytest.raises(ValueError, match="Empty transaction list"):
        est.fit([])


def test_adversarial_single_empty_transaction():
    """Verify single empty transaction [[]] fits safely without crashing."""
    est = AssociationRulesEstimator(min_support=0.1)
    est.fit([[]])
    assert est.status_ == "EXACT"
    assert est.is_complete_ is True
    assert est.is_degraded_ is False
    assert len(est.frequent_itemsets_) == 0
    assert est.rules_.empty
    assert est.get_feature_names_out().shape == (0,)
    assert est.score() == 0.0

    # Transform on empty transaction
    trans = est.transform([[]])
    assert trans.shape == (1, 0)


def test_adversarial_multiple_empty_transactions():
    """Verify [[], [], []] fits safely without crashing."""
    est = AssociationRulesEstimator(min_support=0.1)
    est.fit([[], [], []])
    assert est.status_ == "EXACT"
    assert est.is_complete_ is True
    assert est.rules_.empty
    assert est.itemsets_.empty


def test_adversarial_zero_sample_matrix():
    """Verify 0-sample 2D matrix (0, 5) fits cleanly returning empty itemsets."""
    for algo in ["apriori", "fpgrowth", "eclat"]:
        est = AssociationRulesEstimator(algorithm=algo, min_support=0.1)
        X_zero = np.zeros((0, 5), dtype=bool)
        est.fit(X_zero)
        assert est.status_ == "EXACT"
        assert est.is_complete_ is True
        assert len(est.frequent_itemsets_) == 0
        assert est.rules_.empty


def test_adversarial_zero_feature_matrix():
    """Verify 0-feature 2D matrix (5, 0) fits cleanly returning empty itemsets."""
    for algo in ["apriori", "fpgrowth", "eclat"]:
        est = AssociationRulesEstimator(algorithm=algo, min_support=0.1)
        X_zero = np.zeros((5, 0), dtype=bool)
        est.fit(X_zero)
        assert est.status_ == "EXACT"
        assert est.is_complete_ is True
        assert len(est.frequent_itemsets_) == 0
        assert est.rules_.empty


def test_adversarial_single_item_transactions():
    """Verify single item transactions: frequent itemsets found, 0 rules produced."""
    # 1 transaction with 1 item
    est1 = AssociationRulesEstimator(min_support=0.5)
    est1.fit([["A"]])
    assert est1.status_ == "EXACT"
    assert len(est1.frequent_itemsets_) == 1
    assert est1.rules_.empty
    assert est1.score() == 0.0

    # Multiple disjoint singleton transactions
    est2 = AssociationRulesEstimator(min_support=0.25)
    est2.fit([["A"], ["B"], ["C"], ["D"]])
    assert est2.status_ == "EXACT"
    assert len(est2.frequent_itemsets_) == 4
    # All frequent itemsets are size 1, so no rules can be formed
    assert est2.rules_.empty
    assert len(est2.get_feature_names_out()) == 0


def test_adversarial_all_items_dense_transactions():
    """Verify transactions where every item is present in every transaction."""
    # 15 items, 10 transactions, 100% density
    items = [f"item_{i}" for i in range(15)]
    X_dense = [items for _ in range(10)]

    for algo in ["apriori", "fpgrowth", "eclat"]:
        # Test with max_len=2 constraint
        est = AssociationRulesEstimator(
            algorithm=algo,
            min_support=1.0,
            max_len=2,
            min_threshold=0.5,
        )
        est.fit(X_dense)
        assert est.status_ == "EXACT"
        assert est.is_complete_ is True
        # 15 singletons + C(15, 2) pairs = 15 + 105 = 120 itemsets
        assert len(est.frequent_itemsets_) == 120
        # Each pair {A, B} generates 2 rules: A->B and B->A. 105 * 2 = 210 rules.
        assert len(est.rules_) == 210
        assert np.all(est.rules_["confidence"] == 1.0)
        assert np.all(est.rules_["support"] == 1.0)


def test_adversarial_10000_identical_transactions():
    """Verify 10,000 identical transactions are mined accurately and efficiently."""
    tx = [["laptop", "mouse", "keyboard"]] * 10000

    for algo in ["apriori", "fpgrowth", "eclat"]:
        t0 = time.perf_counter()
        est = AssociationRulesEstimator(
            algorithm=algo,
            min_support=0.5,
            min_threshold=0.5,
        )
        est.fit(tx)
        elapsed = time.perf_counter() - t0

        assert est.status_ == "EXACT"
        assert est.is_complete_ is True
        # 3 singletons, 3 pairs, 1 triple = 7 frequent itemsets
        assert len(est.frequent_itemsets_) == 7
        # Rules: 3 pairs * 2 = 6; 1 triple * (3 + 3) = 6 -> Total 12 rules
        assert len(est.rules_) == 12
        assert np.all(est.rules_["support"] == 1.0)
        assert np.all(est.rules_["confidence"] == 1.0)
        assert np.all(est.rules_["lift"] == 1.0)
        assert np.all(est.rules_["leverage"] == 0.0)
        assert np.all(np.isinf(est.rules_["conviction"]))
        assert np.all(est.rules_["zhangs_metric"] == 0.0)
        assert elapsed < 5.0, f"Algorithm {algo} took too long ({elapsed:.2f}s) on 10k identical transactions"


# =====================================================================
# 2. Adversarial Inputs & Invalid Input Rejection
# =====================================================================

def test_adversarial_negative_values_rejected():
    """Verify negative values in dense and sparse matrices raise ValueError."""
    bad_dense = np.array([[-1.0, 0.0], [0.0, 1.0]])
    bad_sparse = sparse.csr_matrix([[-0.5, 0.0], [0.0, 1.0]])

    for miner in [apriori, fpgrowth, eclat]:
        with pytest.raises(ValueError, match="strictly binary"):
            miner(bad_dense)
        with pytest.raises(ValueError, match="strictly binary"):
            miner(bad_sparse)

    for algo in ["apriori", "fpgrowth", "eclat"]:
        est = AssociationRulesEstimator(algorithm=algo)
        with pytest.raises(ValueError, match="strictly binary"):
            est.fit(bad_dense)
        with pytest.raises(ValueError, match="strictly binary"):
            est.fit(bad_sparse)


def test_adversarial_continuous_floats_rejected():
    """Verify continuous non-binary floating point matrices raise ValueError."""
    bad_dense = np.array([[0.5, 0.2], [0.0, 1.0]])
    bad_sparse = sparse.csr_matrix([[0.7, 0.0], [0.0, 1.0]])

    for miner in [apriori, fpgrowth, eclat]:
        with pytest.raises(ValueError, match="strictly binary"):
            miner(bad_dense)
        with pytest.raises(ValueError, match="strictly binary"):
            miner(bad_sparse)

    for algo in ["apriori", "fpgrowth", "eclat"]:
        est = AssociationRulesEstimator(algorithm=algo)
        with pytest.raises(ValueError, match="strictly binary"):
            est.fit(bad_dense)
        with pytest.raises(ValueError, match="strictly binary"):
            est.fit(bad_sparse)


def test_adversarial_nans_in_matrices_rejected():
    """Verify matrices containing NaNs raise ValueError cleanly."""
    bad_dense = np.array([[np.nan, 1.0], [0.0, 1.0]])
    bad_sparse = sparse.csr_matrix(np.array([[np.nan, 0.0], [0.0, 1.0]]))

    for miner in [apriori, fpgrowth, eclat]:
        with pytest.raises(ValueError, match="NaNs"):
            miner(bad_dense)
        with pytest.raises(ValueError, match="NaNs"):
            miner(bad_sparse)

    for algo in ["apriori", "fpgrowth", "eclat"]:
        est = AssociationRulesEstimator(algorithm=algo)
        with pytest.raises(ValueError, match="NaNs"):
            est.fit(bad_dense)
        with pytest.raises(ValueError, match="NaNs"):
            est.fit(bad_sparse)


def test_adversarial_infs_in_matrices_rejected():
    """Verify matrices containing Infs raise ValueError cleanly."""
    bad_dense_pos = np.array([[np.inf, 1.0], [0.0, 1.0]])
    bad_dense_neg = np.array([[-np.inf, 1.0], [0.0, 1.0]])
    bad_sparse = sparse.csr_matrix(bad_dense_pos)

    for miner in [apriori, fpgrowth, eclat]:
        with pytest.raises(ValueError, match="infinite"):
            miner(bad_dense_pos)
        with pytest.raises(ValueError, match="infinite"):
            miner(bad_dense_neg)
        with pytest.raises(ValueError, match="infinite"):
            miner(bad_sparse)


def test_adversarial_sample_weight_validation():
    """Verify sample_weight invalid inputs (NaN, Inf, negative, zero-sum, length mismatch)."""
    X = np.array([[1, 0], [0, 1]], dtype=bool)

    for miner in [apriori, fpgrowth, eclat]:
        # Length mismatch
        with pytest.raises(ValueError, match="sample_weight length"):
            miner(X, sample_weight=np.array([1.0, 1.0, 1.0]))

        # NaN
        with pytest.raises(ValueError, match="NaN"):
            miner(X, sample_weight=np.array([np.nan, 1.0]))

        # Inf
        with pytest.raises(ValueError, match="infinite"):
            miner(X, sample_weight=np.array([np.inf, 1.0]))

        # Negative Inf
        with pytest.raises(ValueError, match="infinite"):
            miner(X, sample_weight=np.array([-np.inf, 1.0]))

        # Negative weight
        with pytest.raises(ValueError, match="negative"):
            miner(X, sample_weight=np.array([-1.0, 2.0]))

        # Zero sum weights
        with pytest.raises(ValueError, match="strictly positive"):
            miner(X, sample_weight=np.array([0.0, 0.0]))


# =====================================================================
# 3. Resource Budget Enforcement & Degradation Telemetry
# =====================================================================

def test_adversarial_candidate_budget_degradation():
    """Verify candidate budget enforcement across Apriori, FP-Growth, and Eclat."""
    # Dense transaction matrix: 18 items all present
    X_dense = np.ones((10, 18), dtype=bool)

    for algo in ["apriori", "fpgrowth", "eclat"]:
        est = AssociationRulesEstimator(
            algorithm=algo,
            min_support=0.5,
            max_candidates=100,
            on_candidate_explosion="degrade",
        )
        with pytest.warns(DegradedModeWarning, match="Entering degraded mode"):
            est.fit(X_dense)

        # Assert required tri-state and telemetry attributes
        assert est.status_ == "DEGRADED", f"{algo}: status_ is {est.status_}"
        assert est.is_complete_ is False, f"{algo}: is_complete_ must be False"
        assert est.is_degraded_ is True, f"{algo}: is_degraded_ must be True"
        assert est.termination_reason_ is not None, f"{algo}: termination_reason_ must be set"
        assert "Candidate explosion" in est.termination_reason_

        # Verify metadata dictionary
        assert est.metadata_["status"] == "DEGRADED"
        assert est.metadata_["is_complete"] is False
        assert est.metadata_["is_candidate_count_exact"] is False
        assert est.metadata_["future_candidates_status"] in ("UNKNOWN", "EXHAUSTED")
        assert est.metadata_["level_halted"] is not None
        assert est.metadata_["frequent_itemsets_discovered"] > 0
        assert "counters" in est.metadata_


def test_adversarial_candidate_budget_raise():
    """Verify on_candidate_explosion='raise' raises CandidateExplosionError cleanly."""
    X_dense = np.ones((10, 18), dtype=bool)

    for algo in ["apriori", "fpgrowth", "eclat"]:
        est = AssociationRulesEstimator(
            algorithm=algo,
            min_support=0.5,
            max_candidates=100,
            on_candidate_explosion="raise",
        )
        with pytest.raises(CandidateExplosionError, match="Candidate explosion detected"):
            est.fit(X_dense)


def test_adversarial_runtime_budget_degradation():
    """Verify runtime budget max_time_seconds triggers graceful degradation."""
    # Construct a dataset requiring iterative mining
    rng = np.random.default_rng(42)
    X_med = rng.choice([0, 1], size=(200, 16), p=[0.3, 0.7])

    for algo in ["apriori", "fpgrowth", "eclat"]:
        est = AssociationRulesEstimator(
            algorithm=algo,
            min_support=0.2,
            max_time_seconds=0.0001,  # extremely tight budget: 0.1 ms
            on_candidate_explosion="degrade",
        )
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            est.fit(X_med)

        # If execution took longer than 0.1 ms, degraded mode was triggered
        if est.is_degraded_:
            assert est.status_ == "DEGRADED"
            assert est.is_complete_ is False
            assert "Time budget exceeded" in est.termination_reason_
            assert est.metadata_["future_candidates_status"] == "UNKNOWN"
            assert any(issubclass(w.category, DegradedModeWarning) for w in recorded_warnings)


# =====================================================================
# 4. Zhang's Metric Bounds & Extreme Negative Association
# =====================================================================

def test_adversarial_zhangs_metric_synthetic_extreme_negative():
    """Construct synthetic transactions with extreme negative association and test Zhang's metric.

    Specifically tests supp_ac << supp_a * supp_c.
    """
    # 1,000 transactions:
    # A appears in 400 transactions (supp_a = 0.40)
    # C appears in 400 transactions (supp_c = 0.40)
    # Expected independent co-occurrence: 0.4 * 0.4 = 0.16 (160 transactions)
    # Adversarial condition: A and C only co-occur in 1 transaction (supp_ac = 0.001)
    # Highly negative association: supp_ac << supp_a * supp_c
    tx = []
    # 1 transaction with both A and C
    tx.append(["A", "C"])
    # 399 transactions with A only
    for _ in range(399):
        tx.append(["A"])
    # 399 transactions with C only
    for _ in range(399):
        tx.append(["C"])
    # 201 transactions with neither
    for _ in range(201):
        tx.append(["B"])

    est = AssociationRulesEstimator(
        min_support=0.0005,
        metric="zhangs_metric",
        min_threshold=-1.0,  # collect all rules
    )
    est.fit(tx)

    rule_ac = est.rules_[
        (est.rules_["antecedent_names"] == frozenset(["A"])) &
        (est.rules_["consequent_names"] == frozenset(["C"]))
    ]
    assert len(rule_ac) == 1
    z_val = rule_ac["zhangs_metric"].iloc[0]
    assert -1.0 <= z_val <= 1.0, f"Zhang's metric out of bounds: {z_val}"
    assert z_val < -0.9, f"Expected strong negative association, got {z_val}"


def test_adversarial_zhangs_metric_exhaustive_grid_stress():
    """Stress test compute_zhangs_metric over 100,000 probability triples.

    Ensures that for all valid probability configurations:
    - -1.0 <= zhangs_metric <= +1.0
    - No NaNs or Infs
    - Correct limits for disjoint, independent, and nested sets
    """
    rng = np.random.default_rng(2026)

    # 1. Random valid triples: 0 <= supp_ac <= min(supp_a, supp_c)
    n_samples = 50000
    supp_a = rng.uniform(1e-6, 1.0, size=n_samples)
    supp_c = rng.uniform(1e-6, 1.0, size=n_samples)
    max_ac = np.minimum(supp_a, supp_c)
    supp_ac = rng.uniform(0.0, max_ac)

    for sa, sc, sac in zip(supp_a, supp_c, supp_ac):
        val = compute_zhangs_metric(sac, sa, sc)
        assert not math.isnan(val), f"NaN produced for sa={sa}, sc={sc}, sac={sac}"
        assert not math.isinf(val), f"Inf produced for sa={sa}, sc={sc}, sac={sac}"
        assert -1.0 <= val <= 1.0, f"Value {val} out of [-1, 1] for sa={sa}, sc={sc}, sac={sac}"

    # 2. Extreme boundary triples:
    # supp_ac = 0.0 (completely disjoint)
    for sa in [0.001, 0.01, 0.1, 0.5, 0.999]:
        for sc in [0.001, 0.01, 0.1, 0.5, 0.999]:
            val = compute_zhangs_metric(0.0, sa, sc)
            assert np.isclose(val, -1.0), f"Expected -1.0 for disjoint sets, got {val}"

    # supp_ac = sa * sc (exact statistical independence)
    for sa in [0.1, 0.2, 0.5, 0.8]:
        for sc in [0.1, 0.3, 0.5, 0.7]:
            val = compute_zhangs_metric(sa * sc, sa, sc)
            assert np.isclose(val, 0.0, atol=1e-10), f"Expected 0.0 for independent sets, got {val}"

    # supp_ac = min(sa, sc) (complete subset inclusion)
    for sa in [0.2, 0.5, 0.8]:
        for sc in [0.2, 0.5, 0.8]:
            sac = min(sa, sc)
            val = compute_zhangs_metric(sac, sa, sc)
            assert -1.0 <= val <= 1.0
            if sa < 1.0 or sc < 1.0:
                assert val > 0.0, f"Expected positive association, got {val}"


def test_adversarial_rule_partition_disjointness():
    """Verify all generated rules satisfy valid partition: X, Y non-empty, X cap Y = empty."""
    tx = [
        ["A", "B", "C", "D"],
        ["A", "B", "C"],
        ["A", "B", "D"],
        ["B", "C", "D"],
        ["A", "B"],
    ]
    est = AssociationRulesEstimator(min_support=0.3, min_threshold=0.1)
    est.fit(tx)

    assert not est.rules_.empty
    for _, row in est.rules_.iterrows():
        ante = row["antecedent"]
        cons = row["consequent"]
        assert len(ante) > 0, "Antecedent must be non-empty"
        assert len(cons) > 0, "Consequent must be non-empty"
        assert ante.isdisjoint(cons), f"Antecedent and consequent must be disjoint: {ante} & {cons}"
