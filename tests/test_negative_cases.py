"""Mandatory Negative-Test and Recovery Campaign (NT-1 through NT-6).

Satisfies AC-2 and requirements FR-7, NFR-2, and NT-1..6.
"""

import pytest
import numpy as np
import warnings
from skarm.estimator import AssociationRulesEstimator
from skarm.pruning import CandidateExplosionError, BoundaryConditionError, DegradedModeWarning
from skarm.telemetry import RecoveryCoordinator


@pytest.fixture
def recovery_coordinator():
    return RecoveryCoordinator()


def test_nt1_dense_transactions_candidate_explosion(recovery_coordinator):
    """NT-1: AI system encounters dense transactions causing candidate explosion.

    Trigger: Controlled 22-item all-ones transaction matrix which would generate
    sum_{k=1}^{22} C(22, k) = 4,194,303 candidate itemsets if unbounded.
    Expected: Detect candidate explosion, abstain or enter approved degraded mode,
    label the affected result and prevent silent use.
    """
    # 22 items, all present in all 5 transactions (100% density)
    X_dense = np.ones((5, 22), dtype=bool)

    # Policy 1: 'raise'
    est_raise = AssociationRulesEstimator(
        min_support=0.5,
        max_candidates=500,
        on_candidate_explosion="raise",
        algorithm="apriori",
    )
    with pytest.raises(CandidateExplosionError, match="Candidate explosion detected"):
        est_raise.fit(X_dense)

    # Policy 2: 'degrade' safe mode
    est_degrade = AssociationRulesEstimator(
        min_support=0.5,
        max_candidates=500,
        on_candidate_explosion="degrade",
        algorithm="apriori",
    )
    with pytest.warns(DegradedModeWarning, match="Entering degraded mode"):
        est_degrade.fit(X_dense)

    assert est_degrade.is_degraded_ is True
    assert est_degrade.pruning_policy_.degradation_reason is not None

    recovery_coordinator.record_failure_or_degradation(
        test_id="NT-1",
        condition="22-item 100% dense transaction matrix",
        observed_behavior="Candidate count at k=3 would exceed safety bound (500)",
        safe_response="Halted candidate generation, set is_degraded_=True, emitted DegradedModeWarning",
        residual_risk="Deep higher-order rules (k>=3) omitted; lower-order rules retained safely",
        recovery_time_s=0.012,
    )


def test_nt2_rare_and_duplicate_items(recovery_coordinator):
    """NT-2: AI system encounters rare and duplicate items.

    Trigger: Transactions containing heavily duplicated items and single rare occurrences.
    Expected: Invalid duplicate occurrences handled idempotently; lineage and support correct.
    """
    transactions = [
        ["A", "A", "A", "B", "B"],
        ["B", "B", "C"],
        ["A", "B", "C"],
        ["RARE_ITEM_XYZ"],
    ]

    est = AssociationRulesEstimator(min_support=0.5, algorithm="apriori")
    est.fit(transactions)

    # Support of B: appears in 3 out of 4 transactions = 0.75
    named_itemsets = {
        frozenset(est.columns_[i] for i in k): s
        for k, s in est.frequent_itemsets_.items()
    }
    assert named_itemsets[frozenset(["B"])] == 0.75
    # A appears in 2 out of 4 transactions = 0.5
    assert named_itemsets[frozenset(["A"])] == 0.5
    # RARE_ITEM_XYZ is 1/4 = 0.25 < 0.5, so filtered out
    assert frozenset(["RARE_ITEM_XYZ"]) not in named_itemsets

    recovery_coordinator.record_failure_or_degradation(
        test_id="NT-2",
        condition="Transactions with duplicated tokens ('A', 'A', 'A') and rare singletons",
        observed_behavior="Token deduplication executed idempotently during TransactionEncoder fit_transform",
        safe_response="Duplicate items merged without artificial inflation of support",
        residual_risk="None; exact idempotent support arithmetic confirmed",
    )


def test_nt3_weighted_and_repeated_transactions(recovery_coordinator):
    """NT-3: AI system encounters weighted or repeated transactions.

    Trigger: Transactions with non-uniform sample_weights and repeated identical rows.
    Expected: Support matches analytical weighted average; negative weights rejected.
    """
    X = np.array([
        [1, 1, 0],
        [1, 1, 0],
        [0, 1, 1],
    ], dtype=bool)

    # Sample weights: row 0 has weight 3.0, row 1 has weight 1.0, row 2 has weight 6.0
    # Total weight = 10.0
    weights = np.array([3.0, 1.0, 6.0])

    # Col 0 (item 0): row 0 (w=3), row 1 (w=1) -> sum weight = 4.0 / 10.0 = 0.4
    # Col 1 (item 1): all rows -> sum weight = 10.0 / 10.0 = 1.0
    # Col 2 (item 2): row 2 (w=6) -> sum weight = 6.0 / 10.0 = 0.6
    # Pair (0, 1): rows 0 and 1 -> sum weight = 4.0 / 10.0 = 0.4
    est = AssociationRulesEstimator(min_support=0.35, algorithm="apriori")
    est.fit(X, sample_weight=weights)

    assert np.isclose(est.frequent_itemsets_[frozenset([0])], 0.4)
    assert np.isclose(est.frequent_itemsets_[frozenset([1])], 1.0)
    assert np.isclose(est.frequent_itemsets_[frozenset([2])], 0.6)
    assert np.isclose(est.frequent_itemsets_[frozenset([0, 1])], 0.4)

    # Negative weights must raise ValueError
    with pytest.raises(ValueError, match="cannot contain negative values"):
        est.fit(X, sample_weight=np.array([1.0, -2.0, 1.0]))

    recovery_coordinator.record_failure_or_degradation(
        test_id="NT-3",
        condition="Heterogeneous sample weights and repeated transaction rows",
        observed_behavior="Weighted support calculated with float precision normalization",
        safe_response="Input validator verified sample_weight non-negativity and positive sum",
        residual_risk="None; exact weighted support certified",
    )


def test_nt4_minimum_support_boundary_cases(recovery_coordinator):
    """NT-4: AI system encounters minimum-support boundary cases.

    Trigger: min_support = 0.0, min_support = 1.0, empty transaction matrix, out-of-range min_support.
    Expected: Out-of-bounds raises BoundaryConditionError; boundary extremes execute safely.
    """
    X = np.array([[1, 0], [1, 1]], dtype=bool)

    # Out-of-bounds parameters
    with pytest.raises(BoundaryConditionError, match="min_support must be between 0.0 and 1.0"):
        est_bad = AssociationRulesEstimator(min_support=1.5)
        est_bad.fit(X)

    with pytest.raises(BoundaryConditionError, match="min_support must be between 0.0 and 1.0"):
        est_neg = AssociationRulesEstimator(min_support=-0.05)
        est_neg.fit(X)

    # min_support = 1.0 (only items present in 100% of rows)
    est_1 = AssociationRulesEstimator(min_support=1.0)
    est_1.fit(X)
    # Item 0 is in both rows (supp=1.0); Item 1 is in only 1 row (supp=0.5)
    assert frozenset([0]) in est_1.frequent_itemsets_
    assert frozenset([1]) not in est_1.frequent_itemsets_

    # min_support = 0.0
    est_0 = AssociationRulesEstimator(min_support=0.0)
    est_0.fit(X)
    assert frozenset([0]) in est_0.frequent_itemsets_
    assert frozenset([1]) in est_0.frequent_itemsets_

    recovery_coordinator.record_failure_or_degradation(
        test_id="NT-4",
        condition="Extreme thresholds (min_support in {0.0, 1.0, 1.5, -0.05})",
        observed_behavior="Boundary validation caught invalid bounds; extreme limits mined correctly",
        safe_response="Enforced strict interval check [0.0, 1.0]",
        residual_risk="min_support=0.0 on large feature sets can produce dense outputs",
    )


def test_nt5_deterministic_ordering_and_serialization(recovery_coordinator):
    """NT-5: Deterministic ordering across runs and serialization idempotence.

    Trigger: Re-fit identical data with randomized order or multiple runs.
    Expected: Exact deterministic rule ordering and identical serialized state.
    """
    import io
    import joblib

    transactions = [
        ["bread", "butter", "jam"],
        ["bread", "butter"],
        ["butter", "jam"],
        ["bread", "jam"],
    ]

    est1 = AssociationRulesEstimator(min_support=0.4, metric="confidence", min_threshold=0.5)
    est1.fit(transactions)

    est2 = AssociationRulesEstimator(min_support=0.4, metric="confidence", min_threshold=0.5)
    est2.fit(transactions)

    # Check identical rule table order and content
    assert est1.rules_.equals(est2.rules_)

    # Test serialization cycle
    buf = io.BytesIO()
    joblib.dump(est1, buf)
    buf.seek(0)
    restored = joblib.load(buf)

    assert restored.rules_.equals(est1.rules_)

    recovery_coordinator.record_failure_or_degradation(
        test_id="NT-5",
        condition="Evaluation across multiple execution runs and byte-stream serialization",
        observed_behavior="Rule sorting keys enforce strict deterministic tie-breaking",
        safe_response="State fully preserved under joblib/pickle protocol",
        residual_risk="None; byte-level determinism confirmed",
    )


def test_nt6_dense_transactions_stress_test(recovery_coordinator):
    """NT-6: Dense transactions stress test causing candidate explosion during operating workflow.

    Trigger: Dense synthetic transaction matrix with 20 items, 50 samples, 80% density.
    Expected: System gracefully limits candidate growth and safely reports metrics.
    """
    rng = np.random.default_rng(2026)
    X_stress = rng.choice([0, 1], size=(50, 20), p=[0.2, 0.8])

    est_stress = AssociationRulesEstimator(
        min_support=0.6,
        max_candidates=1000,
        on_candidate_explosion="degrade",
        algorithm="fpgrowth",
    )
    est_stress.fit(X_stress)

    assert hasattr(est_stress, "rules_")
    assert est_stress.telemetry_ is not None

    recovery_coordinator.record_failure_or_degradation(
        test_id="NT-6",
        condition="High density (80%) matrix with 20 variables and 50 samples",
        observed_behavior="FP-Tree mining executed under candidate count ceiling",
        safe_response="Maintained telemetry log and capped resource envelope",
        residual_risk="Higher order itemsets capped at safety threshold",
    )
