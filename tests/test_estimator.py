"""Tests for AssociationRulesEstimator scikit-learn compatibility."""

import io
import joblib
import pytest
import numpy as np
from sklearn.base import clone
from skarm.estimator import AssociationRulesEstimator
from skarm.pruning import DegradedModeWarning, CandidateExplosionError, get_current_memory_mb
from tests.fixtures.oracle_data import HAND_COMPUTED_TRANSACTIONS


def test_estimator_basic_fit_predict():
    est = AssociationRulesEstimator(min_support=0.4, metric="confidence", min_threshold=0.6)
    est.fit(HAND_COMPUTED_TRANSACTIONS)

    assert hasattr(est, "rules_")
    assert hasattr(est, "frequent_itemsets_")
    assert not est.rules_.empty

    # Predict recommendations on query transactions
    query = [["A", "C"], ["B"]]
    preds = est.predict(query)
    assert isinstance(preds, list)
    assert len(preds) == 2
    # In ['A', 'C'], rule {A, C} -> B has confidence 2/3, so B should be recommended!
    assert "B" in preds[0]


def test_estimator_transform():
    est = AssociationRulesEstimator(min_support=0.4, metric="confidence", min_threshold=0.6)
    est.fit(HAND_COMPUTED_TRANSACTIONS)
    trans = est.transform(HAND_COMPUTED_TRANSACTIONS)
    assert trans.shape == (len(HAND_COMPUTED_TRANSACTIONS), len(est.rules_))


def test_estimator_sklearn_clone():
    est = AssociationRulesEstimator(min_support=0.3, metric="lift", min_threshold=1.2, algorithm="fpgrowth")
    cloned = clone(est)
    assert cloned.min_support == 0.3
    assert cloned.metric == "lift"
    assert cloned.min_threshold == 1.2
    assert cloned.algorithm == "fpgrowth"
    assert not hasattr(cloned, "rules_")


def test_estimator_serialization_joblib():
    """Verify joblib dump and load preserves exact rule table and parameters (NT-5)."""
    est = AssociationRulesEstimator(min_support=0.4, metric="confidence", min_threshold=0.6)
    est.fit(HAND_COMPUTED_TRANSACTIONS)

    buffer = io.BytesIO()
    joblib.dump(est, buffer)
    buffer.seek(0)
    loaded_est = joblib.load(buffer)

    assert loaded_est.min_support == est.min_support
    assert len(loaded_est.rules_) == len(est.rules_)
    assert loaded_est.frequent_itemsets_ == est.frequent_itemsets_


def test_estimator_get_feature_names_out():
    """Verify Bug B8: get_feature_names_out() returns formatted rule strings."""
    est = AssociationRulesEstimator(min_support=0.4, metric="confidence", min_threshold=0.6)
    est.fit(HAND_COMPUTED_TRANSACTIONS)

    feature_names = est.get_feature_names_out()
    assert isinstance(feature_names, np.ndarray)
    assert len(feature_names) == len(est.rules_)
    assert all("->" in str(name) for name in feature_names)

    # Empty rules check
    est_empty = AssociationRulesEstimator(min_support=0.99)
    est_empty.fit(HAND_COMPUTED_TRANSACTIONS)
    assert len(est_empty.get_feature_names_out()) == 0


def test_estimator_score_and_metrics():
    """Verify Bug B9 & R3: .score() and skarm.metrics scorers for hyperparameter tuning."""
    from skarm.metrics import (
        mean_confidence_scorer,
        mean_lift_scorer,
        total_support_scorer,
        rule_count_scorer,
    )

    est = AssociationRulesEstimator(min_support=0.4, metric="confidence", min_threshold=0.6)
    est.fit(HAND_COMPUTED_TRANSACTIONS)

    # Default score: mean confidence
    score_val = est.score()
    expected_mean_conf = float(est.rules_["confidence"].mean())
    assert np.isclose(score_val, expected_mean_conf)

    # Scorer functions
    assert np.isclose(mean_confidence_scorer(est), expected_mean_conf)
    assert np.isclose(mean_lift_scorer(est), float(est.rules_["lift"].mean()))
    assert np.isclose(total_support_scorer(est), float(sum(est.frequent_itemsets_.values())))
    assert rule_count_scorer(est) == float(len(est.rules_))

    # Test scoring parameter configurations
    est_lift = AssociationRulesEstimator(min_support=0.4, scoring="mean_lift")
    est_lift.fit(HAND_COMPUTED_TRANSACTIONS)
    assert np.isclose(est_lift.score(), float(est_lift.rules_["lift"].mean()))


def test_estimator_formal_contracts_r2():
    """Verify R2: status_ in {EXACT, DEGRADED, FAILED}, itemsets_ DataFrame, structured metadata_."""
    import pandas as pd
    est = AssociationRulesEstimator(min_support=0.4, algorithm="apriori")
    est.fit(HAND_COMPUTED_TRANSACTIONS)

    # Tri-state attributes
    assert est.status_ == "EXACT"
    assert est.is_complete_ is True
    assert est.is_degraded_ is False
    assert est.termination_reason_ is None

    # itemsets_ DataFrame
    assert hasattr(est, "itemsets_")
    assert isinstance(est.itemsets_, pd.DataFrame)
    assert "itemsets" in est.itemsets_.columns
    assert "support" in est.itemsets_.columns
    assert "length" in est.itemsets_.columns
    assert len(est.itemsets_) == len(est.frequent_itemsets_)

    # metadata_ structured dictionary
    assert hasattr(est, "metadata_")
    meta = est.metadata_
    assert meta["status"] == "EXACT"
    assert meta["is_complete"] is True
    assert meta["termination_reason"] is None
    assert meta["future_candidates_status"] == "EXHAUSTED"
    assert meta["frequent_itemsets_discovered"] == len(est.frequent_itemsets_)
    assert meta["is_candidate_count_exact"] is True
    assert "counters" in meta
    assert meta["counters"]["candidates_generated"] > 0
    assert meta["counters"]["joins_attempted"] > 0


def test_estimator_sklearn_tags():
    """Verify R3: __sklearn_tags__() declared on Estimator and TransactionEncoder."""
    from skarm.preprocessing import TransactionEncoder

    est = AssociationRulesEstimator()
    tags = est.__sklearn_tags__()
    assert tags.target_tags.required is False
    assert tags.input_tags.two_d_array is True

    te = TransactionEncoder()
    te_tags = te.__sklearn_tags__()
    assert te_tags.target_tags.required is False


def test_estimator_algorithm_counters():
    """Verify R3: algorithm-specific counters populated in metadata_['counters']."""
    for algo, expected_keys in [
        ("apriori", {"candidates_generated", "joins_attempted", "scans_performed"}),
        ("fpgrowth", {"trees_constructed", "nodes_created", "paths_traversed"}),
        ("eclat", {"intersections_performed", "tid_elements_scanned"}),
    ]:
        est = AssociationRulesEstimator(min_support=0.4, algorithm=algo)
        est.fit(HAND_COMPUTED_TRANSACTIONS)
        counters = est.metadata_["counters"]
        for key in expected_keys:
            assert key in counters, f"Missing counter {key} for algorithm {algo}"
            assert counters[key] >= 0, f"Counter {key} must be non-negative"


def test_estimator_multidimensional_budgets():
    """Verify R3: multi-dimensional resource budgets (candidates, time, memory)."""
    from skarm.pruning import BoundaryConditionError

    # Invalid negative budget raises BoundaryConditionError
    with pytest.raises(BoundaryConditionError, match="max_time_seconds must be positive"):
        AssociationRulesEstimator(max_time_seconds=-1.0).fit(HAND_COMPUTED_TRANSACTIONS)

    with pytest.raises(BoundaryConditionError, match="max_memory_mb must be positive"):
        AssociationRulesEstimator(max_memory_mb=-5.0).fit(HAND_COMPUTED_TRANSACTIONS)

    # Valid parameters store correctly and clone cleanly
    est = AssociationRulesEstimator(max_time_seconds=60.0, max_memory_mb=512.0)
    assert est.max_time_seconds == 60.0
    assert est.max_memory_mb == 512.0
    cloned = clone(est)
    assert cloned.max_time_seconds == 60.0
    assert cloned.max_memory_mb == 512.0


def test_estimator_pipeline_pandas_output():
    """Verify Pipeline with .set_output(transform='pandas') outputs DataFrame with rule column names (R3)."""
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import VarianceThreshold

    pipe = Pipeline([
        ("rules", AssociationRulesEstimator(min_support=0.4, metric="confidence", min_threshold=0.6)),
        ("sel", VarianceThreshold(0.01)),
    ])
    pipe.set_output(transform="pandas")
    df = pipe.fit_transform(HAND_COMPUTED_TRANSACTIONS)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == len(HAND_COMPUTED_TRANSACTIONS)
    assert df.shape[1] > 0
    assert all("->" in str(col) for col in df.columns)


def test_estimator_pipeline_feature_names_out_provenance():
    """Verify Pipeline get_feature_names_out() propagates upstream item names (R3)."""
    from sklearn.pipeline import Pipeline
    from skarm.preprocessing import TransactionEncoder

    pipe = Pipeline([
        ("encoder", TransactionEncoder(sparse_output=False)),
        ("rules", AssociationRulesEstimator(min_support=0.4, metric="confidence", min_threshold=0.6)),
    ])
    pipe.fit(HAND_COMPUTED_TRANSACTIONS)
    feature_names = pipe.get_feature_names_out()

    assert isinstance(feature_names, np.ndarray)
    assert len(feature_names) > 0
    # Rule names should contain original items 'A', 'B', 'C', not generic 'item_0'
    assert any("{C} -> {A}" in str(n) for n in feature_names)
    assert not any("item_" in str(n) for n in feature_names)

    # Test length mismatch raises ValueError
    rules_est = pipe.named_steps["rules"]
    with pytest.raises(ValueError, match="input_features has length 2, but 6 features were seen"):
        rules_est.get_feature_names_out(input_features=["X", "Y"])


def test_estimator_scoring_validation():
    """Verify scoring parameter validation in _validate_parameters (R3)."""
    with pytest.raises(ValueError, match="Invalid scoring metric: invalid_metric"):
        AssociationRulesEstimator(scoring="invalid_metric").fit([["A", "B"]])

    # Valid string scorers
    for valid_scoring in ["mean_confidence", "mean_lift", "total_support", "rule_count"]:
        est = AssociationRulesEstimator(scoring=valid_scoring, min_support=0.4)
        est.fit(HAND_COMPUTED_TRANSACTIONS)
        score_val = est.score()
        assert isinstance(score_val, float)

    # Valid callable scorer
    custom_scorer = lambda estimator, X=None, y=None: 42.0
    est_custom = AssociationRulesEstimator(scoring=custom_scorer, min_support=0.4)
    est_custom.fit(HAND_COMPUTED_TRANSACTIONS)
    assert est_custom.score() == 42.0


def test_fpgrowth_truncate_degraded_metadata():
    """Verify FP-Growth truncation sets future_candidates_status='UNKNOWN' and uniform degradation_reason (R6)."""
    X = (np.random.RandomState(42).rand(50, 10) > 0.3).astype(int)
    est = AssociationRulesEstimator(
        min_support=0.1,
        algorithm="fpgrowth",
        max_candidates=5,
        on_candidate_explosion="truncate",
    )
    with pytest.warns(DegradedModeWarning):
        est.fit(X)

    assert est.status_ == "DEGRADED"
    assert est.is_degraded_ is True
    assert est.is_complete_ is False
    assert est.metadata_["future_candidates_status"] == "UNKNOWN"
    assert "Candidate explosion detected" in est.termination_reason_
    assert len(est.frequent_itemsets_) == 5


def test_eclat_truncate_degraded_metadata():
    """Verify Eclat truncation sets future_candidates_status='UNKNOWN' and uniform degradation_reason (R6)."""
    X = (np.random.RandomState(42).rand(50, 10) > 0.3).astype(int)
    est = AssociationRulesEstimator(
        min_support=0.1,
        algorithm="eclat",
        max_candidates=5,
        on_candidate_explosion="truncate",
    )
    with pytest.warns(DegradedModeWarning):
        est.fit(X)

    assert est.status_ == "DEGRADED"
    assert est.is_degraded_ is True
    assert est.is_complete_ is False
    assert est.metadata_["future_candidates_status"] == "UNKNOWN"
    assert "Candidate explosion detected" in est.termination_reason_
    assert len(est.frequent_itemsets_) == 5


def test_exact_budget_boundary_not_degraded():
    """Verify R6: Reaching exactly max_candidates does NOT trigger degradation; only exceeding it does."""
    X = np.array([[1, 0], [1, 0], [1, 1]], dtype=bool)
    for algo in ["apriori", "fpgrowth", "eclat"]:
        est_exact = AssociationRulesEstimator(algorithm=algo, min_support=0.3, max_candidates=100)
        est_exact.fit(X)
        n_exact = len(est_exact.frequent_itemsets_)

        for policy in ["truncate", "degrade"]:
            est_bound = AssociationRulesEstimator(
                algorithm=algo,
                min_support=0.3,
                max_candidates=n_exact,
                on_candidate_explosion=policy,
            )
            est_bound.fit(X)
            assert est_bound.status_ == "EXACT", f"{algo} with {policy} degraded prematurely at boundary"
            assert est_bound.is_degraded_ is False
            assert est_bound.is_complete_ is True
            assert est_bound.termination_reason_ is None
            assert len(est_bound.frequent_itemsets_) == n_exact


def test_lazy_candidate_budget_apriori():
    """Verify R1: Apriori candidate budget check halts inside candidate generation loop."""
    dense = np.ones((5, 12), dtype=bool)
    est = AssociationRulesEstimator(
        algorithm="apriori",
        min_support=0.5,
        max_candidates=10,
        on_candidate_explosion="degrade",
    )
    with pytest.warns(DegradedModeWarning):
        est.fit(dense)

    assert est.status_ == "DEGRADED"
    assert est.is_degraded_ is True
    # At level 2, candidates generated should not exceed max_candidates + 1
    assert est.metadata_["candidates_generated_at_level"] <= 11


def test_unified_telemetry_schema():
    """Verify R2: metadata_ exposes structured 'operations' and 'resource_usage' across all engines."""
    for algo in ["apriori", "fpgrowth", "eclat"]:
        est = AssociationRulesEstimator(min_support=0.4, algorithm=algo)
        est.fit(HAND_COMPUTED_TRANSACTIONS)
        meta = est.metadata_

        assert "operations" in meta
        assert "resource_usage" in meta
        assert "wall_time_seconds" in meta["resource_usage"]
        assert "peak_rss_mb" in meta["resource_usage"]
        assert meta["resource_usage"]["wall_time_seconds"] >= 0.0
        assert meta["resource_usage"]["peak_rss_mb"] > 0.0
        assert "itemsets_discovered" in meta["operations"]
        assert meta["operations"]["itemsets_discovered"] == len(est.frequent_itemsets_)


def test_runtime_budget_enforcement_all_engines():
    """Verify R4: Runtime budget (max_time_seconds) enforced across apriori, fpgrowth, and eclat."""
    rng = np.random.RandomState(42)
    X = rng.binomial(1, 0.7, size=(50, 15))

    for algo in ["apriori", "fpgrowth", "eclat"]:
        # Microsecond time budget triggers graceful degradation
        est_deg = AssociationRulesEstimator(
            algorithm=algo,
            min_support=0.05,
            max_time_seconds=1e-6,
            on_candidate_explosion="degrade",
        )
        with pytest.warns(DegradedModeWarning, match="Time budget exceeded"):
            est_deg.fit(X)

        assert est_deg.status_ == "DEGRADED"
        assert est_deg.is_degraded_ is True
        assert est_deg.is_complete_ is False
        assert "Time budget exceeded" in est_deg.termination_reason_

        # Raise policy raises CandidateExplosionError
        est_raise = AssociationRulesEstimator(
            algorithm=algo,
            min_support=0.05,
            max_time_seconds=1e-6,
            on_candidate_explosion="raise",
        )
        with pytest.raises(CandidateExplosionError, match="Time budget exceeded"):
            est_raise.fit(X)


def test_memory_budget_enforcement_fit():
    """Verify R3: Memory governor with baseline tracking, peak RSS monitoring, and budget enforcement."""
    rng = np.random.RandomState(42)
    X = rng.binomial(1, 0.7, size=(40, 10))

    # Case 1: Unexceeded memory budget (generous headroom above process baseline)
    # Verifies that when the memory envelope is respected, mining completes in EXACT mode.
    baseline = get_current_memory_mb()
    for algo in ["apriori", "fpgrowth", "eclat"]:
        est_ok = AssociationRulesEstimator(
            algorithm=algo,
            min_support=0.1,
            max_memory_mb=baseline + 1024.0,
        )
        est_ok.fit(X)
        assert est_ok.status_ == "EXACT"
        assert est_ok.is_degraded_ is False
        assert est_ok.peak_memory_mb_ >= est_ok.baseline_memory_mb_
        assert "resource_usage" in est_ok.metadata_
        assert "baseline_rss_mb" in est_ok.metadata_["resource_usage"]
        assert "incremental_rss_mb" in est_ok.metadata_["resource_usage"]

    # Case 2: Exceeded memory budget
    # Sets a threshold just above current process baseline (base + 0.0001 MB) so baseline is valid
    # at invocation time, but allocations during transaction processing trigger the governor.
    for algo in ["apriori", "fpgrowth", "eclat"]:
        curr_base = get_current_memory_mb()
        est_mem = AssociationRulesEstimator(
            algorithm=algo,
            min_support=0.1,
            max_memory_mb=curr_base + 0.0001,
            on_candidate_explosion="degrade",
        )
        with pytest.warns(DegradedModeWarning, match="Memory budget exceeded"):
            est_mem.fit(X)

        assert est_mem.status_ == "DEGRADED"
        assert est_mem.is_degraded_ is True
        assert "Memory budget exceeded" in est_mem.termination_reason_
        assert est_mem.peak_memory_mb_ >= est_mem.baseline_memory_mb_
        assert "candidates_observed_before_halt" in est_mem.metadata_


def test_held_out_validation_scoring():
    """Verify Section 6: scorers evaluate rules discovered during fit against held-out validation data X."""
    from skarm.metrics import mean_confidence_scorer, mean_lift_scorer, total_support_scorer

    # Training data: item 0 and item 1 always co-occur
    # Transactions: [ [0, 1], [0, 1], [0, 1], [2] ]
    X_train = np.array([
        [1, 1, 0],
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
    ], dtype=bool)

    est = AssociationRulesEstimator(min_support=0.5, metric="confidence", min_threshold=0.5, algorithm="apriori")
    est.fit(X_train)
    assert not est.rules_.empty

    # Validation data 1: perfect co-occurrence of 0 and 1
    X_val_perfect = np.array([
        [1, 1, 0],
        [1, 1, 0],
    ], dtype=bool)
    score_perfect = est.score(X_val_perfect)
    assert np.isclose(score_perfect, 1.0)

    # Validation data 2: item 0 occurs but item 1 NEVER occurs
    # {0} -> {1} confidence should drop to 0.0!
    X_val_zero = np.array([
        [1, 0, 0],
        [1, 0, 0],
    ], dtype=bool)
    score_zero = est.score(X_val_zero)
    assert np.isclose(score_zero, 0.0)

    # Validation data 3: antecedent {0} never occurs in validation data
    X_val_no_ante = np.array([
        [0, 0, 1],
        [0, 0, 1],
    ], dtype=bool)
    score_no_ante = est.score(X_val_no_ante)
    assert np.isclose(score_no_ante, 0.0)



