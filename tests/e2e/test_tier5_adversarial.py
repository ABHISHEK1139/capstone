"""Tier 5: Adversarial Coverage Hardening Verification Suite.

KL University Capstone Project Framework: KLCAP-2026-00340
White-box adversarial stress testing against the SKARM engine and test suite:
- Extreme boundary matrices (all zeros dense/sparse, 1x1, 1xN, Nx1, 100% density combinatorial stress)
- Multi-dimensional budget transitions (max_candidates, max_time_seconds, max_memory_mb) under rapid degradation & recovery
- Pickle and joblib model persistence and deserialization with fitted states (status_, is_complete_, itemsets_, rules_)
- Statistical metric numerical stability under extreme support regimes (10^-5 up to 1.0) and mathematical bounds
- Scikit-learn Pipeline and ColumnTransformer integration with set_output(transform="pandas")
- Adversarial sample weight distributions and malformed/unseen query fuzzing
"""

import io
import math
import pickle
import warnings
import pytest
import numpy as np
import pandas as pd
from scipy import sparse
import joblib

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

from skarm.estimator import AssociationRulesEstimator
from skarm.preprocessing import TransactionEncoder
from skarm.pruning import (
    PruningPolicy,
    CandidateExplosionError,
    BoundaryConditionError,
    DegradedModeWarning,
)
from skarm.rules import (
    association_rules,
    compute_leverage,
    compute_conviction,
    compute_zhangs_metric,
)


# ============================================================================
# CLASS 1: Extreme Boundary Matrices
# ============================================================================

class TestTier5BoundaryMatrices:
    """Adversarial testing on degenerate matrix geometries and extreme densities."""

    def test_adv_all_zeros_matrix_dense_and_sparse(self):
        """Verify all 3 engines handle 100% all-zero dense and sparse matrices gracefully."""
        n_samples, n_features = 20, 10
        X_dense_zeros = np.zeros((n_samples, n_features), dtype=bool)
        X_sparse_zeros = sparse.csr_matrix((n_samples, n_features), dtype=bool)

        for algo in ["apriori", "fpgrowth", "eclat"]:
            # Dense all-zeros
            est_dense = AssociationRulesEstimator(algorithm=algo, min_support=0.1)
            est_dense.fit(X_dense_zeros)
            assert len(est_dense.frequent_itemsets_) == 0, f"{algo} found itemsets on all-zeros dense matrix"
            assert est_dense.rules_.empty
            assert est_dense.status_ == "EXACT"
            assert est_dense.is_complete_ is True
            assert est_dense.is_degraded_ is False
            assert est_dense.termination_reason_ is None

            # Prediction, transform, score, and feature names on all-zeros
            preds = est_dense.predict(X_dense_zeros)
            assert len(preds) == n_samples
            assert all(len(p) == 0 for p in preds)

            trans = est_dense.transform(X_dense_zeros)
            assert trans.shape == (n_samples, 0)
            assert trans.dtype == bool

            score = est_dense.score(X_dense_zeros)
            assert score == 0.0
            assert len(est_dense.get_feature_names_out()) == 0

            # Sparse all-zeros
            est_sparse = AssociationRulesEstimator(algorithm=algo, min_support=0.1)
            est_sparse.fit(X_sparse_zeros)
            assert len(est_sparse.frequent_itemsets_) == 0, f"{algo} found itemsets on all-zeros sparse matrix"
            assert est_sparse.rules_.empty
            assert est_sparse.status_ == "EXACT"

    def test_adv_single_cell_and_single_row_matrices(self):
        """Verify behavior on extreme 1x1, Nx1, and 1xN single transaction matrices."""
        # 1x1 matrix (single cell)
        X_1x1 = np.array([[True]], dtype=bool)
        for algo in ["apriori", "fpgrowth", "eclat"]:
            est_1x1 = AssociationRulesEstimator(algorithm=algo, min_support=0.5).fit(X_1x1)
            assert len(est_1x1.frequent_itemsets_) == 1
            assert est_1x1.rules_.empty
            assert est_1x1.predict(X_1x1) == [[]]
            assert est_1x1.transform(X_1x1).shape == (1, 0)

        # 5x1 single-column matrix (5 transactions of 1 item)
        X_5x1 = np.array([[True], [False], [True], [True], [False]], dtype=bool)
        est_col = AssociationRulesEstimator(min_support=0.4).fit(X_5x1)
        assert len(est_col.frequent_itemsets_) == 1  # only item_0 is frequent (3/5 = 0.6)
        assert est_col.rules_.empty  # rules require itemsets of length >= 2
        assert est_col.predict(X_5x1) == [[] for _ in range(5)]

        # 1x8 single-transaction with 8 items (all combinations have support 1.0)
        X_1x8 = np.ones((1, 8), dtype=bool)
        est_row = AssociationRulesEstimator(min_support=0.5, max_len=3, min_threshold=0.5).fit(X_1x8)
        # 8 singletons + 28 pairs + 56 triples = 92 itemsets
        assert len(est_row.frequent_itemsets_) == 8 + 28 + 56
        assert not est_row.rules_.empty
        # All rules must have exact analytical values: support=1.0, confidence=1.0, lift=1.0, conviction=inf
        assert (est_row.rules_["support"] == 1.0).all()
        assert (est_row.rules_["confidence"] == 1.0).all()
        assert (est_row.rules_["lift"] == 1.0).all()
        assert np.isinf(est_row.rules_["conviction"]).all()

    def test_adv_100_percent_density_combinatorial_stress(self):
        """Stress-test 100% density matrix across all 3 engines verifying exact equivalence."""
        n_samples, n_items = 25, 8
        X_ones = np.ones((n_samples, n_items), dtype=bool)

        results = {}
        for algo in ["apriori", "fpgrowth", "eclat"]:
            est = AssociationRulesEstimator(
                algorithm=algo,
                min_support=0.5,
                max_len=4,
                min_threshold=0.5,
            ).fit(X_ones)
            results[algo] = est

            # Combinatorial oracle: C(8,1) + C(8,2) + C(8,3) + C(8,4) = 8 + 28 + 56 + 70 = 162
            assert len(est.frequent_itemsets_) == 162, f"{algo} itemsets count mismatch"
            assert est.status_ == "EXACT"
            assert est.is_complete_ is True

        # Strict cross-engine itemset and support equivalence
        apriori_items = results["apriori"].frequent_itemsets_
        for algo in ["fpgrowth", "eclat"]:
            algo_items = results[algo].frequent_itemsets_
            assert set(apriori_items.keys()) == set(algo_items.keys())
            for k, supp in apriori_items.items():
                assert math.isclose(supp, algo_items[k], abs_tol=1e-12)

            # Rule counts and metric equivalence
            assert len(results["apriori"].rules_) == len(results[algo].rules_)


# ============================================================================
# CLASS 2: Multi-Dimensional Budget Transitions & Rapid Degradation
# ============================================================================

class TestTier5BudgetTransitions:
    """Stress testing multi-dimensional budgets (candidates, time, memory) and state isolation."""

    def test_adv_candidate_budget_rapid_degradation_and_recovery(self):
        """Test rapid degradation under dense burst and clean recovery on re-fit without state leakage."""
        rng = np.random.RandomState(42)
        X_burst = rng.binomial(1, 0.9, size=(30, 14))
        X_normal = pd.DataFrame({
            "milk": [1, 1, 0, 0, 1],
            "bread": [1, 1, 1, 0, 0],
            "butter": [0, 1, 1, 1, 0],
        })

        # Estimator with strict candidate budget
        est = AssociationRulesEstimator(
            min_support=0.5,
            max_candidates=5,
            on_candidate_explosion="degrade",
        )

        # 1. Fit on burst -> must enter degraded mode
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            est.fit(X_burst)
            assert est.status_ == "DEGRADED"
            assert est.is_degraded_ is True
            assert est.is_complete_ is False
            assert est.termination_reason_ is not None
            assert "Candidate explosion detected" in est.termination_reason_
            assert est.metadata_["future_candidates_status"] == "UNKNOWN"
            assert any(issubclass(item.category, DegradedModeWarning) for item in w)

        # 2. Re-fit SAME instance on normal data -> must cleanly recover to EXACT
        est.fit(X_normal)
        assert est.status_ == "EXACT"
        assert est.is_degraded_ is False
        assert est.is_complete_ is True
        assert est.termination_reason_ is None
        assert est.metadata_["future_candidates_status"] == "EXHAUSTED"
        assert est.metadata_["is_candidate_count_exact"] is True

    def test_adv_time_budget_enforcement_and_explosion(self):
        """Verify max_time_seconds triggers graceful degradation or raises exception per policy."""
        rng = np.random.RandomState(123)
        X = rng.binomial(1, 0.65, size=(40, 12))

        # Degrade policy with microsecond time budget
        est_degrade = AssociationRulesEstimator(
            algorithm="apriori",
            min_support=0.05,
            max_time_seconds=1e-6,
            on_candidate_explosion="degrade",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            est_degrade.fit(X)
            assert est_degrade.status_ == "DEGRADED"
            assert est_degrade.is_degraded_ is True
            assert est_degrade.is_complete_ is False
            assert "Time budget exceeded" in est_degrade.termination_reason_
            assert any(issubclass(item.category, DegradedModeWarning) for item in w)

        # Raise policy with microsecond time budget
        est_raise = AssociationRulesEstimator(
            algorithm="apriori",
            min_support=0.05,
            max_time_seconds=1e-6,
            on_candidate_explosion="raise",
        )
        with pytest.raises(CandidateExplosionError, match="Time budget exceeded"):
            est_raise.fit(X)

    def test_adv_memory_budget_governor(self):
        """Verify memory budget checking on PruningPolicy and parameter validation."""
        policy_degrade = PruningPolicy(
            max_memory_mb=100.0,
            on_candidate_explosion="degrade",
        )
        # Below budget
        assert policy_degrade.check_budget(candidate_count=10, level=2, peak_memory_mb=50.0) is True
        assert not policy_degrade.is_degraded

        # Exceed memory budget
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert policy_degrade.check_budget(candidate_count=10, level=2, peak_memory_mb=150.0) is False
            assert policy_degrade.is_degraded is True
            assert "Memory budget exceeded" in policy_degrade.degradation_reason
            assert any(issubclass(item.category, DegradedModeWarning) for item in w)

        # Raise policy on memory budget
        policy_raise = PruningPolicy(
            max_memory_mb=50.0,
            on_candidate_explosion="raise",
        )
        with pytest.raises(CandidateExplosionError, match="Memory budget exceeded"):
            policy_raise.check_budget(peak_memory_mb=75.0)

        # Constructor boundary condition validation
        with pytest.raises(BoundaryConditionError, match="max_time_seconds must be positive"):
            AssociationRulesEstimator(max_time_seconds=-1.0).fit(np.ones((2, 2)))

        with pytest.raises(BoundaryConditionError, match="max_memory_mb must be positive"):
            AssociationRulesEstimator(max_memory_mb=0.0).fit(np.ones((2, 2)))


# ============================================================================
# CLASS 3: Pickle & Joblib Model Persistence
# ============================================================================

class TestTier5ModelPersistence:
    """Rigorous serialization testing for EXACT and DEGRADED states."""

    def test_adv_pickle_roundtrip_exact_and_degraded(self):
        """Verify pickle.dumps and pickle.loads retain all fitted attributes and inference fidelity."""
        df = pd.DataFrame({
            "milk": [1, 1, 0, 1, 1, 0],
            "bread": [1, 1, 1, 0, 1, 1],
            "butter": [1, 0, 1, 1, 1, 0],
            "jam": [0, 1, 1, 0, 0, 1],
        })

        # 1. Exact fitted model
        est_exact = AssociationRulesEstimator(min_support=0.3, min_threshold=0.5).fit(df)
        assert est_exact.status_ == "EXACT"

        pickled_bytes = pickle.dumps(est_exact)
        restored_exact = pickle.loads(pickled_bytes)

        # State attribute checks
        assert restored_exact.status_ == est_exact.status_
        assert restored_exact.is_complete_ == est_exact.is_complete_
        assert restored_exact.is_degraded_ == est_exact.is_degraded_
        assert restored_exact.termination_reason_ == est_exact.termination_reason_
        assert restored_exact.columns_ == est_exact.columns_
        assert restored_exact.metadata_ == est_exact.metadata_

        # DataFrames identity
        pd.testing.assert_frame_equal(restored_exact.itemsets_, est_exact.itemsets_)
        pd.testing.assert_frame_equal(restored_exact.rules_, est_exact.rules_)

        # Functional equivalence
        assert restored_exact.predict(df) == est_exact.predict(df)
        np.testing.assert_array_equal(restored_exact.transform(df), est_exact.transform(df))
        assert math.isclose(restored_exact.score(df), est_exact.score(df), abs_tol=1e-12)
        np.testing.assert_array_equal(
            restored_exact.get_feature_names_out(), est_exact.get_feature_names_out()
        )

        # 2. Degraded fitted model
        est_degraded = AssociationRulesEstimator(
            min_support=0.1,
            max_candidates=2,
            on_candidate_explosion="degrade",
        ).fit(df)
        assert est_degraded.status_ == "DEGRADED"

        pickled_deg_bytes = pickle.dumps(est_degraded)
        restored_degraded = pickle.loads(pickled_deg_bytes)

        assert restored_degraded.status_ == "DEGRADED"
        assert restored_degraded.is_complete_ is False
        assert restored_degraded.is_degraded_ is True
        assert restored_degraded.termination_reason_ == est_degraded.termination_reason_
        assert restored_degraded.metadata_["future_candidates_status"] == "UNKNOWN"
        pd.testing.assert_frame_equal(restored_degraded.itemsets_, est_degraded.itemsets_)

    def test_adv_joblib_roundtrip_exact_and_degraded(self):
        """Verify joblib streaming round-trip retains exact and degraded fitted states."""
        df = pd.DataFrame({
            "A": [1, 1, 0, 1],
            "B": [1, 1, 1, 0],
            "C": [1, 0, 1, 1],
        })

        est_exact = AssociationRulesEstimator(min_support=0.25).fit(df)
        buf = io.BytesIO()
        joblib.dump(est_exact, buf)
        buf.seek(0)
        restored = joblib.load(buf)

        assert restored.status_ == "EXACT"
        assert restored.is_complete_ is True
        pd.testing.assert_frame_equal(restored.rules_, est_exact.rules_)
        assert restored.predict(df) == est_exact.predict(df)

        # Degraded model joblib
        est_deg = AssociationRulesEstimator(
            min_support=0.1, max_candidates=2, on_candidate_explosion="degrade"
        ).fit(df)
        buf_deg = io.BytesIO()
        joblib.dump(est_deg, buf_deg)
        buf_deg.seek(0)
        restored_deg = joblib.load(buf_deg)

        assert restored_deg.status_ == "DEGRADED"
        assert restored_deg.is_complete_ is False
        assert restored_deg.termination_reason_ == est_deg.termination_reason_


# ============================================================================
# CLASS 4: Metric Numerical Stability & Extreme Support Regimes
# ============================================================================

class TestTier5MetricNumericalStability:
    """Stress testing numerical metrics under extreme low-support regimes (10^-5) and mathematical limits."""

    def test_adv_ultra_low_support_regime(self):
        """Verify mining at 10^-5 support on 100,000 transactions without underflow or precision loss."""
        n_samples = 100000
        n_features = 8

        # Target items 0 and 1 appear together in exactly 2 transactions (supp = 2e-5)
        # and item 2 appears in 3 transactions (supp = 3e-5)
        row_indices = [10, 10, 500, 500, 9999]
        col_indices = [0, 1, 0, 1, 2]
        data = np.ones(len(row_indices), dtype=bool)
        X_sparse = sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_samples, n_features),
            dtype=bool,
        )

        for algo in ["apriori", "fpgrowth", "eclat"]:
            est = AssociationRulesEstimator(
                algorithm=algo,
                min_support=1e-5,
                min_threshold=0.5,
            ).fit(X_sparse)

            assert len(est.frequent_itemsets_) == 4, f"{algo} failed to discover itemsets at 1e-5 support"
            assert est.status_ == "EXACT"
            assert len(est.rules_) == 2  # {0} -> {1} and {1} -> {0}

            # Check numerical accuracy of metrics at extreme support
            for _, rule in est.rules_.iterrows():
                assert math.isclose(rule["support"], 2e-5, abs_tol=1e-9)
                assert math.isclose(rule["confidence"], 1.0, abs_tol=1e-9)
                # lift = 1.0 / 2e-5 = 50,000.0
                assert math.isclose(rule["lift"], 50000.0, rel_tol=1e-6)
                assert np.isinf(rule["conviction"])
                assert math.isclose(rule["zhangs_metric"], 1.0, abs_tol=1e-6)
                assert -0.25 <= rule["leverage"] <= 0.25

    def test_adv_metric_mathematical_bounds_stress(self):
        """Stress-test metric calculations across edge probabilities verifying analytical invariants."""
        # Case 1: Statistical independence: supp(AC) = supp(A) * supp(C)
        supp_a, supp_c = 0.4, 0.5
        supp_ac = 0.2
        assert math.isclose(compute_leverage(supp_ac, supp_a, supp_c), 0.0, abs_tol=1e-12)
        assert math.isclose(compute_zhangs_metric(supp_ac, supp_a, supp_c), 0.0, abs_tol=1e-12)

        # Case 2: Perfect positive correlation: conf = 1.0
        assert math.isinf(compute_conviction(0.3, 1.0))
        assert math.isclose(compute_zhangs_metric(0.3, 0.3, 0.3), 1.0, abs_tol=1e-6)

        # Case 3: Complete disjointness: supp(AC) = 0.0 (perfect negative association)
        supp_a_dis, supp_c_dis = 0.4, 0.4
        supp_ac_dis = 0.0
        lev_dis = compute_leverage(supp_ac_dis, supp_a_dis, supp_c_dis)
        zhang_dis = compute_zhangs_metric(supp_ac_dis, supp_a_dis, supp_c_dis)
        assert math.isclose(lev_dis, -0.16, abs_tol=1e-12)
        assert math.isclose(zhang_dis, -1.0, abs_tol=1e-6)

        # Case 4: Universal items (supp = 1.0)
        assert math.isclose(compute_leverage(1.0, 1.0, 1.0), 0.0, abs_tol=1e-12)
        assert math.isclose(compute_zhangs_metric(1.0, 1.0, 1.0), 0.0, abs_tol=1e-12)
        assert math.isinf(compute_conviction(1.0, 1.0))

        # Case 5: Valid probabilities with very small probabilities (1e-6)
        supp_small = 1e-6
        zhang_small = compute_zhangs_metric(supp_small, supp_small, supp_small)
        assert math.isclose(zhang_small, 1.0, abs_tol=1e-6)
        assert not math.isnan(zhang_small)


# ============================================================================
# CLASS 5: Scikit-Learn Pipeline & ColumnTransformer (set_output pandas)
# ============================================================================

class TestTier5PipelineIntegration:
    """Verification of scikit-learn Pipeline and ColumnTransformer with set_output(transform='pandas')."""

    def test_adv_pipeline_set_output_pandas_workflow(self):
        """Verify AssociationRulesEstimator produces pandas DataFrame inside Pipeline with set_output."""
        df = pd.DataFrame({
            "milk": [1, 1, 0, 1, 1, 0, 1, 0],
            "bread": [1, 1, 1, 0, 1, 1, 0, 0],
            "butter": [1, 0, 1, 1, 1, 0, 1, 1],
        })

        pipe = Pipeline([
            ("arm", AssociationRulesEstimator(min_support=0.25, min_threshold=0.5))
        ])
        pipe.set_output(transform="pandas")

        out_df = pipe.fit_transform(df)
        assert isinstance(out_df, pd.DataFrame)
        assert out_df.shape[0] == 8
        assert out_df.shape[1] == len(pipe.named_steps["arm"].rules_)
        assert all("->" in col for col in out_df.columns)
        assert out_df.dtypes.iloc[0] == bool

        # Downstream classifier integration with set_output
        y = np.array([1, 1, 0, 1, 1, 0, 1, 0])
        clf_pipe = Pipeline([
            ("arm", AssociationRulesEstimator(min_support=0.25, min_threshold=0.5)),
            ("clf", LogisticRegression()),
        ])
        clf_pipe.fit(df, y)
        preds = clf_pipe.predict(df)
        assert preds.shape == (8,)

    def test_adv_column_transformer_set_output_pandas_integration(self):
        """Verify ColumnTransformer with set_output('pandas') prefixes rule features and retains remainder."""
        df = pd.DataFrame({
            "item_A": [1, 1, 0, 1],
            "item_B": [1, 1, 1, 0],
            "item_C": [1, 0, 1, 1],
            "num_metric": [10.5, 20.2, 15.1, 8.4],
        })

        ct = ColumnTransformer(
            transformers=[
                (
                    "arm",
                    AssociationRulesEstimator(min_support=0.25, min_threshold=0.5),
                    ["item_A", "item_B", "item_C"],
                )
            ],
            remainder="passthrough",
        )
        ct.set_output(transform="pandas")

        ct_out = ct.fit_transform(df)
        assert isinstance(ct_out, pd.DataFrame)
        assert ct_out.shape[0] == 4
        # Columns must contain arm__ prefixed rule features and remainder__num_metric
        col_names = ct_out.columns.tolist()
        assert any(c.startswith("arm__") for c in col_names)
        assert "remainder__num_metric" in col_names

    def test_adv_zero_rules_pipeline_pandas_handling(self):
        """Verify Pipeline and ColumnTransformer handle zero-rule scenarios gracefully in pandas mode."""
        df = pd.DataFrame({
            "item_1": [1, 0, 0],
            "item_2": [0, 1, 0],
            "num": [10, 20, 30],
        })

        # Pipeline with 0 rules
        pipe = Pipeline([
            ("arm", AssociationRulesEstimator(min_support=0.9))  # no 2-itemset meets 0.9
        ])
        pipe.set_output(transform="pandas")
        out_p = pipe.fit_transform(df[["item_1", "item_2"]])
        assert isinstance(out_p, pd.DataFrame)
        assert out_p.shape == (3, 0)

        # ColumnTransformer with 0 rules
        ct = ColumnTransformer(
            transformers=[
                ("arm", AssociationRulesEstimator(min_support=0.9), ["item_1", "item_2"])
            ],
            remainder="passthrough",
        )
        ct.set_output(transform="pandas")
        out_ct = ct.fit_transform(df)
        assert isinstance(out_ct, pd.DataFrame)
        assert out_ct.shape == (3, 1)
        assert out_ct.columns.tolist() == ["remainder__num"]


# ============================================================================
# CLASS 6: Adversarial Distributions & Query Fuzzing
# ============================================================================

class TestTier5AdversarialOraclesAndFuzzing:
    """Adversarial sample weight distributions, cloning, and malformed query handling."""

    def test_adv_sample_weight_adversarial_distributions(self):
        """Verify outlier transaction weight dominance and sample_weight error validation."""
        # 100 rows containing only item 2, 1 row containing items 0 and 1 with weight 1000
        X = np.zeros((101, 3), dtype=bool)
        X[:100, 2] = True
        X[100, 0] = True
        X[100, 1] = True

        weights = np.ones(101, dtype=float)
        weights[100] = 1000.0  # heavy weight

        # Total weight = 100 * 1 + 1000 = 1100
        # support({0, 1}) = 1000 / 1100 = 0.909 > 0.8
        # support({2}) = 100 / 1100 = 0.091 < 0.8
        est = AssociationRulesEstimator(min_support=0.8, algorithm="apriori").fit(
            X, sample_weight=weights
        )
        frequent = [tuple(sorted(list(it))) for it in est.frequent_itemsets_.keys()]
        assert (0,) in frequent
        assert (1,) in frequent
        assert (0, 1) in frequent
        assert (2,) not in frequent

        # Error checks on sample_weight
        with pytest.raises(ValueError, match="Sum of sample_weights must be strictly positive"):
            AssociationRulesEstimator().fit(X, sample_weight=np.zeros(101))

        with pytest.raises(ValueError, match="sample_weight cannot contain negative values"):
            bad_weights = np.ones(101)
            bad_weights[0] = -1.0
            AssociationRulesEstimator().fit(X, sample_weight=bad_weights)

        with pytest.raises(ValueError, match="sample_weight length"):
            AssociationRulesEstimator().fit(X, sample_weight=np.ones(50))

    def test_adv_unseen_items_and_malformed_queries(self):
        """Verify predict() and transform() on unseen items, mixed tokens, and empty query transactions."""
        df = pd.DataFrame({
            "apple": [1, 1, 0, 1],
            "banana": [1, 1, 1, 0],
            "cherry": [1, 0, 1, 1],
        })
        est = AssociationRulesEstimator(min_support=0.25, min_threshold=0.5).fit(df)

        adversarial_queries = [
            [],                             # completely empty query
            ["alien_item_xyz"],             # novel item never seen during training
            ["apple", "alien_item_xyz"],    # mixed known and novel
            ["banana", "cherry"],           # known combination
        ]

        preds = est.predict(adversarial_queries)
        assert len(preds) == 4
        assert preds[0] == []
        assert preds[1] == []
        # Query 2 had 'apple' -> should recommend consequents of apple rules
        assert len(preds[2]) > 0
        assert "banana" in preds[2] or "cherry" in preds[2]

        # Transform on adversarial queries
        trans = est.transform(adversarial_queries)
        assert trans.shape == (4, len(est.rules_))
        assert not np.any(trans[0])
        assert not np.any(trans[1])
        assert np.any(trans[2])
        assert np.any(trans[3])

    def test_adv_clone_parameter_immutability(self):
        """Verify sklearn.base.clone produces completely independent unfitted estimator."""
        est = AssociationRulesEstimator(
            min_support=0.3,
            metric="lift",
            min_threshold=1.5,
            algorithm="eclat",
            max_len=3,
            max_candidates=5000,
            on_candidate_explosion="truncate",
            scoring="mean_lift",
        )
        # Clone before fit
        cloned_unfitted = clone(est)
        assert cloned_unfitted.min_support == 0.3
        assert cloned_unfitted.metric == "lift"
        assert cloned_unfitted.algorithm == "eclat"
        assert not hasattr(cloned_unfitted, "rules_")

        # Fit original
        df = pd.DataFrame({"A": [1, 1, 0], "B": [1, 1, 1]})
        est.fit(df)
        assert hasattr(est, "rules_")

        # Clone after fit must NOT have fitted attributes
        cloned_after_fit = clone(est)
        assert not hasattr(cloned_after_fit, "rules_")
        assert not hasattr(cloned_after_fit, "status_")
        assert not hasattr(cloned_after_fit, "itemsets_")
        assert cloned_after_fit.scoring == "mean_lift"
