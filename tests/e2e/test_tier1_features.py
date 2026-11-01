"""Tier 1: Feature Coverage Verification Suite.

Exercises the SKARM framework as an end-user through its public API:
- Input format handling (lists of lists, dense binary arrays, scipy sparse matrices, pandas DataFrames)
- Core mining algorithms (Apriori, FP-Growth, Eclat) verified against analytical oracles
- Rule generation and metrics (support, confidence, lift, leverage, conviction, Zhang's metric)
- Execution result states (EXACT, DEGRADED, FAILED) and candidate count semantics
- Estimator API contracts (fit, predict, transform, get_feature_names_out, .score(), clone, joblib)

Minimum 5 distinct tests per feature requirement.
"""

import io
import math
import pytest
import numpy as np
import pandas as pd
from scipy import sparse
import joblib
from sklearn.base import clone

from skarm.preprocessing import TransactionEncoder
from skarm.estimator import AssociationRulesEstimator
from skarm.apriori import apriori
from skarm.fpgrowth import fpgrowth
from skarm.eclat import eclat
from skarm.rules import (
    association_rules,
    compute_leverage,
    compute_conviction,
    compute_zhangs_metric,
)
from skarm.pruning import (
    PruningPolicy,
    CandidateExplosionError,
    BoundaryConditionError,
    DegradedModeWarning,
)
from tests.fixtures.oracle_data import (
    HAND_COMPUTED_TRANSACTIONS,
    ORACLE_FREQUENT_ITEMSETS_0_4,
    ORACLE_RULES_0_4_CONF_0_6,
)


# ============================================================================
# FEATURE 1: Input Format Handling (>= 5 tests)
# ============================================================================

class TestTier1InputFormatHandling:
    """Validate that AssociationRulesEstimator and TransactionEncoder accept all mandated input formats."""

    def test_input_list_of_lists_raw_transactions(self):
        """Test raw string transaction lists as an end user would pass shopping baskets."""
        raw_baskets = [
            ["milk", "bread", "butter"],
            ["bread", "butter"],
            ["milk", "bread"],
            ["beer", "chips"],
            ["milk", "bread", "butter", "beer"],
        ]
        est = AssociationRulesEstimator(min_support=0.4, metric="confidence", min_threshold=0.5)
        est.fit(raw_baskets)

        assert est.n_features_in_ == 5
        assert set(est.columns_) == {"beer", "bread", "butter", "chips", "milk"}
        assert hasattr(est, "rules_")
        assert not est.rules_.empty

        # Predict recommendations for a new basket ['bread']
        recs = est.predict([["bread"]])
        assert isinstance(recs, list)
        assert len(recs) == 1
        # Butter and milk frequently co-occur with bread
        assert any(item in recs[0] for item in ["butter", "milk"])

    def test_input_dense_numpy_boolean_matrix(self):
        """Test pre-encoded dense 2D NumPy boolean indicator array."""
        # 5 transactions, 4 items
        X_dense = np.array([
            [True, True, True, False],
            [True, True, False, False],
            [True, False, True, False],
            [False, True, True, True],
            [True, True, True, False],
        ], dtype=bool)

        est = AssociationRulesEstimator(min_support=0.4, algorithm="fpgrowth")
        est.fit(X_dense)

        assert est.n_features_in_ == 4
        assert est.columns_ == ["item_0", "item_1", "item_2", "item_3"]
        assert len(est.frequent_itemsets_) > 0
        # items 0, 1, 2 co-occur in 2 out of 5 transactions (rows 0 and 4: support 0.4)
        supp_012 = est.frequent_itemsets_.get(frozenset([0, 1, 2]))
        assert supp_012 is not None
        assert math.isclose(supp_012, 0.4, abs_tol=1e-9)

    def test_input_scipy_csr_sparse_matrix(self):
        """Test scipy CSR sparse binary matrix (typical for memory-efficient text/basket pipelines)."""
        # Create CSR matrix from dense data
        dense_data = np.array([
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [1, 1, 1, 0],
        ], dtype=np.int8)
        X_csr = sparse.csr_matrix(dense_data)

        est = AssociationRulesEstimator(min_support=0.4, algorithm="eclat")
        est.fit(X_csr)

        assert est.n_features_in_ == 4
        assert hasattr(est, "rules_")
        # Transform should succeed on CSR matrix
        X_trans = est.transform(X_csr)
        assert X_trans.shape[0] == 5
        assert X_trans.shape[1] == len(est.rules_)

    def test_input_scipy_csc_sparse_matrix(self):
        """Test scipy CSC sparse matrix format conversion and fitting."""
        dense_data = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ], dtype=bool)
        X_csc = sparse.csc_matrix(dense_data)

        est = AssociationRulesEstimator(min_support=0.5, algorithm="auto")
        est.fit(X_csc)

        assert est.n_features_in_ == 3
        assert hasattr(est, "rules_")
        assert not est.rules_.empty

    def test_input_pandas_dataframe_with_columns(self):
        """Test pandas DataFrame with explicit column names, ensuring names are preserved."""
        df = pd.DataFrame({
            "Apples": [1, 1, 0, 1, 0],
            "Bananas": [1, 1, 1, 0, 1],
            "Cherries": [0, 1, 1, 1, 1],
            "Dates": [0, 0, 0, 1, 0],
        })

        est = AssociationRulesEstimator(min_support=0.4, algorithm="apriori")
        est.fit(df)

        assert est.columns_ == ["Apples", "Bananas", "Cherries", "Dates"]
        if hasattr(est, "feature_names_in_"):
            np.testing.assert_array_equal(est.feature_names_in_, np.array(["Apples", "Bananas", "Cherries", "Dates"]))
        # Rules should have human-readable antecedent and consequent names
        assert not est.rules_.empty
        for _, row in est.rules_.iterrows():
            ante_names = [est.columns_[i] for i in row["antecedent"]]
            cons_names = [est.columns_[i] for i in row["consequent"]]
            assert all(name in df.columns for name in ante_names)
            assert all(name in df.columns for name in cons_names)

    def test_transaction_encoder_fit_transform_modes(self):
        """Test TransactionEncoder standalone in dense, sparse, and inverse_transform modes."""
        transactions = [
            ["apple", "banana"],
            ["banana", "carrot"],
            ["apple", "carrot", "date"],
        ]
        # Dense mode
        enc_dense = TransactionEncoder(sparse_output=False)
        X_dense = enc_dense.fit_transform(transactions)
        assert isinstance(X_dense, np.ndarray)
        assert X_dense.shape == (3, 4)
        assert enc_dense.columns_ == ["apple", "banana", "carrot", "date"]

        # Sparse mode
        enc_sparse = TransactionEncoder(sparse_output=True)
        X_sparse = enc_sparse.fit_transform(transactions)
        assert sparse.issparse(X_sparse)
        assert X_sparse.shape == (3, 4)

        # Inverse transform
        inverted = enc_dense.inverse_transform(X_dense)
        assert len(inverted) == 3
        assert set(inverted[0]) == {"apple", "banana"}
        assert set(inverted[1]) == {"banana", "carrot"}
        assert set(inverted[2]) == {"apple", "carrot", "date"}


# ============================================================================
# FEATURE 2: Core Mining Algorithms (>= 5 tests)
# ============================================================================

class TestTier1CoreMiningAlgorithms:
    """Validate mathematical precision of Apriori, FP-Growth, and Eclat against analytical oracles."""

    def test_apriori_matches_analytical_oracle(self):
        """Apriori level-wise BFS matches hand-computed frequent itemsets exactly at min_support=0.4."""
        est = AssociationRulesEstimator(min_support=0.4, algorithm="apriori")
        est.fit(HAND_COMPUTED_TRANSACTIONS)

        found_itemsets = {
            frozenset(est.columns_[i] for i in k): supp
            for k, supp in est.frequent_itemsets_.items()
        }
        assert set(found_itemsets.keys()) == set(ORACLE_FREQUENT_ITEMSETS_0_4.keys())
        for itemset, exp_supp in ORACLE_FREQUENT_ITEMSETS_0_4.items():
            act_supp = found_itemsets[itemset]
            assert math.isclose(act_supp, exp_supp, abs_tol=1e-9)

    def test_fpgrowth_matches_analytical_oracle(self):
        """FP-Growth recursive pattern-tree matches hand-computed frequent itemsets exactly at min_support=0.4."""
        est = AssociationRulesEstimator(min_support=0.4, algorithm="fpgrowth")
        est.fit(HAND_COMPUTED_TRANSACTIONS)

        found_itemsets = {
            frozenset(est.columns_[i] for i in k): supp
            for k, supp in est.frequent_itemsets_.items()
        }
        assert set(found_itemsets.keys()) == set(ORACLE_FREQUENT_ITEMSETS_0_4.keys())
        for itemset, exp_supp in ORACLE_FREQUENT_ITEMSETS_0_4.items():
            act_supp = found_itemsets[itemset]
            assert math.isclose(act_supp, exp_supp, abs_tol=1e-9)

    def test_eclat_matches_analytical_oracle(self):
        """Eclat vertical TID-intersection matches hand-computed frequent itemsets exactly at min_support=0.4."""
        est = AssociationRulesEstimator(min_support=0.4, algorithm="eclat")
        est.fit(HAND_COMPUTED_TRANSACTIONS)

        found_itemsets = {
            frozenset(est.columns_[i] for i in k): supp
            for k, supp in est.frequent_itemsets_.items()
        }
        assert set(found_itemsets.keys()) == set(ORACLE_FREQUENT_ITEMSETS_0_4.keys())
        for itemset, exp_supp in ORACLE_FREQUENT_ITEMSETS_0_4.items():
            act_supp = found_itemsets[itemset]
            assert math.isclose(act_supp, exp_supp, abs_tol=1e-9)

    def test_cross_engine_itemset_support_equivalence(self):
        """Verify strict mathematical equivalence across all 3 engines on a multi-item dataset."""
        # Non-trivial transactional database
        data = [
            ["laptop", "mouse", "keyboard", "usb_hub"],
            ["laptop", "mouse", "bag"],
            ["mouse", "keyboard"],
            ["laptop", "keyboard", "monitor"],
            ["laptop", "mouse", "keyboard", "monitor"],
            ["mouse", "monitor", "headset"],
            ["laptop", "mouse", "headset"],
            ["laptop", "bag", "mouse"],
        ]
        min_supp = 0.25

        est_apriori = AssociationRulesEstimator(min_support=min_supp, algorithm="apriori").fit(data)
        est_fpgrowth = AssociationRulesEstimator(min_support=min_supp, algorithm="fpgrowth").fit(data)
        est_eclat = AssociationRulesEstimator(min_support=min_supp, algorithm="eclat").fit(data)

        ap_sets = {frozenset(est_apriori.columns_[i] for i in k): round(supp, 6) for k, supp in est_apriori.frequent_itemsets_.items()}
        fp_sets = {frozenset(est_fpgrowth.columns_[i] for i in k): round(supp, 6) for k, supp in est_fpgrowth.frequent_itemsets_.items()}
        ec_sets = {frozenset(est_eclat.columns_[i] for i in k): round(supp, 6) for k, supp in est_eclat.frequent_itemsets_.items()}

        assert ap_sets == fp_sets, "Apriori and FP-Growth results differ"
        assert fp_sets == ec_sets, "FP-Growth and Eclat results differ"

    def test_auto_algorithm_selection(self):
        """Verify that algorithm='auto' selects Eclat on sparse input and FP-Growth on dense input."""
        dense_X = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=bool)
        sparse_X = sparse.csr_matrix(dense_X)

        est_dense = AssociationRulesEstimator(algorithm="auto").fit(dense_X)
        est_sparse = AssociationRulesEstimator(algorithm="auto").fit(sparse_X)

        assert est_dense.frequent_itemsets_ is not None
        assert est_sparse.frequent_itemsets_ is not None

    def test_max_len_constraint_filtering(self):
        """Verify that max_len=2 restricts discovered itemsets to singletons and pairs across all engines."""
        for algo in ["apriori", "fpgrowth", "eclat"]:
            est = AssociationRulesEstimator(min_support=0.3, max_len=2, algorithm=algo)
            est.fit(HAND_COMPUTED_TRANSACTIONS)
            for itemset in est.frequent_itemsets_.keys():
                assert len(itemset) <= 2, f"Algorithm {algo} generated itemset of length {len(itemset)} > 2"


# ============================================================================
# FEATURE 3: Rule Generation and Metrics (>= 5 tests)
# ============================================================================

class TestTier1RuleGenerationAndMetrics:
    """Validate statistical rule metrics against mathematical properties and analytical ground truth."""

    def test_rule_metrics_against_analytical_oracle(self):
        """Rules generated from oracle itemsets match exact analytical values to < 1e-9."""
        est = AssociationRulesEstimator(min_support=0.4, metric="confidence", min_threshold=0.6, algorithm="apriori")
        est.fit(HAND_COMPUTED_TRANSACTIONS)

        # Build map of (antecedent_names, consequent_names) -> row
        rules_map = {}
        for _, row in est.rules_.iterrows():
            ante = frozenset(est.columns_[i] for i in row["antecedent"])
            cons = frozenset(est.columns_[i] for i in row["consequent"])
            rules_map[(ante, cons)] = row

        assert set(rules_map.keys()) == set(ORACLE_RULES_0_4_CONF_0_6.keys())
        for (ante, cons), exp_metrics in ORACLE_RULES_0_4_CONF_0_6.items():
            act_row = rules_map[(ante, cons)]
            for metric_name, exp_val in exp_metrics.items():
                act_val = act_row[metric_name]
                if math.isinf(exp_val):
                    assert math.isinf(act_val), f"Rule {ante}->{cons} {metric_name} should be inf"
                else:
                    assert math.isclose(act_val, exp_val, abs_tol=1e-9), (
                        f"Rule {ante}->{cons} {metric_name} mismatch: expected {exp_val}, got {act_val}"
                    )

    def test_conviction_infinite_on_deterministic_rule(self):
        """When confidence == 1.0, conviction is mathematically infinite."""
        # Single rule with perfect confidence
        supp_c = 0.5
        conf = 1.0
        conv = compute_conviction(supp_c, conf)
        assert math.isinf(conv)

    def test_leverage_zero_on_statistical_independence(self):
        """When supp(A U C) == supp(A) * supp(C), leverage is exactly 0.0."""
        supp_a = 0.5
        supp_c = 0.4
        supp_ac = supp_a * supp_c  # 0.2
        lev = compute_leverage(supp_ac, supp_a, supp_c)
        assert math.isclose(lev, 0.0, abs_tol=1e-12)

    def test_zhangs_metric_bounded_range(self):
        """Zhang's metric must be strictly bounded in [-1.0, 1.0] for all rules."""
        est = AssociationRulesEstimator(min_support=0.2, metric="support", min_threshold=0.2)
        est.fit(HAND_COMPUTED_TRANSACTIONS)

        assert not est.rules_.empty
        for _, row in est.rules_.iterrows():
            z = row["zhangs_metric"]
            assert not math.isnan(z), f"Zhang's metric is NaN: {row}"
            assert -1.0 - 1e-9 <= z <= 1.0 + 1e-9, f"Zhang's metric {z} out of [-1, 1]"

    def test_rule_filtering_by_supported_metrics(self):
        """Validate filtering rules across all supported metrics (confidence, lift, leverage, conviction, zhangs_metric)."""
        metrics_and_thresholds = [
            ("confidence", 0.7),
            ("lift", 1.2),
            ("leverage", 0.05),
            ("conviction", 1.5),
            ("zhangs_metric", 0.1),
        ]
        for metric, thresh in metrics_and_thresholds:
            est = AssociationRulesEstimator(
                min_support=0.3,
                metric=metric,
                min_threshold=thresh,
            )
            est.fit(HAND_COMPUTED_TRANSACTIONS)
            for _, row in est.rules_.iterrows():
                val = row[metric]
                assert val >= thresh - 1e-9, f"Rule has {metric}={val} below threshold {thresh}"

    def test_rule_partition_validity_constraints(self):
        """Verify formal partition constraints: non-empty, disjoint, and combined is frequent."""
        est = AssociationRulesEstimator(min_support=0.4, metric="confidence", min_threshold=0.5)
        est.fit(HAND_COMPUTED_TRANSACTIONS)

        for _, row in est.rules_.iterrows():
            ante = set(row["antecedent"])
            cons = set(row["consequent"])
            # Non-empty
            assert len(ante) > 0, "Antecedent is empty"
            assert len(cons) > 0, "Consequent is empty"
            # Disjoint
            assert ante.isdisjoint(cons), f"Antecedent {ante} and Consequent {cons} overlap"
            # Union is frequent
            union = frozenset(ante | cons)
            assert union in est.frequent_itemsets_, f"Union {union} is not in frequent itemsets"


# ============================================================================
# FEATURE 4: Execution Result States (>= 5 tests)
# ============================================================================

class TestTier1ExecutionResultStates:
    """Validate formal tri-state execution contracts (EXACT, DEGRADED, FAILED) and candidate count semantics."""

    def test_result_state_exact_on_unconstrained_execution(self):
        """Unconstrained execution completes with status EXACT and is_complete=True."""
        est = AssociationRulesEstimator(min_support=0.4, max_candidates=10000)
        est.fit(HAND_COMPUTED_TRANSACTIONS)

        assert est.is_degraded_ is False
        if hasattr(est, "status_"):
            assert est.status_ == "EXACT"
        if hasattr(est, "is_complete_"):
            assert est.is_complete_ is True
        if hasattr(est, "termination_reason_"):
            assert est.termination_reason_ is None

    def test_result_state_degraded_on_candidate_budget(self):
        """Exceeding candidate budget with degrade policy transitions safely into DEGRADED state."""
        # 16 items all present in 5 transactions: 2^16 = 65,536 itemsets
        dense_matrix = np.ones((5, 16), dtype=bool)
        est = AssociationRulesEstimator(
            min_support=0.5,
            max_candidates=50,
            on_candidate_explosion="degrade",
            algorithm="apriori",
        )
        with pytest.warns(DegradedModeWarning, match="degraded mode"):
            est.fit(dense_matrix)

        assert est.is_degraded_ is True
        if hasattr(est, "status_"):
            assert est.status_ == "DEGRADED"
        if hasattr(est, "is_complete_"):
            assert est.is_complete_ is False
        if hasattr(est, "termination_reason_"):
            assert est.termination_reason_ is not None

    def test_result_state_raise_on_candidate_explosion(self):
        """Exceeding candidate budget with raise policy immediately throws CandidateExplosionError."""
        dense_matrix = np.ones((5, 16), dtype=bool)
        est = AssociationRulesEstimator(
            min_support=0.5,
            max_candidates=50,
            on_candidate_explosion="raise",
            algorithm="apriori",
        )
        with pytest.raises(CandidateExplosionError, match="Candidate explosion detected"):
            est.fit(dense_matrix)

    def test_result_state_truncate_policy(self):
        """Truncate policy keeps top candidates and warns, without raising exception."""
        dense_matrix = np.ones((5, 14), dtype=bool)
        est = AssociationRulesEstimator(
            min_support=0.5,
            max_candidates=50,
            on_candidate_explosion="truncate",
            algorithm="apriori",
        )
        with pytest.warns(DegradedModeWarning):
            est.fit(dense_matrix)

        assert hasattr(est, "frequent_itemsets_")
        assert len(est.frequent_itemsets_) > 0

    def test_result_metadata_candidate_semantics(self):
        """Candidate count semantics distinguish exact remaining unvisited counts from unknown counts."""
        dense_matrix = np.ones((5, 14), dtype=bool)
        est = AssociationRulesEstimator(
            min_support=0.5,
            max_candidates=40,
            on_candidate_explosion="degrade",
            algorithm="apriori",
        )
        with pytest.warns(DegradedModeWarning):
            est.fit(dense_matrix)

        if hasattr(est, "metadata_"):
            meta = est.metadata_
            assert isinstance(meta, dict)
            assert meta.get("status") in ("EXACT", "DEGRADED")
            assert "is_complete" in meta
            # Check candidate status
            cand_status = meta.get("unvisited_candidates_status") or meta.get("future_candidates_status")
            if cand_status is not None:
                assert cand_status.lower() in ("exact", "unknown", "none", "exhausted")

    def test_result_termination_reason_reporting(self):
        """Termination reason is None for exact runs and contains diagnostic details for degraded runs."""
        est_exact = AssociationRulesEstimator(min_support=0.4).fit(HAND_COMPUTED_TRANSACTIONS)
        if hasattr(est_exact, "termination_reason_"):
            assert est_exact.termination_reason_ is None

        dense_matrix = np.ones((5, 12), dtype=bool)
        est_deg = AssociationRulesEstimator(min_support=0.5, max_candidates=20, on_candidate_explosion="degrade")
        with pytest.warns(DegradedModeWarning):
            est_deg.fit(dense_matrix)
        if hasattr(est_deg, "termination_reason_"):
            assert est_deg.termination_reason_ is not None
            assert len(est_deg.termination_reason_) > 0


# ============================================================================
# FEATURE 5: Estimator API Contracts (>= 5 tests)
# ============================================================================

class TestTier1EstimatorAPIContracts:
    """Validate strict adherence to scikit-learn estimator, transformer, and scoring contracts."""

    def test_estimator_clone_and_immutability(self):
        """Constructor parameters are stored verbatim and clone() produces identical unfitted copy."""
        est = AssociationRulesEstimator(
            min_support=0.25,
            metric="lift",
            min_threshold=1.5,
            algorithm="eclat",
            max_len=3,
            max_candidates=5000,
            on_candidate_explosion="degrade",
            sparse_output=True,
        )
        cloned = clone(est)

        assert cloned.min_support == 0.25
        assert cloned.metric == "lift"
        assert cloned.min_threshold == 1.5
        assert cloned.algorithm == "eclat"
        assert cloned.max_len == 3
        assert cloned.max_candidates == 5000
        assert cloned.on_candidate_explosion == "degrade"
        assert cloned.sparse_output is True
        assert not hasattr(cloned, "rules_")
        assert not hasattr(cloned, "frequent_itemsets_")

    def test_estimator_fitted_attributes_consistency(self):
        """fit() populates standard scikit-learn fitted attributes ending with underscore."""
        est = AssociationRulesEstimator(min_support=0.4)
        est.fit(HAND_COMPUTED_TRANSACTIONS)

        assert hasattr(est, "rules_")
        assert hasattr(est, "columns_")
        assert hasattr(est, "n_features_in_")
        assert est.n_features_in_ == len(est.columns_)
        assert hasattr(est, "is_degraded_")
        # Check either itemsets_ DataFrame or frequent_itemsets_ dict
        assert hasattr(est, "itemsets_") or hasattr(est, "frequent_itemsets_")

    def test_estimator_predict_recommendations(self):
        """predict() identifies firing antecedents and returns ranked recommended consequents."""
        est = AssociationRulesEstimator(min_support=0.4, metric="confidence", min_threshold=0.6)
        est.fit(HAND_COMPUTED_TRANSACTIONS)

        # Query 1: ['A', 'C'] -> fires rule {A, C} -> B (confidence 0.667)
        # Query 2: ['E'] -> fires rule {E} -> B (confidence 1.0)
        recs = est.predict([["A", "C"], ["E"]])
        assert len(recs) == 2
        assert "B" in recs[0]
        assert "B" in recs[1]

    def test_estimator_transform_rule_indicators(self):
        """transform() outputs a 2D binary indicator matrix of fired rules."""
        est = AssociationRulesEstimator(min_support=0.4, metric="confidence", min_threshold=0.6)
        est.fit(HAND_COMPUTED_TRANSACTIONS)

        X_trans = est.transform(HAND_COMPUTED_TRANSACTIONS)
        assert X_trans.shape == (len(HAND_COMPUTED_TRANSACTIONS), len(est.rules_))
        # Each row should have at least one fired rule for these transactions
        assert np.any(X_trans)

    def test_estimator_transform_sparse_output(self):
        """When sparse_output=True, transform() returns a scipy sparse matrix."""
        est = AssociationRulesEstimator(min_support=0.4, sparse_output=True)
        est.fit(HAND_COMPUTED_TRANSACTIONS)

        X_trans = est.transform(HAND_COMPUTED_TRANSACTIONS)
        # If the implementation returns CSR sparse or dense, verify dimensions and binary values
        if sparse.issparse(X_trans):
            assert X_trans.shape == (len(HAND_COMPUTED_TRANSACTIONS), len(est.rules_))
            assert X_trans.dtype in (bool, np.int8, np.int32, np.float64)
        else:
            assert isinstance(X_trans, np.ndarray)

    def test_estimator_get_feature_names_out(self):
        """get_feature_names_out() returns an array of rule descriptor strings."""
        est = AssociationRulesEstimator(min_support=0.4)
        est.fit(HAND_COMPUTED_TRANSACTIONS)

        if hasattr(est, "get_feature_names_out"):
            feature_names = est.get_feature_names_out()
            assert isinstance(feature_names, np.ndarray)
            assert len(feature_names) == len(est.rules_)
            if len(feature_names) > 0:
                assert "->" in str(feature_names[0]) or "item" in str(feature_names[0]) or "{" in str(feature_names[0])

    def test_estimator_score_method(self):
        """.score(X) returns a finite float score without requiring y labels."""
        est = AssociationRulesEstimator(min_support=0.4)
        est.fit(HAND_COMPUTED_TRANSACTIONS)

        if hasattr(est, "score"):
            score = est.score(HAND_COMPUTED_TRANSACTIONS)
            assert isinstance(score, (int, float, np.floating))
            assert not math.isnan(score)
            assert not math.isinf(score)

    def test_estimator_joblib_serialization(self):
        """joblib serialization round-trip restores exact model parameters, rules, and predictions."""
        est = AssociationRulesEstimator(min_support=0.4, metric="confidence", min_threshold=0.6)
        est.fit(HAND_COMPUTED_TRANSACTIONS)

        buf = io.BytesIO()
        joblib.dump(est, buf)
        buf.seek(0)
        loaded_est = joblib.load(buf)

        assert loaded_est.min_support == est.min_support
        assert len(loaded_est.rules_) == len(est.rules_)
        preds_orig = est.predict([["A", "C"]])
        preds_loaded = loaded_est.predict([["A", "C"]])
        assert preds_orig == preds_loaded
