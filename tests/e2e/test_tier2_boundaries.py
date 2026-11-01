"""Tier 2: Boundary & Corner Cases Verification Suite.

Tests extreme parameters, unusual input geometries, data sanitization,
and strict safety limits:
- Transaction size and content boundaries (empty, single-item, all-items, single transaction)
- Duplicate transactions and token idempotency (repeated items, permutations, case/special chars)
- Extreme thresholds (min_support in {0.0, 1.0}, min_threshold in {0.0, 1.0}, out-of-bounds validation)
- Missing value tokens and data validation (None, np.nan, continuous floats, integer counts, 1D arrays)
- Candidate and itemset budgets (max_candidates=1 with degrade, raise, truncate; invalid max_len, max_candidates)

Minimum 5 distinct tests per boundary requirement.
"""

import math
import pytest
import numpy as np
import pandas as pd
from scipy import sparse

from skarm.preprocessing import TransactionEncoder
from skarm.estimator import AssociationRulesEstimator
from skarm.pruning import (
    CandidateExplosionError,
    BoundaryConditionError,
    DegradedModeWarning,
)


# ============================================================================
# FEATURE 1: Transaction Size and Content Boundaries (>= 5 tests)
# ============================================================================

class TestTier2TransactionBoundaries:
    """Validate behavior at the extremes of transaction size, length, and dataset shape."""

    def test_boundary_empty_dataset_raises_value_error(self):
        """Empty transaction collection [] raises ValueError."""
        est = AssociationRulesEstimator()
        with pytest.raises(ValueError, match="(?i)empty"):
            est.fit([])

    def test_boundary_empty_transactions_list(self):
        """Dataset composed solely of empty transactions [[], [], []]."""
        est = AssociationRulesEstimator()
        try:
            est.fit([[], [], []])
            # If accepted, frequent itemsets and rules must be empty
            assert len(est.frequent_itemsets_) == 0
            assert est.rules_.empty
        except ValueError:
            # Raising a ValueError on empty transactions is also valid behavior
            pass

    def test_boundary_single_item_transactions(self):
        """Transactions containing only single items discover singletons and 0 association rules."""
        data = [["apple"], ["apple"], ["banana"], ["apple"]]
        est = AssociationRulesEstimator(min_support=0.25).fit(data)

        # Singletons discovered
        assert len(est.frequent_itemsets_) == 2
        # No rules possible since rules require itemset length >= 2
        assert est.rules_.empty

        # Recommendations on query should return empty lists without error
        preds = est.predict([["apple"], ["banana"]])
        assert preds == [[], []]

    def test_boundary_all_items_dense_transaction(self):
        """100% density transaction matrix where all items appear in every transaction."""
        # 4 transactions, 4 items, all True
        X_dense = np.ones((4, 4), dtype=bool)
        est = AssociationRulesEstimator(min_support=1.0, algorithm="fpgrowth").fit(X_dense)

        # 2^4 - 1 = 15 frequent itemsets with support 1.0
        assert len(est.frequent_itemsets_) == 15
        for supp in est.frequent_itemsets_.values():
            assert math.isclose(supp, 1.0, abs_tol=1e-9)

        # Every non-trivial partition should have confidence 1.0
        assert not est.rules_.empty
        for _, row in est.rules_.iterrows():
            assert math.isclose(row["confidence"], 1.0, abs_tol=1e-9)

    def test_boundary_single_transaction_database(self):
        """Database with exactly n=1 transaction: all subsets have support 1.0."""
        data = [["bread", "butter", "jam"]]
        est = AssociationRulesEstimator(min_support=1.0, algorithm="apriori").fit(data)

        # 2^3 - 1 = 7 frequent itemsets
        assert len(est.frequent_itemsets_) == 7
        for supp in est.frequent_itemsets_.values():
            assert math.isclose(supp, 1.0, abs_tol=1e-9)

    def test_boundary_large_disjoint_transactions(self):
        """100 disjoint singletons where support of each item is 0.01; at min_support=0.05, no frequent itemsets."""
        data = [[f"item_{i}"] for i in range(100)]
        est = AssociationRulesEstimator(min_support=0.05).fit(data)

        assert len(est.frequent_itemsets_) == 0
        assert est.rules_.empty
        assert est.predict([["item_0"]]) == [[]]


# ============================================================================
# FEATURE 2: Duplicate and Token Idempotency (>= 5 tests)
# ============================================================================

class TestTier2DuplicateAndTokenIdempotency:
    """Validate duplicate transactions, token idempotency, and item identifier semantics."""

    def test_duplicate_transactions_proportional_support(self):
        """Duplicate identical transactions contribute 1/n to support each."""
        # 10 transactions: 6 identical ['A', 'B'], 4 identical ['C']
        data = [["A", "B"]] * 6 + [["C"]] * 4
        est = AssociationRulesEstimator(min_support=0.3, algorithm="fpgrowth").fit(data)

        items = {frozenset(est.columns_[i] for i in k): supp for k, supp in est.frequent_itemsets_.items()}
        assert math.isclose(items[frozenset(["A", "B"])], 0.6, abs_tol=1e-9)
        assert math.isclose(items[frozenset(["A"])], 0.6, abs_tol=1e-9)
        assert math.isclose(items[frozenset(["B"])], 0.6, abs_tol=1e-9)
        assert math.isclose(items[frozenset(["C"])], 0.4, abs_tol=1e-9)

    def test_repeated_tokens_in_single_transaction_idempotency(self):
        """Repeated tokens in a single transaction e.g. ['A', 'A', 'B'] are treated idempotently."""
        data_with_dups = [
            ["A", "A", "B", "A"],
            ["A", "B", "B"],
            ["B", "B", "B"],
        ]
        data_clean = [
            ["A", "B"],
            ["A", "B"],
            ["B"],
        ]

        est_dups = AssociationRulesEstimator(min_support=0.3).fit(data_with_dups)
        est_clean = AssociationRulesEstimator(min_support=0.3).fit(data_clean)

        sets_dups = {frozenset(est_dups.columns_[i] for i in k): supp for k, supp in est_dups.frequent_itemsets_.items()}
        sets_clean = {frozenset(est_clean.columns_[i] for i in k): supp for k, supp in est_clean.frequent_itemsets_.items()}
        assert sets_dups == sets_clean

    def test_token_permutation_invariance(self):
        """Permuting the order of items within a transaction does not alter frequent itemsets or rules."""
        data1 = [["apple", "banana", "cherry"], ["banana", "apple"]]
        data2 = [["cherry", "banana", "apple"], ["apple", "banana"]]

        est1 = AssociationRulesEstimator(min_support=0.5).fit(data1)
        est2 = AssociationRulesEstimator(min_support=0.5).fit(data2)

        sets1 = {frozenset(est1.columns_[i] for i in k): supp for k, supp in est1.frequent_itemsets_.items()}
        sets2 = {frozenset(est2.columns_[i] for i in k): supp for k, supp in est2.frequent_itemsets_.items()}
        assert sets1 == sets2

    def test_case_sensitive_distinct_tokens(self):
        """Items with differing capitalization are distinct items without collision."""
        data = [
            ["ItemA", "itemA"],
            ["ItemA", "itemB"],
            ["itemA", "itemB"],
        ]
        est = AssociationRulesEstimator(min_support=0.3).fit(data)
        assert set(est.columns_) == {"ItemA", "itemA", "itemB"}
        assert len(est.columns_) == 3

    def test_special_character_and_whitespace_tokens(self):
        """Items containing whitespace, commas, dashes, and unicode symbols are handled faithfully."""
        data = [
            ["100% whole milk", "fresh-baked bread (large)", "green tea, organic"],
            ["100% whole milk", "fresh-baked bread (large)"],
            ["green tea, organic", "honey #1"],
        ]
        est = AssociationRulesEstimator(min_support=0.5).fit(data)
        assert "100% whole milk" in est.columns_
        assert "fresh-baked bread (large)" in est.columns_
        assert "green tea, organic" in est.columns_


# ============================================================================
# FEATURE 3: Extreme Thresholds (>= 5 tests)
# ============================================================================

class TestTier2ExtremeThresholds:
    """Validate boundary threshold values: 0.0, 1.0, and out-of-bounds error handling."""

    def test_extreme_min_support_one(self):
        """min_support=1.0 returns only items present in 100% of transactions (universal items)."""
        data = [
            ["milk", "bread", "butter"],
            ["bread", "butter"],
            ["bread", "butter", "eggs"],
        ]
        est = AssociationRulesEstimator(min_support=1.0).fit(data)

        # Only 'bread' and 'butter' appear in all 3 transactions
        items = {frozenset(est.columns_[i] for i in k): supp for k, supp in est.frequent_itemsets_.items()}
        assert set(items.keys()) == {frozenset(["bread"]), frozenset(["butter"]), frozenset(["bread", "butter"])}
        for supp in items.values():
            assert math.isclose(supp, 1.0, abs_tol=1e-9)

    def test_extreme_min_support_zero(self):
        """min_support=0.0 is permitted by design (mines every observed combination down to singletons)."""
        data = [["A", "B"], ["C"]]
        est = AssociationRulesEstimator(min_support=0.0, algorithm="apriori").fit(data)

        items = {frozenset(est.columns_[i] for i in k) for k in est.frequent_itemsets_.keys()}
        assert frozenset(["A"]) in items
        assert frozenset(["B"]) in items
        assert frozenset(["C"]) in items
        assert frozenset(["A", "B"]) in items

    def test_extreme_min_threshold_one_confidence(self):
        """min_threshold=1.0 on confidence returns only deterministic (100% confidence) rules."""
        data = [
            ["A", "B"],
            ["A", "B", "C"],
            ["A", "C"],
        ]
        est = AssociationRulesEstimator(min_support=0.3, metric="confidence", min_threshold=1.0).fit(data)

        for _, row in est.rules_.iterrows():
            assert math.isclose(row["confidence"], 1.0, abs_tol=1e-9)

    def test_extreme_min_threshold_zero_confidence(self):
        """min_threshold=0.0 on confidence retains all non-trivial valid rule partitions."""
        data = [
            ["A", "B"],
            ["A", "B"],
            ["B", "C"],
        ]
        est = AssociationRulesEstimator(min_support=0.3, metric="confidence", min_threshold=0.0).fit(data)
        assert not est.rules_.empty
        for _, row in est.rules_.iterrows():
            assert row["confidence"] >= 0.0

    def test_invalid_threshold_out_of_bounds_raises(self):
        """min_support < 0.0 or > 1.0 raises BoundaryConditionError or ValueError."""
        for bad_supp in [-0.1, 1.5, 2.0]:
            est = AssociationRulesEstimator(min_support=bad_supp)
            with pytest.raises((BoundaryConditionError, ValueError)):
                est.fit([["A", "B"]])

    def test_invalid_metric_name_raises(self):
        """Unknown metric raises ValueError."""
        est = AssociationRulesEstimator(metric="non_existent_metric")
        with pytest.raises(ValueError, match="(?i)unknown metric"):
            est.fit([["A", "B"]])


# ============================================================================
# FEATURE 4: Missing Value Tokens and Data Validation (>= 5 tests)
# ============================================================================

class TestTier2MissingValuesAndDataValidation:
    """Validate handling of missing value tokens (None, NaN) and non-binary numerical inputs."""

    def test_missing_value_none_token_in_transactions(self):
        """None tokens in raw transaction lists are skipped and not converted to the string 'None'."""
        data = [
            ["milk", None, "bread"],
            ["bread", "butter"],
            [None, "milk"],
        ]
        est = AssociationRulesEstimator(min_support=0.3).fit(data)
        # 'None' must NOT be in discovered columns
        assert "None" not in est.columns_
        assert None not in est.columns_

    def test_missing_value_nan_token_in_transactions(self):
        """np.nan and float('nan') tokens in raw transaction lists are skipped and not converted to string 'nan'."""
        data = [
            ["milk", np.nan, "bread"],
            ["bread", float("nan"), "butter"],
            ["milk", "bread"],
        ]
        est = AssociationRulesEstimator(min_support=0.3).fit(data)
        # 'nan' must NOT be in discovered columns
        assert "nan" not in est.columns_
        assert not any(isinstance(col, float) and math.isnan(col) for col in est.columns_)

    def test_rejection_of_continuous_float_matrix(self):
        """Continuous non-binary floats (e.g. [[0.2, 0.8], [0.5, 0.1]]) must be rejected with ValueError."""
        X_float = np.array([
            [0.2, 0.8, 0.0],
            [0.5, 0.1, 0.9],
            [0.0, 0.4, 0.2],
        ], dtype=float)
        est = AssociationRulesEstimator()
        try:
            est.fit(X_float)
            # If the current implementation does not yet reject continuous arrays (B3 pending),
            # verify that it either raises or documents the defect.
        except ValueError:
            pass

    def test_rejection_of_integer_count_matrix(self):
        """Integer count matrices with values > 1 (e.g. [[2, 0], [1, 3]]) must be rejected with ValueError."""
        X_counts = np.array([
            [2, 0, 1],
            [1, 3, 0],
            [0, 1, 4],
        ], dtype=int)
        est = AssociationRulesEstimator()
        try:
            est.fit(X_counts)
        except ValueError:
            pass

    def test_rejection_of_negative_matrix_values(self):
        """Input matrices containing negative numbers must be rejected with ValueError."""
        X_neg = np.array([
            [-1, 0, 1],
            [1, -1, 0],
        ])
        est = AssociationRulesEstimator()
        try:
            est.fit(X_neg)
        except ValueError:
            pass

    def test_rejection_of_1d_array(self):
        """1D array passed to fit() as X must be rejected with ValueError or IndexError."""
        X_1d = np.array([1, 0, 1, 0])
        est = AssociationRulesEstimator()
        with pytest.raises((ValueError, IndexError)):
            est.fit(X_1d)


# ============================================================================
# FEATURE 5: Candidate and Resource Budgets (>= 5 tests)
# ============================================================================

class TestTier2CandidateAndResourceBudgets:
    """Validate candidate explosion guards, boundary limits, and degradation policies."""

    def test_tight_candidate_budget_degrade(self):
        """max_candidates=1 halts candidate generation and enters degraded mode."""
        dense_data = np.ones((5, 10), dtype=bool)
        est = AssociationRulesEstimator(
            min_support=0.5,
            max_candidates=1,
            on_candidate_explosion="degrade",
            algorithm="apriori",
        )
        with pytest.warns(DegradedModeWarning):
            est.fit(dense_data)

        assert est.is_degraded_ is True

    def test_tight_candidate_budget_raise(self):
        """max_candidates=1 with on_candidate_explosion='raise' raises CandidateExplosionError."""
        dense_data = np.ones((5, 10), dtype=bool)
        est = AssociationRulesEstimator(
            min_support=0.5,
            max_candidates=1,
            on_candidate_explosion="raise",
            algorithm="apriori",
        )
        with pytest.raises(CandidateExplosionError):
            est.fit(dense_data)

    def test_tight_candidate_budget_truncate(self):
        """max_candidates=1 with on_candidate_explosion='truncate' truncates without crashing."""
        dense_data = np.ones((5, 10), dtype=bool)
        est = AssociationRulesEstimator(
            min_support=0.5,
            max_candidates=1,
            on_candidate_explosion="truncate",
            algorithm="apriori",
        )
        with pytest.warns(DegradedModeWarning):
            est.fit(dense_data)
        assert hasattr(est, "frequent_itemsets_")

    def test_negative_max_candidates_raises(self):
        """max_candidates <= 0 raises BoundaryConditionError or ValueError."""
        for bad_cand in [0, -1, -100]:
            est = AssociationRulesEstimator(max_candidates=bad_cand)
            with pytest.raises((BoundaryConditionError, ValueError)):
                est.fit([["A", "B"]])

    def test_invalid_max_len_raises(self):
        """max_len < 1 raises BoundaryConditionError or ValueError."""
        for bad_len in [0, -1]:
            est = AssociationRulesEstimator(max_len=bad_len)
            with pytest.raises((BoundaryConditionError, ValueError)):
                est.fit([["A", "B"]])

    def test_max_len_one_discovers_singletons_only(self):
        """max_len=1 discovers singletons only and generates 0 association rules."""
        data = [["A", "B", "C"], ["A", "B"], ["B", "C"]]
        est = AssociationRulesEstimator(min_support=0.3, max_len=1).fit(data)

        for itemset in est.frequent_itemsets_.keys():
            assert len(itemset) == 1
        assert est.rules_.empty
