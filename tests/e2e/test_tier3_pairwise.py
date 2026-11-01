"""Tier 3: Cross-Feature Combinations (Pairwise Interactions) Verification Suite.

Validates multi-feature interactions across data representations, engines,
pipelines, model selection, and resource limits:
- Sparse CSR matrix + Eclat + candidate budget degradation + metadata inspection
- Dense DataFrame + FP-Growth + pipeline integration + LogisticRegression downstream
- Transaction list + Apriori + GridSearchCV hyperparameter tuning with scoring
- Sample weights + multi-engine equivalence check (weighted duplicates vs explicit weights)
- Sparse CSR + auto algorithm selection + Zhang's metric filtering + predict recommendations
- Multi-dimensional resource budgets + cross-engine status consistency
"""

import math
import pytest
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from skarm.preprocessing import TransactionEncoder
from skarm.estimator import AssociationRulesEstimator
from skarm.pruning import DegradedModeWarning
from tests.fixtures.oracle_data import HAND_COMPUTED_TRANSACTIONS


class TestTier3PairwiseInteractions:
    """Validate pairwise interactions between data representations, engines, and sklearn pipelines."""

    def test_pairwise_sparse_csr_eclat_candidate_budget_degradation(self):
        """Sparse CSR matrix + Eclat + tight candidate budget degradation + metadata inspection."""
        # 20 transactions x 15 items with high density
        np.random.seed(42)
        dense_block = np.ones((20, 15), dtype=bool)
        X_csr = sparse.csr_matrix(dense_block)

        est = AssociationRulesEstimator(
            min_support=0.3,
            max_candidates=25,
            on_candidate_explosion="degrade",
            algorithm="eclat",
        )
        with pytest.warns(DegradedModeWarning):
            est.fit(X_csr)

        assert est.is_degraded_ is True
        assert hasattr(est, "frequent_itemsets_")
        assert len(est.frequent_itemsets_) > 0

        assert est.algorithm == "eclat"
        if hasattr(est, "metadata_"):
            meta = est.metadata_
            assert meta.get("status") == "DEGRADED"
            assert "future_candidates_status" in meta or "unvisited_candidates_status" in meta
            cand_status = meta.get("future_candidates_status") or meta.get("unvisited_candidates_status")
            if cand_status:
                assert cand_status.lower() in ("unknown", "exact", "none", "exhausted")

    def test_pairwise_dense_dataframe_fpgrowth_pipeline_classifier(self):
        """Dense DataFrame + FP-Growth + transform() rule features into downstream LogisticRegression."""
        # Synthetic binary classification where target y is correlated with presence of item_0 and item_1
        np.random.seed(123)
        n_samples = 60
        X_data = np.random.choice([0, 1], size=(n_samples, 6), p=[0.5, 0.5])
        # Target y = 1 if both item_0 and item_1 are present, else 0
        y = ((X_data[:, 0] == 1) & (X_data[:, 1] == 1)).astype(int)

        df = pd.DataFrame(X_data, columns=[f"item_{i}" for i in range(6)])

        # Construct scikit-learn Pipeline
        pipe = Pipeline([
            ("rules", AssociationRulesEstimator(
                min_support=0.15,
                metric="confidence",
                min_threshold=0.4,
                algorithm="fpgrowth",
                sparse_output=False,
            )),
            ("clf", LogisticRegression(solver="liblinear", random_state=42)),
        ])

        pipe.fit(df, y)
        preds = pipe.predict(df)
        assert preds.shape == (n_samples,)
        # Check accuracy is better than random (> 0.6)
        acc = np.mean(preds == y)
        assert acc >= 0.6, f"Pipeline accuracy {acc} is unexpectedly low"

    def test_pairwise_transaction_list_apriori_gridsearchcv(self):
        """Transaction list + Apriori + GridSearchCV hyperparameter tuning."""
        # Pre-encoded matrix for standard cross-validation splitting
        enc = TransactionEncoder(sparse_output=False)
        X_arr = enc.fit_transform(HAND_COMPUTED_TRANSACTIONS)
        # Dummy target for CV
        y = np.array([1, 0, 1, 0, 1])

        param_grid = {
            "min_support": [0.4, 0.6],
            "min_threshold": [0.5, 0.7],
        }

        from skarm.metrics import mean_confidence_scorer

        grid = GridSearchCV(
            estimator=AssociationRulesEstimator(algorithm="apriori", metric="confidence"),
            param_grid=param_grid,
            cv=2,
            scoring=mean_confidence_scorer,
        )

        grid.fit(X_arr, y)

        assert hasattr(grid, "best_params_")
        assert "min_support" in grid.best_params_
        assert "min_threshold" in grid.best_params_
        best_est = grid.best_estimator_
        assert isinstance(best_est, AssociationRulesEstimator)
        assert hasattr(best_est, "rules_")

    def test_pairwise_sample_weight_multi_engine_equivalence(self):
        """Sample weights + multi-engine equivalence check (weighted duplicates vs explicit weights)."""
        # Base transactions
        base_tx = [
            ["apple", "banana"],
            ["banana", "cherry"],
            ["apple", "cherry"],
            ["apple", "banana", "cherry"],
        ]
        weights = np.array([3.0, 2.0, 1.0, 4.0])  # total weight = 10.0

        # Construct duplicated dataset matching integer weights
        duplicated_tx = (
            [["apple", "banana"]] * 3
            + [["banana", "cherry"]] * 2
            + [["apple", "cherry"]] * 1
            + [["apple", "banana", "cherry"]] * 4
        )
        assert len(duplicated_tx) == 10

        enc = TransactionEncoder(sparse_output=False)
        X_base = enc.fit_transform(base_tx)
        X_dup = enc.transform(duplicated_tx)

        for algo in ["apriori", "fpgrowth", "eclat"]:
            est_weighted = AssociationRulesEstimator(min_support=0.3, algorithm=algo)
            est_weighted.fit(X_base, sample_weight=weights)

            est_unweighted = AssociationRulesEstimator(min_support=0.3, algorithm=algo)
            est_unweighted.fit(X_dup)

            sets_w = {k: round(v, 6) for k, v in est_weighted.frequent_itemsets_.items()}
            sets_u = {k: round(v, 6) for k, v in est_unweighted.frequent_itemsets_.items()}
            assert sets_w == sets_u, f"Algorithm {algo}: weighted itemsets {sets_w} != unweighted {sets_u}"

    def test_pairwise_sparse_auto_selection_zhangs_predict(self):
        """Sparse CSR + auto algorithm selection + Zhang's metric filtering + predict recommendations."""
        data = [
            ["milk", "bread", "butter"],
            ["milk", "bread"],
            ["bread", "butter"],
            ["milk", "cookies"],
            ["milk", "bread", "butter", "cookies"],
            ["beer", "chips"],
            ["beer", "chips", "nuts"],
        ]
        enc = TransactionEncoder(sparse_output=True)
        X_sparse = enc.fit_transform(data)

        est = AssociationRulesEstimator(
            min_support=0.25,
            metric="zhangs_metric",
            min_threshold=0.1,
            algorithm="auto",
        )
        est.fit(X_sparse)

        # Verify all rules have zhangs_metric >= 0.1 and <= 1.0
        assert not est.rules_.empty
        for _, row in est.rules_.iterrows():
            z = row["zhangs_metric"]
            assert 0.1 - 1e-9 <= z <= 1.0 + 1e-9

        # Recommendations for a basket with ['milk']
        query_sparse = enc.transform([["milk"]])
        preds = est.predict(query_sparse)
        assert len(preds) == 1
        # Predicted recommendations should be valid items from data
        for item in preds[0]:
            assert item in est.columns_

    def test_pairwise_multidimensional_budget_cross_engine_consistency(self):
        """Cross-engine consistency under tight candidate budget with degrade policy."""
        dense_data = np.ones((8, 12), dtype=bool)
        results = {}

        for algo in ["apriori", "fpgrowth", "eclat"]:
            est = AssociationRulesEstimator(
                min_support=0.4,
                max_candidates=20,
                on_candidate_explosion="degrade",
                algorithm=algo,
            )
            with pytest.warns(DegradedModeWarning):
                est.fit(dense_data)
            results[algo] = est.is_degraded_

        assert results["apriori"] is True
        assert results["fpgrowth"] is True
        assert results["eclat"] is True
