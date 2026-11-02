"""Adversarial stress testing and empirical oracle verification for Milestone 5.

This test module empirically challenges:
1. Memory Governor Bounded Search:
   - Synthetic transaction matrix generation across varying sizes, densities, and representations.
   - Bounded search under tight memory budgets for both 'degrade' and 'raise' policies.
   - Verifying level_halted >= 1, is_degraded_ = True, status_ = "DEGRADED", and DegradedModeWarning.
   - Verifying CandidateExplosionError on 'raise' with informative messages.
   - Verification across all engines ('apriori', 'fpgrowth', 'eclat', 'auto').
2. Estimator Pipeline & Mixin Conventions:
   - Inheritance and MRO verification for TransformerMixin and BaseEstimator.
   - Scikit-learn Pipeline with StandardScaler + Binarizer + AssociationRulesEstimator + Classifier.
   - Rejection of continuous non-binary arrays (StandardScaler without binarizer).
   - Pipeline set_output(transform="pandas") feature propagation.
   - Hyperparameter search with GridSearchCV across multiple built-in and callable scorers.
   - clone(), get_params(), set_params() contract compliance.
   - Empirical oracle verification of fit_transform() and transform() on unseen transactions.
   - Graceful handling of novel/unseen items in query transactions.
"""

import time
import copy
import pytest
import numpy as np
import pandas as pd
from scipy import sparse
from unittest.mock import patch

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Binarizer, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from skarm.estimator import AssociationRulesEstimator
from skarm.pruning import (
    PruningPolicy,
    CandidateExplosionError,
    DegradedModeWarning,
    BoundaryConditionError,
    get_current_memory_mb,
)
from skarm.preprocessing import TransactionEncoder
from skarm.metrics import (
    mean_confidence_scorer,
    mean_lift_scorer,
    total_support_scorer,
    rule_count_scorer,
)


# =====================================================================
# Synthetic Transaction Matrix Generators & Fixtures
# =====================================================================

def generate_synthetic_transactions(
    n_samples: int,
    n_items: int,
    density: float,
    format_type: str = "numpy_dense",
    seed: int = 42,
):
    """Generate synthetic transaction data with exact structural controls.

    Parameters
    ----------
    n_samples : int
        Number of transactions (rows).
    n_items : int
        Number of items/features (columns).
    density : float
        Target Bernoulli probability for item presence in [0.0, 1.0].
    format_type : {'numpy_dense', 'scipy_csr', 'scipy_csc', 'pandas_df', 'tx_list'}
        Data representation format.
    seed : int
        Random seed for determinism.

    Returns
    -------
    data : representation matching format_type
    """
    rng = np.random.RandomState(seed)
    dense_matrix = (rng.uniform(0.0, 1.0, size=(n_samples, n_items)) < density).astype(int)

    # Ensure no empty rows or empty columns for stable mining
    for r in range(n_samples):
        if np.sum(dense_matrix[r]) == 0:
            dense_matrix[r, rng.randint(0, n_items)] = 1
    for c in range(n_items):
        if np.sum(dense_matrix[:, c]) == 0:
            dense_matrix[rng.randint(0, n_samples), c] = 1

    if format_type == "numpy_dense":
        return dense_matrix
    elif format_type == "scipy_csr":
        return sparse.csr_matrix(dense_matrix, dtype=bool)
    elif format_type == "scipy_csc":
        return sparse.csc_matrix(dense_matrix, dtype=bool)
    elif format_type == "pandas_df":
        cols = [f"item_{i}" for i in range(n_items)]
        return pd.DataFrame(dense_matrix, columns=cols)
    elif format_type == "tx_list":
        cols = [f"token_{i}" for i in range(n_items)]
        tx_list = []
        for r in range(n_samples):
            items = [cols[c] for c in range(n_items) if dense_matrix[r, c] == 1]
            tx_list.append(items)
        return tx_list
    else:
        raise ValueError(f"Unknown format_type: {format_type}")


class ControllableDynamicMemorySampler:
    """Mock memory sampler that simulates active dynamic memory growth during search level >= 1."""

    def __init__(self, polling_interval: float = 0.01):
        self.polling_interval = polling_interval
        self.baseline_mb: float = 100.0
        self._peak_mb: float = 100.0
        self._check_count: int = 0

    @property
    def peak_mb(self) -> float:
        self._check_count += 1
        # Level 0 pre-fit check sees normal baseline (100 MB); active mining sees 150 MB
        if self._check_count <= 1:
            return self.baseline_mb
        return 150.0

    def start(self):
        pass

    def stop(self) -> float:
        self._peak_mb = 150.0
        return 150.0

    def is_alive(self) -> bool:
        return False


# =====================================================================
# PART 1: Memory Governor Bounded Search Adversarial Tests
# =====================================================================

class TestMemoryGovernorBoundedSearch:
    """Empirical verification of memory governor across synthetic matrices and engines."""

    @pytest.mark.parametrize(
        "n_samples,n_items,density,fmt",
        [
            (50, 12, 0.75, "numpy_dense"),    # Small dense
            (250, 30, 0.08, "scipy_csr"),      # Medium sparse
            (100, 20, 0.55, "pandas_df"),      # Medium dense DataFrame
            (400, 50, 0.04, "scipy_csc"),      # Medium-large sparse CSC
            (60, 10, 0.60, "tx_list"),         # Transaction list
        ],
    )
    @pytest.mark.parametrize("algo", ["apriori", "fpgrowth", "eclat", "auto"])
    def test_memory_governor_degrade_policy_synthetic_grid(
        self, n_samples, n_items, density, fmt, algo
    ):
        """Verify on_candidate_explosion='degrade' cleanly halts search, emits DegradedModeWarning,

        sets status_='DEGRADED', is_degraded_=True, and records level_halted >= 1 across sizes and densities.
        """
        X = generate_synthetic_transactions(n_samples, n_items, density, format_type=fmt, seed=101)

        with patch("skarm.pruning.MemorySampler", ControllableDynamicMemorySampler), \
             patch("skarm.pruning.get_current_memory_mb", return_value=100.0), \
             patch("skarm.estimator.get_current_memory_mb", return_value=100.0):

            est = AssociationRulesEstimator(
                algorithm=algo,
                min_support=0.2,
                min_threshold=0.3,
                max_memory_mb=105.0,  # Exceeded by simulated 150 MB in search
                on_candidate_explosion="degrade",
            )

            with pytest.warns(DegradedModeWarning, match="Memory budget exceeded"):
                est.fit(X)

            # Core Tri-State and Metadata Assertions
            assert est.status_ == "DEGRADED"
            assert est.is_degraded_ is True
            assert est.is_complete_ is False
            assert "Memory budget exceeded" in est.termination_reason_
            assert est.metadata_["status"] == "DEGRADED"
            assert est.metadata_["is_complete"] is False
            assert est.metadata_["level_halted"] is not None
            assert est.metadata_["level_halted"] >= 1
            assert est.metadata_["future_candidates_status"] == "UNKNOWN"

            # Check that itemsets and rules dataframes exist and are structurally valid
            assert hasattr(est, "itemsets_")
            assert isinstance(est.itemsets_, pd.DataFrame)
            assert list(est.itemsets_.columns) == ["itemsets", "length", "support"]

            assert hasattr(est, "rules_")
            assert isinstance(est.rules_, pd.DataFrame)

            # Transformed test output should not crash even in degraded state
            transformed = est.transform(X)
            assert isinstance(transformed, np.ndarray)
            assert transformed.shape[0] == n_samples
            assert transformed.shape[1] == len(est.rules_)

    @pytest.mark.parametrize("algo", ["apriori", "fpgrowth", "eclat", "auto"])
    def test_memory_governor_raise_policy_synthetic(self, algo):
        """Verify on_candidate_explosion='raise' raises CandidateExplosionError with informative message."""
        X = generate_synthetic_transactions(80, 15, 0.65, format_type="numpy_dense", seed=202)

        with patch("skarm.pruning.MemorySampler", ControllableDynamicMemorySampler), \
             patch("skarm.pruning.get_current_memory_mb", return_value=100.0), \
             patch("skarm.estimator.get_current_memory_mb", return_value=100.0):

            est = AssociationRulesEstimator(
                algorithm=algo,
                min_support=0.2,
                max_memory_mb=105.0,
                on_candidate_explosion="raise",
            )

            with pytest.raises(CandidateExplosionError, match=r"Memory budget exceeded: 150\.0MB exceeds max_memory_mb=105\.0MB\."):
                est.fit(X)

    def test_memory_governor_exact_when_budget_sufficient(self):
        """Verify that when memory budget is generous, execution completes with status_='EXACT'."""
        X = generate_synthetic_transactions(60, 12, 0.50, format_type="numpy_dense", seed=303)
        current_rss = get_current_memory_mb()

        for algo in ["apriori", "fpgrowth", "eclat", "auto"]:
            est = AssociationRulesEstimator(
                algorithm=algo,
                min_support=0.3,
                max_memory_mb=current_rss + 2048.0,
                on_candidate_explosion="degrade",
            )
            est.fit(X)

            assert est.status_ == "EXACT"
            assert est.is_degraded_ is False
            assert est.is_complete_ is True
            assert est.termination_reason_ is None
            assert est.metadata_["level_halted"] is None
            assert est.metadata_["status"] == "EXACT"
            assert est.metadata_["future_candidates_status"] == "EXHAUSTED"

    def test_memory_governor_level0_budget_trip(self):
        """Verify memory governor triggers immediately at level 0 if baseline exceeds max_memory_mb."""
        X = generate_synthetic_transactions(40, 10, 0.50, format_type="numpy_dense", seed=404)

        with patch("skarm.pruning.get_current_memory_mb", return_value=250.0), \
             patch("skarm.estimator.get_current_memory_mb", return_value=250.0):

            est = AssociationRulesEstimator(
                algorithm="fpgrowth",
                min_support=0.2,
                max_memory_mb=200.0,  # Below initial baseline (250 MB)
                on_candidate_explosion="degrade",
            )
            with pytest.warns(DegradedModeWarning, match="Memory budget exceeded"):
                est.fit(X)

            assert est.status_ == "DEGRADED"
            assert est.is_degraded_ is True
            assert est.metadata_["level_halted"] == 0
            assert len(est.frequent_itemsets_) == 0

    def test_memory_governor_and_candidate_limit_interaction(self):
        """Verify proper behavior when both candidate limit and memory budget are configured."""
        X = generate_synthetic_transactions(80, 14, 0.80, format_type="numpy_dense", seed=505)

        # Trigger candidate limit first (max_candidates=5) while memory budget is large
        est_cand = AssociationRulesEstimator(
            algorithm="apriori",
            min_support=0.1,
            max_candidates=5,
            max_memory_mb=10000.0,
            on_candidate_explosion="degrade",
        )
        with pytest.warns(DegradedModeWarning, match="Candidate explosion detected"):
            est_cand.fit(X)

        assert est_cand.status_ == "DEGRADED"
        assert "Candidate explosion" in est_cand.termination_reason_
        assert est_cand.metadata_["level_halted"] >= 1


# =====================================================================
# PART 2: Estimator Pipeline & Mixin Conventions
# =====================================================================

class TestEstimatorPipelineAndMixinConventions:
    """Empirical verification of scikit-learn Pipeline, mixin conventions, and oracles."""

    def test_mixin_inheritance_and_mro_order(self):
        """Verify AssociationRulesEstimator inherits from TransformerMixin before BaseEstimator (check_mixin_order)."""
        bases = AssociationRulesEstimator.__bases__
        assert bases == (TransformerMixin, BaseEstimator), (
            f"Expected (TransformerMixin, BaseEstimator), got {bases}"
        )
        mro = AssociationRulesEstimator.__mro__
        idx_transformer = mro.index(TransformerMixin)
        idx_base = mro.index(BaseEstimator)
        assert idx_transformer < idx_base, "TransformerMixin must precede BaseEstimator in MRO"

        # Verify estimator tags declare appropriate input formats
        tags = AssociationRulesEstimator().__sklearn_tags__()
        assert tags.target_tags.required is False
        assert tags.input_tags.sparse is True
        assert tags.input_tags.two_d_array is True

    def test_pipeline_standard_scaler_with_binarizer_and_classifier(self):
        """Verify AssociationRulesEstimator operates cleanly in a Pipeline with StandardScaler + Binarizer -> LogisticRegression."""
        rng = np.random.RandomState(42)
        n_samples = 100
        n_features = 8

        # Continuous dataset simulating real-world feature collection
        X_raw = rng.normal(loc=5.0, scale=2.0, size=(n_samples, n_features))
        # Target correlated with features 0 and 1 being high
        y = ((X_raw[:, 0] > 5.0) & (X_raw[:, 1] > 5.0)).astype(int)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("binarizer", Binarizer(threshold=0.0)),  # threshold standard normal at 0
            ("rules", AssociationRulesEstimator(
                min_support=0.15,
                metric="confidence",
                min_threshold=0.5,
                algorithm="fpgrowth",
            )),
            ("clf", LogisticRegression(random_state=42, solver="liblinear")),
        ])

        # Fit end-to-end pipeline
        pipe.fit(X_raw, y)

        # Predict and score
        preds = pipe.predict(X_raw)
        assert preds.shape == (n_samples,)
        acc = pipe.score(X_raw, y)
        assert acc >= 0.65, f"Expected classification accuracy >= 0.65, got {acc}"

        # Inspect intermediate rule transformer step
        rules_step = pipe.named_steps["rules"]
        assert hasattr(rules_step, "rules_")
        assert len(rules_step.rules_) > 0

    def test_pipeline_rejection_of_continuous_features_without_binarization(self):
        """Verify AssociationRulesEstimator strictly rejects continuous floats (StandardScaler without binarizer) per Bug B3."""
        rng = np.random.RandomState(42)
        X_continuous = rng.normal(loc=0.0, scale=1.0, size=(50, 6))

        pipe_invalid = Pipeline([
            ("scaler", StandardScaler()),
            ("rules", AssociationRulesEstimator(algorithm="apriori")),
        ])

        with pytest.raises(ValueError, match="Input matrix values must be strictly binary"):
            pipe_invalid.fit(X_continuous)

    def test_pipeline_set_output_pandas_and_feature_names(self):
        """Verify Pipeline with set_output(transform='pandas') outputs DataFrame with rule names."""
        X_df = generate_synthetic_transactions(60, 8, 0.45, format_type="pandas_df", seed=777)

        pipe = Pipeline([
            ("rules", AssociationRulesEstimator(
                min_support=0.2,
                min_threshold=0.4,
                algorithm="eclat",
            ))
        ])
        pipe.set_output(transform="pandas")
        pipe.fit(X_df)

        transformed_df = pipe.transform(X_df)
        assert isinstance(transformed_df, pd.DataFrame)
        assert transformed_df.shape[0] == 60
        assert transformed_df.shape[1] == len(pipe.named_steps["rules"].rules_)

        # Check column names match get_feature_names_out()
        expected_feature_names = pipe.named_steps["rules"].get_feature_names_out()
        assert list(transformed_df.columns) == list(expected_feature_names)
        for col_name in transformed_df.columns:
            assert " -> " in col_name
            assert col_name.startswith("{")

    def test_pipeline_zero_rule_scenario_graceful_handling(self):
        """Verify Pipeline handles zero generated rules gracefully (empty boolean array)."""
        X = generate_synthetic_transactions(40, 6, 0.1, format_type="numpy_dense", seed=888)

        # High support yields 0 rules
        pipe = Pipeline([
            ("rules", AssociationRulesEstimator(
                min_support=0.99,
                metric="confidence",
                min_threshold=0.99,
            ))
        ])
        pipe.fit(X)

        transformed = pipe.transform(X)
        assert transformed.shape == (40, 0)
        assert transformed.dtype == bool

        feature_names = pipe.named_steps["rules"].get_feature_names_out()
        assert len(feature_names) == 0

    def test_gridsearchcv_hyperparameter_tuning_across_scorers(self):
        """Verify AssociationRulesEstimator in GridSearchCV with multiple parameters and scoring metrics."""
        X = generate_synthetic_transactions(80, 8, 0.40, format_type="numpy_dense", seed=999)
        dummy_y = np.ones(80, dtype=int)

        param_grid = {
            "min_support": [0.25, 0.35],
            "min_threshold": [0.4, 0.6],
            "algorithm": ["apriori", "fpgrowth"],
        }

        # Test built-in scorers: mean_confidence, mean_lift, total_support, rule_count
        for scorer_name, scorer_fn in [
            ("mean_confidence", mean_confidence_scorer),
            ("mean_lift", mean_lift_scorer),
            ("total_support", total_support_scorer),
            ("rule_count", rule_count_scorer),
        ]:
            grid = GridSearchCV(
                estimator=AssociationRulesEstimator(metric="confidence", scoring=scorer_name),
                param_grid=param_grid,
                cv=2,
                scoring=scorer_fn,
            )
            grid.fit(X, dummy_y)

            assert hasattr(grid, "best_params_")
            assert hasattr(grid, "best_score_")
            assert isinstance(grid.best_score_, float)
            assert isinstance(grid.best_estimator_, AssociationRulesEstimator)
            assert hasattr(grid.best_estimator_, "rules_")

        # Custom callable scorer
        def custom_scorer(estimator, X_val, y_val=None):
            return float(len(estimator.rules_) * 2.5)

        grid_custom = GridSearchCV(
            estimator=AssociationRulesEstimator(algorithm="eclat"),
            param_grid={"min_support": [0.3, 0.4]},
            cv=2,
            scoring=custom_scorer,
        )
        grid_custom.fit(X, dummy_y)
        assert grid_custom.best_score_ >= 0.0

    def test_estimator_clone_get_set_params(self):
        """Verify clone(), get_params(), and set_params() contract compliance."""
        est = AssociationRulesEstimator(
            min_support=0.22,
            metric="lift",
            min_threshold=1.5,
            algorithm="eclat",
            max_len=4,
            max_candidates=25000,
            on_candidate_explosion="raise",
            sparse_output=False,
            max_time_seconds=12.0,
            max_memory_mb=512.0,
            scoring="mean_lift",
        )

        # 1. get_params()
        params = est.get_params(deep=True)
        assert params["min_support"] == 0.22
        assert params["metric"] == "lift"
        assert params["min_threshold"] == 1.5
        assert params["algorithm"] == "eclat"
        assert params["max_len"] == 4
        assert params["max_candidates"] == 25000
        assert params["on_candidate_explosion"] == "raise"
        assert params["sparse_output"] is False
        assert params["max_time_seconds"] == 12.0
        assert params["max_memory_mb"] == 512.0
        assert params["scoring"] == "mean_lift"

        # 2. clone() on unfitted estimator
        cloned_unfitted = clone(est)
        assert cloned_unfitted.get_params() == params
        assert cloned_unfitted is not est

        # 3. Fit original estimator and clone fitted estimator
        X = generate_synthetic_transactions(40, 6, 0.5, format_type="numpy_dense", seed=123)
        est.fit(X)
        assert hasattr(est, "rules_")
        assert hasattr(est, "status_")

        cloned_fitted = clone(est)
        assert not hasattr(cloned_fitted, "rules_")
        assert not hasattr(cloned_fitted, "status_")
        assert not hasattr(cloned_fitted, "itemsets_")
        assert cloned_fitted.get_params() == params

        # 4. set_params()
        mutated = est.set_params(min_support=0.45, algorithm="fpgrowth")
        assert mutated is est
        assert est.min_support == 0.45
        assert est.algorithm == "fpgrowth"

    def test_oracle_rule_firing_transform_on_unseen_transactions(self):
        """Mathematical Oracle Verification:

        Verify that for every unseen transaction T and rule X -> Y:
        est.transform(T)[rule_idx] == True iff X is a subset of T.
        """
        # Train dataset
        X_train = generate_synthetic_transactions(100, 10, 0.50, format_type="numpy_dense", seed=42)
        est = AssociationRulesEstimator(
            min_support=0.15,
            metric="confidence",
            min_threshold=0.5,
            algorithm="fpgrowth",
        )
        est.fit(X_train)
        assert len(est.rules_) > 0, "No rules discovered on training set"

        # Test dataset (unseen)
        X_test = generate_synthetic_transactions(150, 10, 0.40, format_type="numpy_dense", seed=999)
        transformed = est.transform(X_test)
        assert transformed.shape == (150, len(est.rules_))
        assert transformed.dtype == bool

        # Independent Python Mathematical Oracle
        rule_antecedents = list(est.rules_["antecedent"])  # list of frozensets of item indices
        for i in range(150):
            tx_row = X_test[i]
            tx_item_set = set(np.where(tx_row)[0])
            for r_idx, ante in enumerate(rule_antecedents):
                oracle_fired = ante.issubset(tx_item_set)
                assert transformed[i, r_idx] == oracle_fired, (
                    f"Oracle mismatch at sample {i}, rule {r_idx}: "
                    f"antecedent={ante}, tx={tx_item_set}, "
                    f"oracle={oracle_fired}, skarm={transformed[i, r_idx]}"
                )

    def test_fit_transform_equivalence_oracle(self):
        """Verify that est.fit_transform(X) is strictly identical to est.fit(X).transform(X)."""
        X = generate_synthetic_transactions(70, 8, 0.45, format_type="numpy_dense", seed=555)

        est1 = AssociationRulesEstimator(min_support=0.2, min_threshold=0.4, algorithm="eclat")
        out_fit_transform = est1.fit_transform(X)

        est2 = AssociationRulesEstimator(min_support=0.2, min_threshold=0.4, algorithm="eclat")
        est2.fit(X)
        out_transform = est2.transform(X)

        np.testing.assert_array_equal(out_fit_transform, out_transform)

    def test_unseen_transactions_with_novel_items_graceful_handling(self):
        """Verify transform() gracefully handles unseen/novel item tokens not in the training universe."""
        train_tx = [
            ["milk", "bread"],
            ["bread", "butter"],
            ["milk", "bread", "butter"],
            ["milk", "eggs"],
            ["bread", "eggs"],
        ]

        est = AssociationRulesEstimator(min_support=0.2, metric="confidence", min_threshold=0.5)
        est.fit(train_tx)

        # Unseen query transactions with novel items: "coffee", "jam", "beer"
        test_tx = [
            ["milk", "bread", "coffee"],        # contains milk, bread (known) + coffee (novel)
            ["coffee", "jam", "beer"],          # completely novel items
            [],                                 # empty transaction
            ["bread", "butter"],                # exact match
        ]

        transformed = est.transform(test_tx)
        assert transformed.shape == (4, len(est.rules_))
        assert transformed.dtype == bool

        # Completely novel transaction must fire zero rules (no known antecedents)
        assert not np.any(transformed[1, :])
        # Empty transaction must fire zero rules
        assert not np.any(transformed[2, :])

    def test_column_transformer_mixed_pipeline(self):
        """Verify AssociationRulesEstimator functions downstream of a ColumnTransformer."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder

        # Mixed feature table: cols 0-1 continuous, cols 2-3 categorical strings
        df_mixed = pd.DataFrame({
            "num1": [1.2, 5.4, 0.3, 8.1, 4.2, 9.0] * 10,
            "num2": [2.5, 1.1, 7.8, 3.4, 6.7, 0.2] * 10,
            "cat1": ["A", "B", "A", "C", "B", "A"] * 10,
            "cat2": ["X", "Y", "X", "Y", "X", "Y"] * 10,
        })
        y = np.array([0, 1, 0, 1, 1, 0] * 10)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("scale", StandardScaler()), ("bin", Binarizer(threshold=0.0))]), ["num1", "num2"]),
                ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["cat1", "cat2"]),
            ]
        )

        pipe = Pipeline([
            ("prep", preprocessor),
            ("rules", AssociationRulesEstimator(min_support=0.1, min_threshold=0.3, algorithm="auto")),
            ("clf", LogisticRegression(solver="liblinear", random_state=42)),
        ])

        pipe.fit(df_mixed, y)
        preds = pipe.predict(df_mixed)
        assert preds.shape == (60,)
        acc = pipe.score(df_mixed, y)
        assert 0.0 <= acc <= 1.0

    def test_extreme_density_combinatorial_explosion_governor(self):
        """Adversarial Stress Test: All-ones dense matrix (2^16 - 1 = 65,535 candidate lattice).

        Ensures all 4 engines ('apriori', 'fpgrowth', 'eclat', 'auto') bound execution under candidate and memory caps
        without hanging, recursing infinitely, or raising unhandled exceptions.
        """
        # 30 samples, 16 features, all 1s
        X_all_ones = np.ones((30, 16), dtype=int)

        for algo in ["apriori", "fpgrowth", "eclat", "auto"]:
            # Degrade policy with low candidate budget (50 candidates)
            est_deg = AssociationRulesEstimator(
                algorithm=algo,
                min_support=0.5,
                max_candidates=50,
                on_candidate_explosion="degrade",
            )
            with pytest.warns(DegradedModeWarning):
                est_deg.fit(X_all_ones)

            assert est_deg.status_ == "DEGRADED"
            assert est_deg.is_degraded_ is True
            assert est_deg.metadata_["level_halted"] is not None
            assert len(est_deg.frequent_itemsets_) <= 100

            # Raise policy with low candidate budget
            est_raise = AssociationRulesEstimator(
                algorithm=algo,
                min_support=0.5,
                max_candidates=50,
                on_candidate_explosion="raise",
            )
            with pytest.raises(CandidateExplosionError):
                est_raise.fit(X_all_ones)

    def test_gridsearchcv_with_degraded_fold_resilience(self):
        """Verify GridSearchCV completes cleanly and scores correctly even when some parameter combinations degrade."""
        X = generate_synthetic_transactions(60, 10, 0.70, format_type="numpy_dense", seed=321)
        dummy_y = np.zeros(60)

        # One parameter setting will degrade (max_candidates=5), while another stays exact (max_candidates=10000)
        param_grid = {
            "max_candidates": [5, 10000],
            "min_support": [0.4],
        }

        grid = GridSearchCV(
            estimator=AssociationRulesEstimator(algorithm="auto", on_candidate_explosion="degrade"),
            param_grid=param_grid,
            cv=2,
            scoring=mean_confidence_scorer,
        )
        # Suppress or catch warnings from the intentionally degraded parameter setting
        with pytest.warns(DegradedModeWarning):
            grid.fit(X, dummy_y)

        assert hasattr(grid, "best_estimator_")
        assert hasattr(grid, "cv_results_")
        assert len(grid.cv_results_["params"]) == 2

    def test_comprehensive_metadata_schema_verification(self):
        """Verify metadata_ dictionary strictly conforms to formal contracts (R2 & R3)."""
        X = generate_synthetic_transactions(50, 10, 0.40, format_type="numpy_dense", seed=111)
        est = AssociationRulesEstimator(min_support=0.2, min_threshold=0.4, algorithm="fpgrowth")
        est.fit(X)

        meta = est.metadata_
        expected_keys = [
            "status",
            "is_complete",
            "termination_reason",
            "level_halted",
            "candidates_observed_before_halt",
            "excess_candidates_observed",
            "unexplored_candidates_count",
            "candidates_generated_at_level",
            "candidates_remaining_at_level",
            "future_candidates_status",
            "frequent_itemsets_discovered",
            "is_candidate_count_exact",
            "algorithm_selected",
            "density",
            "sparsity",
            "matrix_density",
            "matrix_sparsity",
            "counters",
            "operations",
            "resource_usage",
        ]
        for key in expected_keys:
            assert key in meta, f"Missing required metadata key: {key}"

        res_usage = meta["resource_usage"]
        assert "wall_time_seconds" in res_usage
        assert "peak_rss_mb" in res_usage
        assert "baseline_rss_mb" in res_usage
        assert "incremental_rss_mb" in res_usage
        assert res_usage["wall_time_seconds"] >= 0.0
        assert res_usage["peak_rss_mb"] >= res_usage["baseline_rss_mb"]

