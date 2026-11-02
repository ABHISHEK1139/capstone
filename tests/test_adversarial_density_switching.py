"""Adversarial stress-testing and empirical verification of algorithm='auto' density/sparsity threshold.

Milestone 1 Empirical Verification:
1. Very sparse matrices (s >= 0.85, d <= 0.15): CSR and dense arrays, verifying engine selection,
   ground truth density_/sparsity_, and itemset/rule identity with explicit eclat.
2. Moderately dense matrices (s < 0.85, d > 0.15): verifying engine selection,
   ground truth density_/sparsity_, and itemset/rule identity with explicit fpgrowth.
3. Boundary condition matrices at exact density 0.15 (s = 0.85) and epsilon-neighborhoods.
4. Degenerate matrices: 0-samples, 0-features, (0, 0), all-zeros (s=1.0), all-ones (d=1.0),
   1x1, 1xN, Nx1.
5. Cross-engine algorithmic equivalence at boundary and telemetry/metadata integrity.
"""

import math
import pytest
import numpy as np
import pandas as pd
from scipy import sparse
from pandas.testing import assert_frame_equal

from skarm.estimator import AssociationRulesEstimator


# =====================================================================
# 1. Very Sparse Matrices (s >= 0.85, d <= 0.15)
# =====================================================================

class TestVerySparseMatrices:
    """Stress tests on matrices with sparsity >= 0.85 (density <= 0.15)."""

    @pytest.mark.parametrize("density", [0.01, 0.04, 0.08, 0.12, 0.149])
    @pytest.mark.parametrize("as_csr", [False, True])
    def test_sparse_matrix_engine_selection_and_equivalence(self, density, as_csr):
        """Verify algorithm='auto' selects 'eclat', accurately calculates density/sparsity,
        and produces identical itemsets and rules to explicit algorithm='eclat'.
        """
        rng = np.random.default_rng(int(density * 10000))
        n_rows, n_cols = 120, 25
        total_cells = n_rows * n_cols
        target_nnz = int(round(density * total_cells))

        flat_indices = rng.choice(total_cells, size=target_nnz, replace=False)
        flat_arr = np.zeros(total_cells, dtype=bool)
        flat_arr[flat_indices] = True
        X_dense = flat_arr.reshape((n_rows, n_cols))

        actual_nnz = int(np.count_nonzero(X_dense))
        expected_density = float(actual_nnz / total_cells)
        expected_sparsity = float(1.0 - expected_density)
        assert expected_sparsity >= 0.85, f"Expected sparse, got s={expected_sparsity}"

        X = sparse.csr_matrix(X_dense) if as_csr else X_dense

        # Fit 'auto' estimator
        est_auto = AssociationRulesEstimator(
            algorithm="auto",
            min_support=0.03,
            min_threshold=0.1,
            metric="confidence",
        )
        est_auto.fit(X)

        # 1. Verify engine selection
        assert est_auto.algorithm_selected_ == "eclat"
        assert est_auto.metadata_["algorithm_selected"] == "eclat"

        # 2. Verify density and sparsity match mathematical ground truth
        assert math.isclose(est_auto.density_, expected_density, rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(est_auto.sparsity_, expected_sparsity, rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(est_auto.metadata_["density"], expected_density, rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(est_auto.metadata_["sparsity"], expected_sparsity, rel_tol=1e-9, abs_tol=1e-12)

        # 3. Fit explicit 'eclat' estimator
        est_eclat = AssociationRulesEstimator(
            algorithm="eclat",
            min_support=0.03,
            min_threshold=0.1,
            metric="confidence",
        )
        est_eclat.fit(X)

        # 4. Verify ZERO mathematical discrepancy in frequent itemsets
        assert set(est_auto.frequent_itemsets_.keys()) == set(est_eclat.frequent_itemsets_.keys())
        for k in est_auto.frequent_itemsets_:
            assert math.isclose(
                est_auto.frequent_itemsets_[k],
                est_eclat.frequent_itemsets_[k],
                rel_tol=1e-9,
                abs_tol=1e-12,
            )

        # 5. Verify itemsets_ DataFrame
        assert_frame_equal(est_auto.itemsets_, est_eclat.itemsets_)

        # 6. Verify rules_ DataFrame
        assert_frame_equal(est_auto.rules_, est_eclat.rules_)

        # 7. Verify Eclat specific counters present
        assert "intersections_performed" in est_auto.metadata_["counters"]
        assert "tid_elements_scanned" in est_auto.metadata_["counters"]


# =====================================================================
# 2. Moderately Dense Matrices (s < 0.85, d > 0.15)
# =====================================================================

class TestModeratelyDenseMatrices:
    """Stress tests on matrices with sparsity < 0.85 (density > 0.15)."""

    @pytest.mark.parametrize("density", [0.151, 0.25, 0.40, 0.65, 0.85])
    @pytest.mark.parametrize("as_csr", [False, True])
    def test_dense_matrix_engine_selection_and_equivalence(self, density, as_csr):
        """Verify algorithm='auto' selects 'fpgrowth', accurately calculates density/sparsity,
        and produces identical itemsets and rules to explicit algorithm='fpgrowth'.
        """
        rng = np.random.default_rng(int(density * 10000))
        n_rows, n_cols = 80, 15
        total_cells = n_rows * n_cols
        target_nnz = int(round(density * total_cells))

        flat_indices = rng.choice(total_cells, size=target_nnz, replace=False)
        flat_arr = np.zeros(total_cells, dtype=bool)
        flat_arr[flat_indices] = True
        X_dense = flat_arr.reshape((n_rows, n_cols))

        actual_nnz = int(np.count_nonzero(X_dense))
        expected_density = float(actual_nnz / total_cells)
        expected_sparsity = float(1.0 - expected_density)
        assert expected_sparsity < 0.85, f"Expected dense, got s={expected_sparsity}"

        X = sparse.csr_matrix(X_dense) if as_csr else X_dense

        # Fit 'auto' estimator with max_len=3 to avoid exponential combinatorial explosion on high density
        est_auto = AssociationRulesEstimator(
            algorithm="auto",
            min_support=0.3,
            max_len=3,
            min_threshold=0.5,
            metric="confidence",
        )
        est_auto.fit(X)

        # 1. Verify engine selection
        assert est_auto.algorithm_selected_ == "fpgrowth"
        assert est_auto.metadata_["algorithm_selected"] == "fpgrowth"

        # 2. Verify density and sparsity match mathematical ground truth
        assert math.isclose(est_auto.density_, expected_density, rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(est_auto.sparsity_, expected_sparsity, rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(est_auto.metadata_["density"], expected_density, rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(est_auto.metadata_["sparsity"], expected_sparsity, rel_tol=1e-9, abs_tol=1e-12)

        # 3. Fit explicit 'fpgrowth' estimator
        est_fp = AssociationRulesEstimator(
            algorithm="fpgrowth",
            min_support=0.3,
            max_len=3,
            min_threshold=0.5,
            metric="confidence",
        )
        est_fp.fit(X)

        # 4. Verify ZERO mathematical discrepancy in frequent itemsets
        assert set(est_auto.frequent_itemsets_.keys()) == set(est_fp.frequent_itemsets_.keys())
        for k in est_auto.frequent_itemsets_:
            assert math.isclose(
                est_auto.frequent_itemsets_[k],
                est_fp.frequent_itemsets_[k],
                rel_tol=1e-9,
                abs_tol=1e-12,
            )

        # 5. Verify itemsets_ DataFrame
        assert_frame_equal(est_auto.itemsets_, est_fp.itemsets_)

        # 6. Verify rules_ DataFrame
        assert_frame_equal(est_auto.rules_, est_fp.rules_)

        # 7. Verify FP-Growth specific counters present
        assert "trees_constructed" in est_auto.metadata_["counters"]
        assert "nodes_created" in est_auto.metadata_["counters"]
        assert "paths_traversed" in est_auto.metadata_["counters"]


# =====================================================================
# 3. Boundary Condition Matrices (Exact d = 0.15, s = 0.85 & Epsilon-Neighborhoods)
# =====================================================================

class TestBoundaryConditions:
    """Stress tests exactly at and infinitesimally around the 85% sparsity threshold."""

    @pytest.mark.parametrize("shape", [(20, 10), (100, 20), (500, 40), (1000, 10)])
    @pytest.mark.parametrize("as_csr", [False, True])
    def test_exact_boundary_density_0_15_selects_eclat(self, shape, as_csr):
        """Construct matrices with exact density 0.15 (sparsity 0.85).
        Verifies branch behavior: sparsity >= 0.85 must select 'eclat'.
        """
        n_rows, n_cols = shape
        total_cells = n_rows * n_cols
        exact_nnz = int(0.15 * total_cells)
        assert exact_nnz == 0.15 * total_cells, "Matrix shape must admit exact integer 15% cells"

        rng = np.random.default_rng(42)
        flat_indices = rng.choice(total_cells, size=exact_nnz, replace=False)
        X_mat = np.zeros(total_cells, dtype=bool)
        X_mat[flat_indices] = True
        X_dense = X_mat.reshape((n_rows, n_cols))
        assert np.count_nonzero(X_dense) == exact_nnz

        X = sparse.csr_matrix(X_dense) if as_csr else X_dense

        est = AssociationRulesEstimator(algorithm="auto", min_support=0.05, max_len=3).fit(X)

        assert est.algorithm_selected_ == "eclat"
        assert est.metadata_["algorithm_selected"] == "eclat"
        assert est.density_ == 0.15
        assert est.sparsity_ == 0.85
        assert est.metadata_["matrix_density"] == 0.15
        assert est.metadata_["matrix_sparsity"] == 0.85

        # Check full equivalence with explicit eclat
        est_eclat = AssociationRulesEstimator(algorithm="eclat", min_support=0.05, max_len=3).fit(X)
        assert est.frequent_itemsets_ == est_eclat.frequent_itemsets_
        assert_frame_equal(est.itemsets_, est_eclat.itemsets_)
        assert_frame_equal(est.rules_, est_eclat.rules_)

    def test_epsilon_perturbation_around_boundary(self):
        """Test sharp transition:
        d = 0.1500000... - eps (s >= 0.85) -> eclat
        d = 0.1500000... + eps (s < 0.85) -> fpgrowth
        """
        # Shape 100 x 20 = 2000 cells.
        # Exactly 15% = 300 cells.
        # Below: 299 cells -> d = 299/2000 = 0.1495, s = 0.8505 >= 0.85 -> eclat
        # Above: 301 cells -> d = 301/2000 = 0.1505, s = 0.8495 < 0.85 -> fpgrowth

        rng = np.random.default_rng(123)
        perm = rng.permutation(2000)

        # Case A: 299 ones (sparsity > 0.85)
        X_below = np.zeros(2000, dtype=bool)
        X_below[perm[:299]] = True
        est_below = AssociationRulesEstimator(algorithm="auto", min_support=0.05, max_len=3).fit(X_below.reshape(100, 20))
        assert est_below.algorithm_selected_ == "eclat"
        assert est_below.density_ == 299 / 2000
        assert est_below.sparsity_ == 1.0 - (299 / 2000)

        # Case B: 300 ones (exact boundary: sparsity == 0.85)
        X_exact = np.zeros(2000, dtype=bool)
        X_exact[perm[:300]] = True
        est_exact = AssociationRulesEstimator(algorithm="auto", min_support=0.05, max_len=3).fit(X_exact.reshape(100, 20))
        assert est_exact.algorithm_selected_ == "eclat"
        assert est_exact.density_ == 0.15
        assert est_exact.sparsity_ == 0.85

        # Case C: 301 ones (sparsity < 0.85)
        X_above = np.zeros(2000, dtype=bool)
        X_above[perm[:301]] = True
        est_above = AssociationRulesEstimator(algorithm="auto", min_support=0.05, max_len=3).fit(X_above.reshape(100, 20))
        assert est_above.algorithm_selected_ == "fpgrowth"
        assert est_above.density_ == 301 / 2000
        assert est_above.sparsity_ == 1.0 - (301 / 2000)

    def test_fine_grained_large_matrix_boundary(self):
        """Test on large 10,000-cell matrix:
        1499 ones -> s = 0.8501 -> eclat
        1500 ones -> s = 0.8500 -> eclat
        1501 ones -> s = 0.8499 -> fpgrowth
        """
        rng = np.random.default_rng(456)
        perm = rng.permutation(10000)

        # 1499 ones
        X_1499 = np.zeros(10000, dtype=bool)
        X_1499[perm[:1499]] = True
        est_1499 = AssociationRulesEstimator(algorithm="auto", min_support=0.05, max_len=3).fit(X_1499.reshape(1000, 10))
        assert est_1499.algorithm_selected_ == "eclat"

        # 1500 ones
        X_1500 = np.zeros(10000, dtype=bool)
        X_1500[perm[:1500]] = True
        est_1500 = AssociationRulesEstimator(algorithm="auto", min_support=0.05, max_len=3).fit(X_1500.reshape(1000, 10))
        assert est_1500.algorithm_selected_ == "eclat"

        # 1501 ones
        X_1501 = np.zeros(10000, dtype=bool)
        X_1501[perm[:1501]] = True
        est_1501 = AssociationRulesEstimator(algorithm="auto", min_support=0.05, max_len=3).fit(X_1501.reshape(1000, 10))
        assert est_1501.algorithm_selected_ == "fpgrowth"
        assert est_1501.algorithm_selected_ == "fpgrowth"


# =====================================================================
# 4. Degenerate and Extreme Edge Cases
# =====================================================================

class TestDegenerateCases:
    """Stress tests on degenerate matrices (empty, zeros, ones, singletons)."""

    def test_empty_zero_samples_matrix(self):
        """0 samples: shape (0, 10) in dense, CSR, and DataFrame."""
        for X in [
            np.zeros((0, 10), dtype=bool),
            sparse.csr_matrix((0, 10), dtype=bool),
            pd.DataFrame(np.zeros((0, 10), dtype=bool)),
        ]:
            est = AssociationRulesEstimator(algorithm="auto", min_support=0.1).fit(X)
            assert est.density_ == 0.0
            assert est.sparsity_ == 1.0
            assert est.algorithm_selected_ == "eclat"
            assert est.status_ == "EXACT"
            assert est.is_complete_ is True
            assert len(est.frequent_itemsets_) == 0
            assert est.rules_.empty
            assert est.itemsets_.empty
            assert est.score() == 0.0

    def test_empty_zero_features_matrix(self):
        """0 features: shape (10, 0) in dense, CSR, and DataFrame."""
        for X in [
            np.zeros((10, 0), dtype=bool),
            sparse.csr_matrix((10, 0), dtype=bool),
            pd.DataFrame(np.zeros((10, 0), dtype=bool)),
        ]:
            est = AssociationRulesEstimator(algorithm="auto", min_support=0.1).fit(X)
            assert est.density_ == 0.0
            assert est.sparsity_ == 1.0
            assert est.algorithm_selected_ == "eclat"
            assert est.status_ == "EXACT"
            assert est.is_complete_ is True
            assert len(est.frequent_itemsets_) == 0
            assert est.rules_.empty
            assert est.itemsets_.empty
            assert est.score() == 0.0

    def test_empty_zero_by_zero_matrix(self):
        """0x0 matrix: shape (0, 0) in dense, CSR, and DataFrame."""
        for X in [
            np.zeros((0, 0), dtype=bool),
            sparse.csr_matrix((0, 0), dtype=bool),
            pd.DataFrame(np.zeros((0, 0), dtype=bool)),
        ]:
            est = AssociationRulesEstimator(algorithm="auto", min_support=0.1).fit(X)
            assert est.density_ == 0.0
            assert est.sparsity_ == 1.0
            assert est.algorithm_selected_ == "eclat"
            assert est.status_ == "EXACT"
            assert est.is_complete_ is True
            assert len(est.frequent_itemsets_) == 0
            assert est.rules_.empty
            assert est.itemsets_.empty

    def test_all_zeros_matrix(self):
        """All zeros matrix (sparsity 1.0, density 0.0) -> eclat."""
        for X in [
            np.zeros((25, 10), dtype=bool),
            sparse.csr_matrix(np.zeros((25, 10), dtype=bool)),
            pd.DataFrame(np.zeros((25, 10), dtype=bool)),
        ]:
            est = AssociationRulesEstimator(algorithm="auto", min_support=0.1).fit(X)
            assert est.density_ == 0.0
            assert est.sparsity_ == 1.0
            assert est.algorithm_selected_ == "eclat"
            assert est.status_ == "EXACT"
            assert len(est.frequent_itemsets_) == 0
            assert est.rules_.empty
            assert est.metadata_["matrix_density"] == 0.0
            assert est.metadata_["matrix_sparsity"] == 1.0

    def test_all_ones_matrix(self):
        """All ones matrix (density 1.0, sparsity 0.0) -> fpgrowth."""
        for X in [
            np.ones((20, 8), dtype=bool),
            sparse.csr_matrix(np.ones((20, 8), dtype=bool)),
            pd.DataFrame(np.ones((20, 8), dtype=bool)),
        ]:
            est = AssociationRulesEstimator(algorithm="auto", min_support=0.5).fit(X)
            assert est.density_ == 1.0
            assert est.sparsity_ == 0.0
            assert est.algorithm_selected_ == "fpgrowth"
            assert est.status_ == "EXACT"
            # 8 items all present in 100% of transactions: 2^8 - 1 = 255 itemsets
            assert len(est.frequent_itemsets_) == 255
            assert not est.rules_.empty
            assert np.all(est.rules_["support"] == 1.0)
            assert np.all(est.rules_["confidence"] == 1.0)
            assert est.metadata_["matrix_density"] == 1.0
            assert est.metadata_["matrix_sparsity"] == 0.0

    def test_single_element_matrices(self):
        """1x1 matrices: [[0]] (sparsity 1.0 -> eclat) and [[1]] (density 1.0 -> fpgrowth)."""
        # [[0]]
        X_zero = np.array([[0]], dtype=bool)
        est_zero = AssociationRulesEstimator(algorithm="auto", min_support=0.5).fit(X_zero)
        assert est_zero.density_ == 0.0
        assert est_zero.sparsity_ == 1.0
        assert est_zero.algorithm_selected_ == "eclat"
        assert len(est_zero.frequent_itemsets_) == 0

        # [[1]]
        X_one = np.array([[1]], dtype=bool)
        est_one = AssociationRulesEstimator(algorithm="auto", min_support=0.5).fit(X_one)
        assert est_one.density_ == 1.0
        assert est_one.sparsity_ == 0.0
        assert est_one.algorithm_selected_ == "fpgrowth"
        assert len(est_one.frequent_itemsets_) == 1
        assert est_one.frequent_itemsets_[frozenset([0])] == 1.0

    def test_single_row_matrix(self):
        """1x10 matrix: 1 sample with 10 features."""
        # 1 one out of 10 -> d = 0.10, s = 0.90 -> eclat
        X_sparse_row = np.zeros((1, 10), dtype=bool)
        X_sparse_row[0, 3] = True
        est_sparse = AssociationRulesEstimator(algorithm="auto", min_support=0.5).fit(X_sparse_row)
        assert math.isclose(est_sparse.density_, 0.10)
        assert math.isclose(est_sparse.sparsity_, 0.90)
        assert est_sparse.algorithm_selected_ == "eclat"
        assert len(est_sparse.frequent_itemsets_) == 1

        # 9 ones out of 10 -> d = 0.90, s = 0.10 -> fpgrowth
        X_dense_row = np.ones((1, 10), dtype=bool)
        X_dense_row[0, 0] = False
        est_dense = AssociationRulesEstimator(algorithm="auto", min_support=0.5).fit(X_dense_row)
        assert math.isclose(est_dense.density_, 0.90)
        assert math.isclose(est_dense.sparsity_, 0.10)
        assert est_dense.algorithm_selected_ == "fpgrowth"

    def test_single_column_matrix(self):
        """10x1 matrix: 10 samples with 1 feature."""
        # 1 one out of 10 -> d = 0.10, s = 0.90 -> eclat
        X_sparse_col = np.zeros((10, 1), dtype=bool)
        X_sparse_col[3, 0] = True
        est_sparse = AssociationRulesEstimator(algorithm="auto", min_support=0.05).fit(X_sparse_col)
        assert math.isclose(est_sparse.density_, 0.10)
        assert math.isclose(est_sparse.sparsity_, 0.90)
        assert est_sparse.algorithm_selected_ == "eclat"
        assert len(est_sparse.frequent_itemsets_) == 1

        # 8 ones out of 10 -> d = 0.80, s = 0.20 -> fpgrowth
        X_dense_col = np.zeros((10, 1), dtype=bool)
        X_dense_col[:8, 0] = True
        est_dense = AssociationRulesEstimator(algorithm="auto", min_support=0.5).fit(X_dense_col)
        assert math.isclose(est_dense.density_, 0.80)
        assert math.isclose(est_dense.sparsity_, 0.20)
        assert est_dense.algorithm_selected_ == "fpgrowth"
        assert len(est_dense.frequent_itemsets_) == 1


# =====================================================================
# 5. Cross-Engine Mathematical Equivalence at Boundary
# =====================================================================

class TestCrossEngineEquivalenceAtBoundary:
    """Verify that on the boundary dataset (d = 0.15), all engines (auto/eclat, fpgrowth, apriori)
    produce mathematically identical frequent itemsets and rules.
    """

    def test_cross_engine_exact_agreement_at_boundary(self):
        """Generate a realistic transaction matrix at exact density 0.15 and compare all engines."""
        rng = np.random.default_rng(2026)
        n_rows, n_cols = 100, 20
        total_cells = n_rows * n_cols
        exact_nnz = 300  # 300 / 2000 = 0.15

        flat_idx = rng.choice(total_cells, size=exact_nnz, replace=False)
        flat = np.zeros(total_cells, dtype=bool)
        flat[flat_idx] = True
        X = flat.reshape((n_rows, n_cols))

        min_supp = 0.04
        min_thresh = 0.2

        est_auto = AssociationRulesEstimator(algorithm="auto", min_support=min_supp, min_threshold=min_thresh).fit(X)
        est_eclat = AssociationRulesEstimator(algorithm="eclat", min_support=min_supp, min_threshold=min_thresh).fit(X)
        est_fp = AssociationRulesEstimator(algorithm="fpgrowth", min_support=min_supp, min_threshold=min_thresh).fit(X)
        est_ap = AssociationRulesEstimator(algorithm="apriori", min_support=min_supp, min_threshold=min_thresh).fit(X)

        assert est_auto.algorithm_selected_ == "eclat"

        # Pairwise frequent itemset comparisons
        assert set(est_auto.frequent_itemsets_.keys()) == set(est_eclat.frequent_itemsets_.keys())
        assert set(est_auto.frequent_itemsets_.keys()) == set(est_fp.frequent_itemsets_.keys())
        assert set(est_auto.frequent_itemsets_.keys()) == set(est_ap.frequent_itemsets_.keys())

        # Exact support comparison
        for itemset, supp in est_auto.frequent_itemsets_.items():
            assert math.isclose(supp, est_eclat.frequent_itemsets_[itemset], abs_tol=1e-12)
            assert math.isclose(supp, est_fp.frequent_itemsets_[itemset], abs_tol=1e-12)
            assert math.isclose(supp, est_ap.frequent_itemsets_[itemset], abs_tol=1e-12)

        # Check rule count and rules content
        assert len(est_auto.rules_) == len(est_eclat.rules_)
        assert len(est_auto.rules_) == len(est_fp.rules_)
        assert len(est_auto.rules_) == len(est_ap.rules_)

        # Rules DataFrames match between auto and eclat
        assert_frame_equal(est_auto.rules_, est_eclat.rules_)


# =====================================================================
# 6. Telemetry and Metadata Robustness
# =====================================================================

class TestMetadataAndTelemetryRobustness:
    """Verify that metadata_ dictionary is properly populated under all configurations."""

    @pytest.mark.parametrize("algorithm", ["auto", "eclat", "fpgrowth", "apriori"])
    def test_metadata_structure_and_types(self, algorithm):
        X = np.array([
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 1],
        ], dtype=bool)

        est = AssociationRulesEstimator(algorithm=algorithm, min_support=0.3).fit(X)
        meta = est.metadata_

        # Required top-level keys
        assert "status" in meta
        assert "is_complete" in meta
        assert "termination_reason" in meta
        assert "algorithm_selected" in meta
        assert "density" in meta
        assert "sparsity" in meta
        assert "matrix_density" in meta
        assert "matrix_sparsity" in meta
        assert "counters" in meta
        assert "operations" in meta
        assert "resource_usage" in meta

        # Resource usage keys
        res = meta["resource_usage"]
        assert "wall_time_seconds" in res
        assert "peak_rss_mb" in res
        assert "baseline_rss_mb" in res
        assert "incremental_rss_mb" in res
        assert res["wall_time_seconds"] >= 0.0
        assert res["peak_rss_mb"] > 0.0

        # Numeric rounding contracts
        assert isinstance(meta["matrix_density"], float)
        assert isinstance(meta["matrix_sparsity"], float)
        assert meta["matrix_density"] == round(est.density_, 6)
        assert meta["matrix_sparsity"] == round(est.sparsity_, 6)
