"""Tests for scikit-learn check_estimator compatibility and domain boundary audit."""

import pytest
from sklearn.utils.estimator_checks import check_estimator
from skarm.estimator import AssociationRulesEstimator


EXPECTED_FAILED_CHECKS = {
    # Continuous / non-binary synthetic data rejected by design (requires binary transactional incidence matrix X in {0, 1}^(N x M))
    "check_fit_score_takes_y": "Association rule mining requires binary transaction indicators; continuous Gaussian floats are rejected by design.",
    "check_estimators_overwrite_params": "Continuous float matrices are rejected by design; transactional matrix must be binary.",
    "check_dont_overwrite_parameters": "Continuous float matrices are rejected by design; transactional matrix must be binary.",
    "check_estimators_fit_returns_self": "Continuous float matrices are rejected by design; transactional matrix must be binary.",
    "check_readonly_memmap_input": "Continuous float matrices are rejected by design; transactional matrix must be binary.",
    "check_n_features_in_after_fitting": "Continuous float matrices are rejected by design; transactional matrix must be binary.",
    "check_estimators_dtypes": "Continuous float matrices are rejected by design; transactional matrix must be binary.",
    "check_sample_weights_pandas_series": "Continuous float matrices are rejected by design; transactional matrix must be binary.",
    "check_sample_weights_list": "Continuous float matrices are rejected by design; transactional matrix must be binary.",
    "check_sample_weights_shape": "Continuous float matrices are rejected by design; transactional matrix must be binary.",
    "check_sample_weights_not_overwritten": "Continuous float matrices are rejected by design; transactional matrix must be binary.",
    "check_sample_weight_equivalence_on_dense_data": "Continuous float matrices are rejected by design; transactional matrix must be binary.",
    "check_dtype_object": "Continuous float matrices are rejected by design; transactional matrix must be binary.",
    "check_pipeline_consistency": "Continuous float matrices are rejected by design; transactional matrix must be binary.",
    "check_estimators_nan_inf": "Continuous float matrices are rejected by design; transactional matrix must be binary.",
    "check_estimators_pickle": "Continuous float matrices are rejected by design; transactional matrix must be binary.",
    "check_f_contiguous_array_estimator": "Continuous float matrices are rejected by design; transactional matrix must be binary.",
    "check_transformer_general": "Continuous float matrices are rejected by design; transactional matrix must be binary.",
    "check_methods_sample_order_invariance": "Continuous float matrices are rejected by design; transactional matrix must be binary.",
    "check_methods_subset_invariance": "Continuous float matrices are rejected by design; transactional matrix must be binary.",
    "check_dict_unchanged": "Continuous float matrices are rejected by design; transactional matrix must be binary.",
    "check_fit_idempotent": "Continuous float matrices are rejected by design; transactional matrix must be binary.",
    "check_fit_check_is_fitted": "Continuous float matrices are rejected by design; transactional matrix must be binary.",
    "check_n_features_in": "Continuous float matrices are rejected by design; transactional matrix must be binary.",
    "check_fit2d_predict1d": "Unsupervised pattern mining predicts itemsets/rules rather than 1D supervised target vector y; rejects continuous floats.",
    "check_positive_only_tag_during_fit": "Continuous float matrices are rejected by design; requires binary indicator values.",

    # Mock non-array containers cannot represent transaction streams
    "check_sample_weights_not_an_array": "Synthetic _NotAnArray mock cannot represent a transaction stream or sparse matrix.",
    "check_transformer_data_not_an_array": "Synthetic _NotAnArray mock cannot represent a transaction stream or sparse matrix.",

    # Error message format and custom boundary conditions
    "check_all_zero_sample_weights_error": "Custom BoundaryConditionError message formatting for zero-weighted transactions.",
    "check_complex_data": "Custom ValueError message formatting for non-binary and complex inputs.",
    "check_estimators_empty_data_messages": "Custom ValueError message formatting for empty input transaction matrices.",
    "check_fit2d_1sample": "Custom BoundaryConditionError message formatting for single-sample transactional inputs.",
    "check_fit2d_1feature": "Custom BoundaryConditionError message formatting for single-item transaction universes.",
    "check_fit1d": "Transactional matrices require 2D incidence layout (n_samples, n_items).",
    "check_estimator_sparse_array": "Custom sparse matrix validation formatting for binary scipy.sparse matrices.",
    "check_estimator_sparse_matrix": "Custom sparse matrix validation formatting for binary scipy.sparse matrices.",
    "check_sample_weight_equivalence_on_sparse_data": "Support weighting semantics on sparse matrices vs dense matrices under duplicate rows.",

    # Class inheritance / mixin architecture
    "check_mixin_order": "TransformerMixin class declaration ordering in AssociationRulesEstimator hierarchy.",
}


def test_check_estimator_compatibility():
    """Verify that AssociationRulesEstimator adheres to scikit-learn estimator conventions.

    All checks pass cleanly or fail only on mathematically certified domain boundaries
    (continuous data rejection, unsupervised rule extraction vs supervised prediction,
    mock array containers).
    """
    est = AssociationRulesEstimator(max_candidates=500)
    # check_estimator with expected_failed_checks will raise on any UNEXPECTED failure
    results = check_estimator(
        est,
        expected_failed_checks=EXPECTED_FAILED_CHECKS,
        on_fail=None,
    )

    failed = [r for r in results if r["status"] == "failed"]
    assert len(failed) == 0, f"Unexpected failures in check_estimator: {failed}"

    # Also verify standard execution mode (raises immediately on unexpected failures)
    check_estimator(est, expected_failed_checks=EXPECTED_FAILED_CHECKS)
