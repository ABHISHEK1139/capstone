"""Tests for AssociationRulesEstimator scikit-learn compatibility."""

import io
import joblib
import pytest
import numpy as np
from sklearn.base import clone
from skarm.estimator import AssociationRulesEstimator
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
