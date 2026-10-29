"""Tests for FP-Growth miner against Apriori and analytical oracle."""

import math
import pytest
import numpy as np
from skarm.preprocessing import TransactionEncoder
from skarm.fpgrowth import fpgrowth
from skarm.apriori import apriori
from tests.fixtures.oracle_data import (
    HAND_COMPUTED_TRANSACTIONS,
    ORACLE_FREQUENT_ITEMSETS_0_4,
)


def test_fpgrowth_matches_oracle():
    te = TransactionEncoder()
    X = te.fit_transform(HAND_COMPUTED_TRANSACTIONS)
    col_names = te.columns_

    itemsets = fpgrowth(X, min_support=0.4)
    named_itemsets = {
        frozenset(col_names[idx] for idx in k): supp
        for k, supp in itemsets.items()
    }

    assert set(named_itemsets.keys()) == set(ORACLE_FREQUENT_ITEMSETS_0_4.keys())
    for itemset, expected_supp in ORACLE_FREQUENT_ITEMSETS_0_4.items():
        assert math.isclose(named_itemsets[itemset], expected_supp, abs_tol=1e-9)


def test_fpgrowth_matches_apriori_synthetic():
    """Compare FP-Growth and Apriori on larger synthetic transaction matrix."""
    rng = np.random.default_rng(42)
    X = rng.choice([0, 1], size=(100, 12), p=[0.75, 0.25])

    apriori_res = apriori(X, min_support=0.1)
    fpgrowth_res = fpgrowth(X, min_support=0.1)

    assert set(apriori_res.keys()) == set(fpgrowth_res.keys())
    for k in apriori_res:
        assert math.isclose(apriori_res[k], fpgrowth_res[k], abs_tol=1e-9)


def test_fpgrowth_sparse():
    te = TransactionEncoder(sparse_output=True)
    X_sparse = te.fit_transform(HAND_COMPUTED_TRANSACTIONS)
    col_names = te.columns_

    itemsets = fpgrowth(X_sparse, min_support=0.4)
    named = {frozenset(col_names[i] for i in k): s for k, s in itemsets.items()}
    assert set(named.keys()) == set(ORACLE_FREQUENT_ITEMSETS_0_4.keys())
