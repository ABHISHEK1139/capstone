"""Tests for conventional Apriori baseline against analytical oracle."""

import math
import pytest
import numpy as np
from scipy import sparse
from skarm.preprocessing import TransactionEncoder
from skarm.apriori import apriori
from tests.fixtures.oracle_data import (
    HAND_COMPUTED_TRANSACTIONS,
    ORACLE_FREQUENT_ITEMSETS_0_4,
)


def test_apriori_matches_oracle_dense():
    te = TransactionEncoder()
    X = te.fit_transform(HAND_COMPUTED_TRANSACTIONS)
    col_names = te.columns_

    itemsets = apriori(X, min_support=0.4)

    # Convert numeric indices back to item names
    named_itemsets = {
        frozenset(col_names[idx] for idx in k): supp
        for k, supp in itemsets.items()
    }

    # Verify 100% precision and recall versus oracle
    assert set(named_itemsets.keys()) == set(ORACLE_FREQUENT_ITEMSETS_0_4.keys()), (
        f"Mismatch: found {set(named_itemsets.keys())} vs oracle {set(ORACLE_FREQUENT_ITEMSETS_0_4.keys())}"
    )

    for itemset, expected_supp in ORACLE_FREQUENT_ITEMSETS_0_4.items():
        actual_supp = named_itemsets[itemset]
        assert math.isclose(actual_supp, expected_supp, abs_tol=1e-9), (
            f"Support mismatch for {itemset}: expected {expected_supp}, got {actual_supp}"
        )


def test_apriori_sparse_input():
    te = TransactionEncoder(sparse_output=True)
    X_sparse = te.fit_transform(HAND_COMPUTED_TRANSACTIONS)
    col_names = te.columns_

    itemsets = apriori(X_sparse, min_support=0.4)
    named_itemsets = {
        frozenset(col_names[idx] for idx in k): supp
        for k, supp in itemsets.items()
    }

    assert set(named_itemsets.keys()) == set(ORACLE_FREQUENT_ITEMSETS_0_4.keys())
    for itemset, expected_supp in ORACLE_FREQUENT_ITEMSETS_0_4.items():
        assert math.isclose(named_itemsets[itemset], expected_supp, abs_tol=1e-9)


def test_apriori_max_len():
    te = TransactionEncoder()
    X = te.fit_transform(HAND_COMPUTED_TRANSACTIONS)
    itemsets = apriori(X, min_support=0.4, max_len=2)

    # Must contain no 3-itemset
    for k in itemsets.keys():
        assert len(k) <= 2
    # Should contain 1-itemsets and 2-itemsets
    lens = {len(k) for k in itemsets.keys()}
    assert lens == {1, 2}
