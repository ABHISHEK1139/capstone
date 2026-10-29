"""Tests for Eclat vertical miner against Apriori and analytical oracle."""

import math
import pytest
import numpy as np
from skarm.preprocessing import TransactionEncoder
from skarm.eclat import eclat
from skarm.apriori import apriori
from tests.fixtures.oracle_data import (
    HAND_COMPUTED_TRANSACTIONS,
    ORACLE_FREQUENT_ITEMSETS_0_4,
)


def test_eclat_matches_oracle():
    te = TransactionEncoder()
    X = te.fit_transform(HAND_COMPUTED_TRANSACTIONS)
    col_names = te.columns_

    itemsets = eclat(X, min_support=0.4)
    named_itemsets = {
        frozenset(col_names[idx] for idx in k): supp
        for k, supp in itemsets.items()
    }

    assert set(named_itemsets.keys()) == set(ORACLE_FREQUENT_ITEMSETS_0_4.keys())
    for itemset, expected_supp in ORACLE_FREQUENT_ITEMSETS_0_4.items():
        assert math.isclose(named_itemsets[itemset], expected_supp, abs_tol=1e-9)


def test_eclat_matches_apriori():
    rng = np.random.default_rng(101)
    X = rng.choice([0, 1], size=(80, 10), p=[0.7, 0.3])

    apriori_res = apriori(X, min_support=0.15)
    eclat_res = eclat(X, min_support=0.15)

    assert set(apriori_res.keys()) == set(eclat_res.keys())
    for k in apriori_res:
        assert math.isclose(apriori_res[k], eclat_res[k], abs_tol=1e-9)
