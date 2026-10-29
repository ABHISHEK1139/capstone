"""Unit tests for TransactionEncoder."""

import pytest
import numpy as np
from scipy import sparse
from skarm.preprocessing import TransactionEncoder


def test_transaction_encoder_dense():
    transactions = [
        ["milk", "bread"],
        ["bread", "butter"],
        ["milk", "bread", "butter"],
    ]
    te = TransactionEncoder(sparse_output=False)
    X = te.fit_transform(transactions)

    assert isinstance(X, np.ndarray)
    assert X.shape == (3, 3)
    assert te.columns_ == ["bread", "butter", "milk"]

    # In row 0: bread (col 0), milk (col 2)
    assert np.array_equal(X[0], [True, False, True])
    # In row 1: bread (col 0), butter (col 1)
    assert np.array_equal(X[1], [True, True, False])
    # In row 2: all True
    assert np.array_equal(X[2], [True, True, True])

    # Inverse transform
    inv = te.inverse_transform(X)
    assert inv == [["bread", "milk"], ["bread", "butter"], ["bread", "butter", "milk"]]


def test_transaction_encoder_sparse():
    transactions = [
        ["apple", "banana"],
        ["banana", "cherry"],
        ["apple"],
    ]
    te = TransactionEncoder(sparse_output=True)
    X = te.fit_transform(transactions)

    assert sparse.issparse(X)
    assert X.shape == (3, 3)
    assert te.columns_ == ["apple", "banana", "cherry"]

    inv = te.inverse_transform(X)
    assert inv == [["apple", "banana"], ["banana", "cherry"], ["apple"]]


def test_transaction_encoder_duplicate_items():
    """Verify idempotent handling of duplicate items (NT-2)."""
    transactions = [
        ["milk", "bread", "milk", "bread", "milk"],
        ["beer", "beer"],
    ]
    te = TransactionEncoder(sparse_output=False)
    X = te.fit_transform(transactions)

    assert te.columns_ == ["beer", "bread", "milk"]
    # Row 0: bread, milk
    assert np.array_equal(X[0], [False, True, True])
    # Row 1: beer
    assert np.array_equal(X[1], [True, False, False])


def test_transaction_encoder_unknown_items():
    te = TransactionEncoder(handle_unknown="ignore")
    te.fit([["a", "b"]])
    X = te.transform([["a", "c", "d"]])
    assert np.array_equal(X[0], [True, False])

    te_err = TransactionEncoder(handle_unknown="error")
    te_err.fit([["a", "b"]])
    with pytest.raises(ValueError, match="Encountered unknown item"):
        te_err.transform([["a", "c"]])


def test_transaction_encoder_empty_input():
    te = TransactionEncoder()
    with pytest.raises(ValueError, match="Empty transaction list"):
        te.fit([])
