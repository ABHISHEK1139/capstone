"""Preprocessing tools for transaction datasets."""

from typing import Iterable, List, Sequence, Union, Optional, Any
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class TransactionEncoder(BaseEstimator, TransformerMixin):
    """Encodes transactions (iterables of items) into one-hot boolean matrices.

    Converts transaction data into a binary indicator matrix where each row
    corresponds to a transaction and each column corresponds to a distinct item.
    Supports both dense NumPy boolean arrays and Scipy CSR sparse matrices.

    Parameters
    ----------
    sparse_output : bool, default=False
        If True, `transform` returns a `scipy.sparse.csr_matrix`.
        If False, returns a 2D boolean NumPy array.
    handle_unknown : {'ignore', 'error'}, default='ignore'
        Whether to raise an error or ignore unknown items encountered during transform.
    dtype : data-type, default=bool
        Desired output array or matrix data-type.

    Attributes
    ----------
    columns_ : list of str/obj
        Sorted list of unique items discovered across all transactions during fit.
    item_to_idx_ : dict
        Mapping from item identifier to column index.
    n_features_in_ : int
        Number of distinct features (items) observed during fit.
    """

    def __init__(
        self,
        sparse_output: bool = False,
        handle_unknown: str = "ignore",
        dtype: Any = bool,
    ):
        self.sparse_output = sparse_output
        self.handle_unknown = handle_unknown
        self.dtype = dtype

    def fit(self, X: Iterable[Iterable[Any]], y: Optional[Any] = None) -> "TransactionEncoder":
        """Learn unique items across all transactions.

        Parameters
        ----------
        X : iterable of iterables
            Transaction data, e.g., [['milk', 'bread'], ['bread', 'butter']].
        y : None
            Ignored. Present for API consistency with scikit-learn.

        Returns
        -------
        self : TransactionEncoder
            Fitted encoder.
        """
        if not hasattr(X, "__iter__"):
            raise ValueError("Input X must be an iterable of transactions.")

        unique_items = set()
        has_transactions = False
        for transaction in X:
            has_transactions = True
            if not hasattr(transaction, "__iter__") or isinstance(transaction, (str, bytes)):
                # If transaction itself is a string/bytes, treat as iterable of individual characters or raise?
                # Usually a transaction is a list/tuple/set of item strings.
                if isinstance(transaction, (str, bytes)):
                    unique_items.add(transaction)
                    continue
            for item in transaction:
                unique_items.add(item)

        if not has_transactions:
            raise ValueError("Empty transaction list provided to fit().")

        try:
            self.columns_ = sorted(list(unique_items))
        except TypeError:
            # If items are not mutually comparable, sort string representations
            self.columns_ = sorted(list(unique_items), key=lambda x: str(x))

        self.item_to_idx_ = {item: idx for idx, item in enumerate(self.columns_)}
        self.n_features_in_ = len(self.columns_)
        return self

    def transform(self, X: Iterable[Iterable[Any]]) -> Union[np.ndarray, sparse.csr_matrix]:
        """Transform transactions into binary one-hot representation.

        Handles duplicate items within a single transaction idempotently (NT-2).

        Parameters
        ----------
        X : iterable of iterables
            Transactions to encode.

        Returns
        -------
        X_encoded : np.ndarray or scipy.sparse.csr_matrix
            One-hot encoded binary representation.
        """
        check_is_fitted(self, attributes=["columns_", "item_to_idx_"])

        if self.handle_unknown not in ("ignore", "error"):
            raise ValueError(f"handle_unknown must be 'ignore' or 'error', got {self.handle_unknown}")

        # Materialize transactions or iterate
        transactions = list(X)
        n_samples = len(transactions)
        n_features = len(self.columns_)

        if self.sparse_output:
            # Build sparse matrix via coordinate lists (indptr, indices)
            indptr = [0]
            indices = []
            for transaction in transactions:
                # Deduplicate items within transaction (NT-2 requirement)
                seen_indices = set()
                for item in transaction:
                    if item in self.item_to_idx_:
                        seen_indices.add(self.item_to_idx_[item])
                    elif self.handle_unknown == "error":
                        raise ValueError(f"Encountered unknown item '{item}' during transform.")
                # Sorted column indices for CSR invariant
                sorted_idx = sorted(seen_indices)
                indices.extend(sorted_idx)
                indptr.append(len(indices))

            data = np.ones(len(indices), dtype=self.dtype)
            matrix = sparse.csr_matrix(
                (data, indices, indptr),
                shape=(n_samples, n_features),
                dtype=self.dtype,
            )
            return matrix
        else:
            matrix = np.zeros((n_samples, n_features), dtype=self.dtype)
            for row_idx, transaction in enumerate(transactions):
                for item in transaction:
                    if item in self.item_to_idx_:
                        col_idx = self.item_to_idx_[item]
                        matrix[row_idx, col_idx] = 1
                    elif self.handle_unknown == "error":
                        raise ValueError(f"Encountered unknown item '{item}' during transform.")
            return matrix

    def fit_transform(self, X: Iterable[Iterable[Any]], y: Optional[Any] = None) -> Union[np.ndarray, sparse.csr_matrix]:
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: Union[np.ndarray, sparse.csr_matrix]) -> List[List[Any]]:
        """Transform binary matrix back to transactions.

        Parameters
        ----------
        X : np.ndarray or scipy.sparse.csr_matrix
            Binary representation of transactions.

        Returns
        -------
        transactions : list of list of items
            Decoded transactions.
        """
        check_is_fitted(self, attributes=["columns_"])

        if sparse.issparse(X):
            X = X.tocsr()
            transactions = []
            for i in range(X.shape[0]):
                row_cols = X.indices[X.indptr[i]: X.indptr[i + 1]]
                transactions.append([self.columns_[c] for c in row_cols])
            return transactions
        else:
            X = np.asarray(X, dtype=bool)
            transactions = []
            for row in X:
                transactions.append([self.columns_[idx] for idx, val in enumerate(row) if val])
            return transactions

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        """Get output feature names for transformation."""
        check_is_fitted(self, attributes=["columns_"])
        return np.array(self.columns_, dtype=object)
