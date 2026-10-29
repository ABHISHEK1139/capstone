"""Scikit-learn compatible Association Rule Mining Estimator."""

from typing import Optional, Union, Iterable, List, Dict, Any, FrozenSet
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from skarm.preprocessing import TransactionEncoder
from skarm.apriori import apriori
from skarm.fpgrowth import fpgrowth
from skarm.eclat import eclat
from skarm.rules import association_rules
from skarm.pruning import PruningPolicy, BoundaryConditionError
from skarm.telemetry import TelemetryLogger


class AssociationRulesEstimator(BaseEstimator):
    """Scikit-learn compatible estimator for association rule mining.

    Supports Apriori, FP-Growth, and Eclat algorithms with configurable
    candidate explosion guards, rule metrics, and serialization.

    Parameters
    ----------
    min_support : float, default=0.1
        Minimum support threshold in [0.0, 1.0].
    metric : str, default='confidence'
        Metric for filtering rules ('confidence', 'lift', 'support', 'leverage', 'conviction', 'zhangs_metric').
    min_threshold : float, default=0.5
        Minimum threshold for the chosen metric.
    algorithm : {'auto', 'apriori', 'fpgrowth', 'eclat'}, default='auto'
        Mining algorithm:
        - 'auto': Automatically select fastest engine based on input density.
        - 'apriori': Standard level-wise Apriori baseline.
        - 'fpgrowth': Prefix-tree pattern growth.
        - 'eclat': Vertical TID-list intersection.
    max_len : int or None, default=None
        Maximum itemset length.
    max_candidates : int, default=100000
        Maximum candidates allowed per level to prevent exponential explosion.
    on_candidate_explosion : {'raise', 'truncate', 'degrade'}, default='degrade'
        Action taken when candidate count crosses max_candidates (NT-1, NT-6).

    Attributes
    ----------
    frequent_itemsets_ : dict
        Mapping from frozenset of item indices to support values.
    rules_ : pd.DataFrame
        DataFrame containing discovered association rules.
    encoder_ : TransactionEncoder or None
        Fitted encoder if transactions were provided as iterables.
    columns_ : list of str
        Item names/column headers.
    n_features_in_ : int
        Number of items/features seen during fit.
    is_degraded_ : bool
        Whether candidate bounding forced the estimator into degraded mode.
    telemetry_ : TelemetryLogger
        Diagnostic logs including peak memory, execution time, and level metrics.
    """

    def __init__(
        self,
        min_support: float = 0.1,
        metric: str = "confidence",
        min_threshold: float = 0.5,
        algorithm: str = "auto",
        max_len: Optional[int] = None,
        max_candidates: int = 100000,
        on_candidate_explosion: str = "degrade",
        sparse_output: bool = True,
    ):
        self.min_support = min_support
        self.metric = metric
        self.min_threshold = min_threshold
        self.algorithm = algorithm
        self.max_len = max_len
        self.max_candidates = max_candidates
        self.on_candidate_explosion = on_candidate_explosion
        self.sparse_output = sparse_output

    def _validate_parameters(self):
        """Validate constructor parameters against boundary conditions (NT-4)."""
        if not (0.0 <= self.min_support <= 1.0):
            raise BoundaryConditionError(f"min_support must be between 0.0 and 1.0, got {self.min_support}")
        if self.max_candidates <= 0:
            raise BoundaryConditionError(f"max_candidates must be > 0, got {self.max_candidates}")
        if self.max_len is not None and self.max_len < 1:
            raise BoundaryConditionError(f"max_len must be >= 1, got {self.max_len}")
        if self.algorithm not in ("auto", "apriori", "fpgrowth", "eclat"):
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        valid_metrics = {"confidence", "lift", "support", "leverage", "conviction", "zhangs_metric"}
        if self.metric not in valid_metrics:
            raise ValueError(f"Unknown metric '{self.metric}'. Choose from {valid_metrics}")

    def fit(
        self,
        X: Union[Iterable[Iterable[Any]], np.ndarray, sparse.csr_matrix],
        y: Optional[Any] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "AssociationRulesEstimator":
        """Fit frequent itemsets and generate association rules.

        Parameters
        ----------
        X : iterable of iterables, or 2D array-like/sparse matrix
            Transactions or binary indicator matrix.
        y : None
            Ignored.
        sample_weight : np.ndarray of shape (n_samples,) or None
            Sample weights per transaction (NT-3).

        Returns
        -------
        self : AssociationRulesEstimator
            Fitted estimator.
        """
        self._validate_parameters()

        # A DataFrame of 0/1 indicators is a matrix, not an iterable of
        # transactions (iterating it would yield column names).
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        # Handle raw iterables of transactions vs pre-encoded matrices
        if isinstance(X, (np.ndarray, sparse.spmatrix)):
            X_matrix = X
            self.encoder_ = None
            self.columns_ = [f"item_{i}" for i in range(X_matrix.shape[1])]
            self.n_features_in_ = X_matrix.shape[1]
        else:
            # Transaction list of lists
            self.encoder_ = TransactionEncoder(sparse_output=self.sparse_output)
            X_matrix = self.encoder_.fit_transform(X)
            self.columns_ = list(self.encoder_.columns_)
            self.n_features_in_ = len(self.columns_)

        # Setup pruning and telemetry
        self.pruning_policy_ = PruningPolicy(
            max_candidates=self.max_candidates,
            max_len=self.max_len,
            min_support=self.min_support,
            on_candidate_explosion=self.on_candidate_explosion,
        )
        self.telemetry_ = TelemetryLogger()

        # Select algorithm
        algo = self.algorithm
        if algo == "auto":
            if sparse.issparse(X_matrix):
                algo = "eclat"
            else:
                algo = "fpgrowth"

        # Execute mining
        if algo == "apriori":
            self.frequent_itemsets_ = apriori(
                X_matrix,
                min_support=self.min_support,
                max_len=self.max_len,
                max_candidates=self.max_candidates,
                sample_weight=sample_weight,
                pruning_policy=self.pruning_policy_,
                telemetry=self.telemetry_,
            )
        elif algo == "fpgrowth":
            self.frequent_itemsets_ = fpgrowth(
                X_matrix,
                min_support=self.min_support,
                max_len=self.max_len,
                max_candidates=self.max_candidates,
                sample_weight=sample_weight,
                pruning_policy=self.pruning_policy_,
                telemetry=self.telemetry_,
            )
        elif algo == "eclat":
            self.frequent_itemsets_ = eclat(
                X_matrix,
                min_support=self.min_support,
                max_len=self.max_len,
                max_candidates=self.max_candidates,
                sample_weight=sample_weight,
                pruning_policy=self.pruning_policy_,
                telemetry=self.telemetry_,
            )

        self.is_degraded_ = self.pruning_policy_.is_degraded

        # Generate association rules
        self.rules_ = association_rules(
            self.frequent_itemsets_,
            metric=self.metric,
            min_threshold=self.min_threshold,
            feature_names=self.columns_,
        )

        return self

    def predict(self, X: Union[Iterable[Iterable[Any]], np.ndarray, sparse.csr_matrix]) -> List[List[str]]:
        """Predict recommended consequent items for each transaction.

        For each transaction, finds rules where antecedent is a subset of the transaction,
        and returns the consequents ranked by the rule metric.

        Parameters
        ----------
        X : transactions or binary indicator matrix

        Returns
        -------
        recommendations : list of list of str
            Recommended items per transaction.
        """
        check_is_fitted(self, attributes=["rules_", "columns_"])

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        # Materialize one-shot iterables (e.g. generators) so the empty-rules
        # path and the transaction loop both see the same transactions.
        if not isinstance(X, (np.ndarray, sparse.spmatrix)):
            X = list(X)

        if self.rules_.empty:
            return [[] for _ in range(len(X))]

        # Prepare transactions as set of indices
        if self.encoder_ is not None and not isinstance(X, (np.ndarray, sparse.spmatrix)):
            tx_sets = [
                {self.encoder_.item_to_idx_[item] for item in tx if item in self.encoder_.item_to_idx_}
                for tx in X
            ]
        elif isinstance(X, sparse.spmatrix):
            X_csr = X.tocsr()
            tx_sets = [
                set(X_csr.indices[X_csr.indptr[i]: X_csr.indptr[i + 1]])
                for i in range(X_csr.shape[0])
            ]
        else:
            X_arr = np.asarray(X, dtype=bool)
            tx_sets = [
                set(np.where(row)[0])
                for row in X_arr
            ]

        results = []
        # Pre-extract rules as tuples of (antecedent_set, consequent_indices)
        rule_tuples = [
            (row["antecedent"], row["consequent"])
            for _, row in self.rules_.iterrows()
        ]

        for tx in tx_sets:
            recs = []
            seen_items = set()
            for ante, cons in rule_tuples:
                if ante.issubset(tx):
                    for item_idx in cons:
                        if item_idx not in tx and item_idx not in seen_items:
                            seen_items.add(item_idx)
                            recs.append(self.columns_[item_idx])
            results.append(recs)

        return results

    def transform(self, X: Union[Iterable[Iterable[Any]], np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """Transform transactions into binary vector of triggered rule firings."""
        check_is_fitted(self, attributes=["rules_"])
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if not isinstance(X, (np.ndarray, sparse.spmatrix)):
            X = list(X)
        n_rules = len(self.rules_)
        if n_rules == 0:
            n_samples = len(X)
            return np.zeros((n_samples, 0), dtype=bool)

        # Pre-extract antecedents
        rule_antes = list(self.rules_["antecedent"])

        if self.encoder_ is not None and not isinstance(X, (np.ndarray, sparse.spmatrix)):
            tx_sets = [
                {self.encoder_.item_to_idx_[item] for item in tx if item in self.encoder_.item_to_idx_}
                for tx in X
            ]
        elif isinstance(X, sparse.spmatrix):
            X_csr = X.tocsr()
            tx_sets = [
                set(X_csr.indices[X_csr.indptr[i]: X_csr.indptr[i + 1]])
                for i in range(X_csr.shape[0])
            ]
        else:
            X_arr = np.asarray(X, dtype=bool)
            tx_sets = [set(np.where(row)[0]) for row in X_arr]

        n_samples = len(tx_sets)
        output = np.zeros((n_samples, n_rules), dtype=bool)
        for i, tx in enumerate(tx_sets):
            for r_idx, ante in enumerate(rule_antes):
                if ante.issubset(tx):
                    output[i, r_idx] = True

        return output

    def get_frequent_itemsets(self, use_names: bool = True) -> pd.DataFrame:
        """Return frequent itemsets as a formatted DataFrame."""
        check_is_fitted(self, attributes=["frequent_itemsets_", "columns_"])
        records = []
        for itemset, supp in self.frequent_itemsets_.items():
            if use_names:
                items_repr = frozenset(self.columns_[i] for i in itemset)
            else:
                items_repr = itemset
            records.append({
                "itemsets": items_repr,
                "length": len(itemset),
                "support": supp,
            })
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values(by=["length", "support"], ascending=[True, False], kind="stable")
            df = df.reset_index(drop=True)
        else:
            df = pd.DataFrame(columns=["itemsets", "length", "support"])
        return df

    def get_rules(self) -> pd.DataFrame:
        """Return DataFrame of discovered association rules."""
        check_is_fitted(self, attributes=["rules_"])
        return self.rules_.copy()

    def explore(self):
        """Return an interactive RuleExplorer instance for the fitted rules."""
        check_is_fitted(self, attributes=["rules_"])
        from skarm.explorer import RuleExplorer
        return RuleExplorer(self.rules_)
