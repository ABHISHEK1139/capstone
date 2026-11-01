"""Scikit-learn compatible Association Rule Mining Estimator."""

import time
from typing import Optional, Union, Iterable, List, Dict, Any, FrozenSet, Set, Sequence
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from skarm.preprocessing import TransactionEncoder
from skarm.apriori import apriori
from skarm.fpgrowth import fpgrowth
from skarm.eclat import eclat
from skarm.rules import association_rules
from skarm.pruning import PruningPolicy, BoundaryConditionError, get_current_memory_mb
from skarm.telemetry import TelemetryLogger


class AssociationRulesEstimator(BaseEstimator, TransformerMixin):
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
        max_time_seconds: Optional[float] = None,
        max_memory_mb: Optional[float] = None,
        scoring: str = "mean_confidence",
    ):
        self.min_support = min_support
        self.metric = metric
        self.min_threshold = min_threshold
        self.algorithm = algorithm
        self.max_len = max_len
        self.max_candidates = max_candidates
        self.on_candidate_explosion = on_candidate_explosion
        self.sparse_output = sparse_output
        self.max_time_seconds = max_time_seconds
        self.max_memory_mb = max_memory_mb
        self.scoring = scoring

    def _validate_parameters(self):
        """Validate constructor parameters against boundary conditions (NT-4)."""
        if not (0.0 <= self.min_support <= 1.0):
            raise BoundaryConditionError(f"min_support must be between 0.0 and 1.0, got {self.min_support}")
        if self.max_candidates <= 0:
            raise BoundaryConditionError(f"max_candidates must be > 0, got {self.max_candidates}")
        if self.max_len is not None and self.max_len < 1:
            raise BoundaryConditionError(f"max_len must be >= 1, got {self.max_len}")
        if self.max_time_seconds is not None and self.max_time_seconds <= 0:
            raise BoundaryConditionError(f"max_time_seconds must be positive, got {self.max_time_seconds}")
        if self.max_memory_mb is not None and self.max_memory_mb <= 0:
            raise BoundaryConditionError(f"max_memory_mb must be positive, got {self.max_memory_mb}")
        if self.algorithm not in ("auto", "apriori", "fpgrowth", "eclat"):
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        valid_metrics = {"confidence", "lift", "support", "leverage", "conviction", "zhangs_metric"}
        if self.metric not in valid_metrics:
            raise ValueError(f"Unknown metric '{self.metric}'. Choose from {valid_metrics}")
        valid_scorings = {"mean_confidence", "mean_lift", "total_support", "rule_count"}
        if self.scoring not in valid_scorings and not callable(self.scoring):
            raise ValueError(f"Invalid scoring metric: {self.scoring}")

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
        fit_start_time = time.perf_counter()
        self._validate_parameters()

        # Handle Pandas DataFrame: preserve column names (Bug B4)
        if isinstance(X, pd.DataFrame):
            df_columns = list(X.columns)
            X_matrix = X.to_numpy()
            self.columns_ = df_columns
            self.feature_names_in_ = np.array(df_columns, dtype=object)
            self.n_features_in_ = len(df_columns)
            self.encoder_ = None
        elif isinstance(X, (np.ndarray, sparse.spmatrix)):
            X_matrix = X
            self.encoder_ = None
            self.columns_ = [f"item_{i}" for i in range(X_matrix.shape[1])]
            self.n_features_in_ = X_matrix.shape[1]
            self.feature_names_in_ = np.array(self.columns_, dtype=object)
        else:
            # Transaction list of lists
            self.encoder_ = TransactionEncoder(sparse_output=self.sparse_output)
            X_matrix = self.encoder_.fit_transform(X)
            self.columns_ = list(self.encoder_.columns_)
            self.n_features_in_ = len(self.columns_)
            self.feature_names_in_ = np.array(self.columns_, dtype=object)

        # Setup pruning policy and telemetry
        self.pruning_policy_ = PruningPolicy(
            max_candidates=self.max_candidates,
            max_len=self.max_len,
            min_support=self.min_support,
            on_candidate_explosion=self.on_candidate_explosion,
            max_time_seconds=self.max_time_seconds,
            max_memory_mb=self.max_memory_mb,
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
        self.status_ = "DEGRADED" if self.is_degraded_ else "EXACT"
        self.is_complete_ = (self.status_ == "EXACT")
        self.termination_reason_ = self.pruning_policy_.degradation_reason if self.is_degraded_ else None

        # Generate association rules
        self.rules_ = association_rules(
            self.frequent_itemsets_,
            metric=self.metric,
            min_threshold=self.min_threshold,
            feature_names=self.columns_,
        )

        # Standard fitted attribute itemsets_ as a DataFrame (retaining frequent_itemsets_ as alias)
        self.itemsets_ = self.get_frequent_itemsets(use_names=True)

        total_fit_time = time.perf_counter() - fit_start_time
        peak_rss = get_current_memory_mb()

        # Structured metadata dictionary (R2)
        counters = getattr(self.pruning_policy_, "counters", {})
        if not counters and hasattr(self.telemetry_, "counters"):
            counters = self.telemetry_.counters

        self.metadata_ = {
            "status": self.status_,
            "is_complete": self.is_complete_,
            "termination_reason": self.termination_reason_,
            "level_halted": self.pruning_policy_.level_halted,
            "candidates_generated_at_level": self.pruning_policy_.candidates_generated_at_level,
            "candidates_remaining_at_level": self.pruning_policy_.candidates_remaining_at_level,
            "future_candidates_status": self.pruning_policy_.future_candidates_status,
            "frequent_itemsets_discovered": len(self.frequent_itemsets_),
            "is_candidate_count_exact": (self.status_ == "EXACT"),
            "counters": dict(counters),
            "operations": dict(counters),
            "resource_usage": {
                "wall_time_seconds": round(total_fit_time, 4),
                "peak_rss_mb": round(peak_rss, 2),
            },
        }

        return self

    def _to_tx_sets(self, X: Any) -> List[Set[int]]:
        """Convert query transactions or binary matrix into list of item index sets (Bug B5)."""
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        if isinstance(X, sparse.spmatrix):
            X_csr = X.tocsr()
            return [
                set(X_csr.indices[X_csr.indptr[i]: X_csr.indptr[i + 1]])
                for i in range(X_csr.shape[0])
            ]

        if isinstance(X, np.ndarray):
            X_arr = np.asarray(X, dtype=bool)
            return [set(np.where(row)[0]) for row in X_arr]

        # Iterable of transactions (e.g. list of lists)
        tx_list = list(X)
        col_to_idx = {col: idx for idx, col in enumerate(self.columns_)}
        tx_sets = []
        for tx in tx_list:
            if isinstance(tx, (str, bytes)):
                tx = [tx]
            tx_indices = set()
            for item in tx:
                if self.encoder_ is not None and item in self.encoder_.item_to_idx_:
                    tx_indices.add(self.encoder_.item_to_idx_[item])
                elif item in col_to_idx:
                    tx_indices.add(col_to_idx[item])
                elif isinstance(item, (int, np.integer)) and 0 <= item < len(self.columns_):
                    tx_indices.add(int(item))
            tx_sets.append(tx_indices)
        return tx_sets

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
        tx_sets = self._to_tx_sets(X)

        if self.rules_.empty:
            return [[] for _ in range(len(tx_sets))]

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
        tx_sets = self._to_tx_sets(X)
        n_samples = len(tx_sets)
        n_rules = len(self.rules_)
        if n_rules == 0:
            return np.zeros((n_samples, 0), dtype=bool)

        # Pre-extract antecedents
        rule_antes = list(self.rules_["antecedent"])

        output = np.zeros((n_samples, n_rules), dtype=bool)
        for i, tx in enumerate(tx_sets):
            for r_idx, ante in enumerate(rule_antes):
                if ante.issubset(tx):
                    output[i, r_idx] = True

        return output

    def get_feature_names_out(self, input_features: Optional[Any] = None) -> np.ndarray:
        """Get output feature names for transformation (Bug B8).

        Returns array of rule representations in the format:
        '{antecedent} -> {consequent}'
        """
        check_is_fitted(self, attributes=["rules_"])
        if self.rules_.empty:
            return np.array([], dtype=object)

        if input_features is not None:
            if hasattr(self, "n_features_in_") and len(input_features) != self.n_features_in_:
                raise ValueError(
                    f"input_features has length {len(input_features)}, but "
                    f"{self.n_features_in_} features were seen during fit."
                )
            feature_names_map = list(input_features)
        elif hasattr(self, "columns_") and self.columns_:
            feature_names_map = self.columns_
        else:
            feature_names_map = [f"item_{i}" for i in range(getattr(self, "n_features_in_", 0))]

        feature_names = []
        for _, row in self.rules_.iterrows():
            ante_items = [
                feature_names_map[i] if i < len(feature_names_map) else str(i)
                for i in row["antecedent"]
            ]
            cons_items = [
                feature_names_map[i] if i < len(feature_names_map) else str(i)
                for i in row["consequent"]
            ]
            ante_str = "{" + ", ".join(sorted(str(x) for x in ante_items)) + "}"
            cons_str = "{" + ", ".join(sorted(str(x) for x in cons_items)) + "}"
            feature_names.append(f"{ante_str} -> {cons_str}")

        return np.array(feature_names, dtype=object)

    def score(self, X: Any = None, y: Optional[Any] = None) -> float:
        """Calculate unsupervised model score for hyperparameter tuning.

        When validation data X is provided, computes held-out metrics on X.
        """
        check_is_fitted(self, attributes=["rules_"])
        if self.rules_.empty:
            return 0.0
        scoring = getattr(self, "scoring", "mean_confidence")
        if callable(scoring):
            return float(scoring(self, X, y))
        elif scoring == "mean_lift":
            from skarm.metrics import mean_lift_scorer
            return float(mean_lift_scorer(self, X, y))
        elif scoring == "total_support":
            from skarm.metrics import total_support_scorer
            return float(total_support_scorer(self, X, y))
        elif scoring == "rule_count":
            from skarm.metrics import rule_count_scorer
            return float(rule_count_scorer(self, X, y))
        else:
            from skarm.metrics import mean_confidence_scorer
            return float(mean_confidence_scorer(self, X, y))

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

    def __sklearn_tags__(self):
        """Declare scikit-learn estimator tags for interoperability (R3)."""
        tags = super().__sklearn_tags__()
        tags.target_tags.required = False
        tags.input_tags.sparse = True
        tags.input_tags.two_d_array = True
        try:
            from sklearn.utils._tags import TransformerTags
            tags.transformer_tags = TransformerTags(preserves_dtype=[])
        except Exception:
            pass
        return tags
