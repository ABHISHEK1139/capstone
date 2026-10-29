"""Eclat (Equivalence Class Clustering and bottom-up Lattice Traversal) vertical miner."""

import time
from typing import Dict, List, Tuple, Optional, Union, FrozenSet, Set
import numpy as np
from scipy import sparse

from skarm.pruning import PruningPolicy
from skarm.telemetry import TelemetryLogger


def eclat(
    X: Union[np.ndarray, sparse.csr_matrix],
    min_support: float = 0.1,
    max_len: Optional[int] = None,
    max_candidates: int = 100000,
    sample_weight: Optional[np.ndarray] = None,
    pruning_policy: Optional[PruningPolicy] = None,
    telemetry: Optional[TelemetryLogger] = None,
) -> Dict[FrozenSet[int], float]:
    """Mine frequent itemsets using the vertical format Eclat algorithm.

    Uses TID-list (Transaction ID) intersections for high-efficiency support counting.

    Parameters
    ----------
    X : np.ndarray or scipy.sparse.csr_matrix
        Binary transaction matrix.
    min_support : float, default=0.1
        Minimum support threshold.
    max_len : int or None, default=None
        Maximum length of itemsets.
    max_candidates : int, default=100000
        Maximum candidate limit.
    sample_weight : np.ndarray of shape (n_samples,) or None
        Optional transaction sample weights.
    pruning_policy : PruningPolicy or None
        Custom pruning policy manager.
    telemetry : TelemetryLogger or None
        Telemetry collector for profiling.

    Returns
    -------
    frequent_itemsets : dict
        Mapping from frozenset of item indices to support value.
    """
    if pruning_policy is None:
        pruning_policy = PruningPolicy(
            max_candidates=max_candidates,
            max_len=max_len,
            min_support=min_support,
        )

    if telemetry is not None:
        telemetry.start()

    n_samples, n_features = X.shape
    if n_samples == 0 or n_features == 0:
        return {}

    # Weight validation (NT-3)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if len(sample_weight) != n_samples:
            raise ValueError(f"sample_weight length {len(sample_weight)} != n_samples {n_samples}")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative values.")
        total_weight = float(np.sum(sample_weight))
        if total_weight <= 0.0:
            raise ValueError("Sum of sample_weights must be strictly positive.")
    else:
        total_weight = float(n_samples)

    min_weight = min_support * total_weight

    # Build vertical TID lists: item -> set of transaction IDs
    tid_lists: Dict[int, Set[int]] = {}
    is_sparse = sparse.issparse(X)

    if is_sparse:
        X_csc = X.tocsc()
        for j in range(n_features):
            tids = set(X_csc.indices[X_csc.indptr[j]: X_csc.indptr[j + 1]])
            if sample_weight is not None:
                w_sum = sum(sample_weight[t] for t in tids)
            else:
                w_sum = len(tids)
            if w_sum >= min_weight - 1e-12:
                tid_lists[j] = tids
    else:
        X_arr = np.asarray(X, dtype=bool)
        for j in range(n_features):
            tids = set(np.where(X_arr[:, j])[0])
            if sample_weight is not None:
                w_sum = sum(sample_weight[t] for t in tids)
            else:
                w_sum = len(tids)
            if w_sum >= min_weight - 1e-12:
                tid_lists[j] = tids

    frequent_itemsets: Dict[FrozenSet[int], float] = {}

    def get_support(tids: Set[int]) -> float:
        if sample_weight is not None:
            return float(sum(sample_weight[t] for t in tids) / total_weight)
        return float(len(tids) / total_weight)

    # Initial frequent 1-itemsets
    initial_items = sorted(tid_lists.keys())
    for item in initial_items:
        frequent_itemsets[frozenset([item])] = get_support(tid_lists[item])

    # Recursive equivalence class expansion
    def extend(prefix: Tuple[int, ...], items_with_tids: List[Tuple[int, Set[int]]]):
        n_items = len(items_with_tids)
        for i in range(n_items):
            item_i, tids_i = items_with_tids[i]
            new_prefix = prefix + (item_i,)
            new_items_with_tids: List[Tuple[int, Set[int]]] = []

            for j in range(i + 1, n_items):
                item_j, tids_j = items_with_tids[j]
                inter_tids = tids_i.intersection(tids_j)

                if sample_weight is not None:
                    inter_weight = sum(sample_weight[t] for t in inter_tids)
                else:
                    inter_weight = len(inter_tids)

                if inter_weight >= min_weight - 1e-12:
                    supp = float(inter_weight / total_weight)
                    pair_itemset = frozenset(new_prefix + (item_j,))
                    frequent_itemsets[pair_itemset] = supp

                    # Guardrail check
                    if not pruning_policy.check_candidate_count(len(frequent_itemsets), level=len(pair_itemset)):
                        return

                    new_items_with_tids.append((item_j, inter_tids))

            if new_items_with_tids and pruning_policy.should_continue_length(len(new_prefix)):
                extend(new_prefix, new_items_with_tids)
                if pruning_policy.is_degraded:
                    return

    init_pairs = [(item, tid_lists[item]) for item in initial_items]
    if pruning_policy.should_continue_length(1):
        extend((), init_pairs)

    if telemetry is not None:
        telemetry.stop({
            "algorithm": "eclat",
            "total_frequent_itemsets": len(frequent_itemsets),
            "is_degraded": pruning_policy.is_degraded,
            "peak_candidates": pruning_policy.peak_candidate_count,
        })

    return frequent_itemsets
