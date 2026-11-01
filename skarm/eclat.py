"""Eclat (Equivalence Class Clustering and bottom-up Lattice Traversal) vertical miner."""

import time
from typing import Dict, List, Tuple, Optional, Union, FrozenSet, Set
import numpy as np
from scipy import sparse

from skarm.pruning import PruningPolicy
from skarm.telemetry import TelemetryLogger


def _validate_matrix(X: Union[np.ndarray, sparse.spmatrix]):
    """Validate that input matrix values are strictly binary {0, 1} or boolean."""
    if sparse.issparse(X):
        if X.data.dtype.kind in ('f', 'c'):
            if np.isnan(X.data).any():
                raise ValueError("Input matrix contains NaNs.")
            if np.isinf(X.data).any():
                raise ValueError("Input matrix contains infinite values.")
        if X.data.dtype != bool:
            if not np.all((X.data == 0) | (X.data == 1)):
                raise ValueError("Input matrix values must be strictly binary (0, 1, or boolean). Continuous, negative, or non-binary values are not allowed.")
    else:
        arr = np.asarray(X)
        if arr.dtype.kind in ('f', 'c'):
            if np.isnan(arr).any():
                raise ValueError("Input matrix contains NaNs.")
            if np.isinf(arr).any():
                raise ValueError("Input matrix contains infinite values.")
        if arr.dtype != bool:
            if not np.all((arr == 0) | (arr == 1)):
                raise ValueError("Input matrix values must be strictly binary (0, 1, or boolean). Continuous, negative, or non-binary values are not allowed.")


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
    _validate_matrix(X)

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

    # Weight validation (NT-3, B10)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if len(sample_weight) != n_samples:
            raise ValueError(f"sample_weight length {len(sample_weight)} != n_samples {n_samples}")
        if np.isnan(sample_weight).any():
            raise ValueError("sample_weight cannot contain NaN values.")
        if np.isinf(sample_weight).any():
            raise ValueError("sample_weight cannot contain infinite values.")
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

    total_start = time.perf_counter()
    counters = {
        "intersections_performed": 0,
        "tid_elements_scanned": 0,
    }

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
                counters["intersections_performed"] += 1
                counters["tid_elements_scanned"] += len(tids_i) + len(tids_j)

                if inter_weight_check := (sum(sample_weight[t] for t in inter_tids) if sample_weight is not None else len(inter_tids)):
                    pass
                inter_weight = inter_weight_check

                if inter_weight >= min_weight - 1e-12:
                    supp = float(inter_weight / total_weight)
                    pair_itemset = frozenset(new_prefix + (item_j,))
                    frequent_itemsets[pair_itemset] = supp
                    elapsed = time.perf_counter() - total_start

                    # Guardrail check
                    can_proceed = pruning_policy.check_budget(
                        candidate_count=len(frequent_itemsets),
                        level=len(pair_itemset),
                        elapsed_time=elapsed,
                    )
                    if not can_proceed:
                        return

                    if pruning_policy.on_candidate_explosion == "truncate" and len(frequent_itemsets) >= pruning_policy.max_candidates:
                        pruning_policy.is_degraded = True
                        pruning_policy.future_candidates_status = "UNKNOWN"
                        pruning_policy.degradation_reason = "max_candidates_reached"
                        return

                    new_items_with_tids.append((item_j, inter_tids))

            # extend(new_prefix) records itemsets of length len(new_prefix) + 2,
            # so it may only run when that length is within max_len. The old
            # gate (len(new_prefix)) was off by one and mined one level too deep.
            if new_items_with_tids and pruning_policy.should_continue_length(len(new_prefix) + 1):
                extend(new_prefix, new_items_with_tids)
                if pruning_policy.is_degraded:
                    return

    init_pairs = [(item, tid_lists[item]) for item in initial_items]
    if pruning_policy.should_continue_length(1):
        extend((), init_pairs)

    counters["itemsets_discovered"] = len(frequent_itemsets)
    counters["candidates_generated"] = len(frequent_itemsets)

    pruning_policy.counters = counters
    if telemetry is not None:
        telemetry.counters = counters
        telemetry.stop({
            "algorithm": "eclat",
            "total_frequent_itemsets": len(frequent_itemsets),
            "is_degraded": pruning_policy.is_degraded,
            "peak_candidates": pruning_policy.peak_candidate_count,
            "counters": counters,
        })

    return frequent_itemsets
