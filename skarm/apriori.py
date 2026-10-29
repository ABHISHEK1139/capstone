"""Conventional Apriori algorithm implementation for frequent itemset mining."""

import time
from typing import Dict, List, Tuple, Set, FrozenSet, Optional, Union
import numpy as np
from scipy import sparse

from skarm.pruning import PruningPolicy, CandidateExplosionError
from skarm.telemetry import TelemetryLogger


def apriori(
    X: Union[np.ndarray, sparse.csr_matrix],
    min_support: float = 0.1,
    max_len: Optional[int] = None,
    max_candidates: int = 100000,
    sample_weight: Optional[np.ndarray] = None,
    pruning_policy: Optional[PruningPolicy] = None,
    telemetry: Optional[TelemetryLogger] = None,
) -> Dict[FrozenSet[int], float]:
    """Find all frequent itemsets using the conventional Apriori algorithm.

    Parameters
    ----------
    X : np.ndarray or scipy.sparse.csr_matrix of shape (n_samples, n_features)
        Binary transaction matrix where X[i, j] is 1/True if transaction i contains item j.
    min_support : float, default=0.1
        Minimum support threshold in [0.0, 1.0].
    max_len : int or None, default=None
        Maximum length of itemsets to mine.
    max_candidates : int, default=100000
        Maximum candidate limit at any level to prevent exponential explosion.
    sample_weight : np.ndarray of shape (n_samples,) or None
        Optional transaction weights (NT-3).
    pruning_policy : PruningPolicy or None
        Custom pruning policy manager.
    telemetry : TelemetryLogger or None
        Telemetry collector for profiling runtime and memory.

    Returns
    -------
    frequent_itemsets : dict
        Mapping from frozenset of item indices to float support value.
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

    is_sparse = sparse.issparse(X)
    if is_sparse:
        X_csr = X.tocsr()
        # Compute 1-itemset support
        if sample_weight is not None:
            # Weighted column sums
            # X_csr is (n_samples, n_features)
            weighted_counts = np.zeros(n_features, dtype=float)
            for i in range(n_samples):
                cols = X_csr.indices[X_csr.indptr[i]: X_csr.indptr[i + 1]]
                weighted_counts[cols] += sample_weight[i]
            col_supports = weighted_counts / total_weight
        else:
            col_supports = np.array(X_csr.sum(axis=0)).flatten() / total_weight
    else:
        X_arr = np.asarray(X, dtype=bool)
        if sample_weight is not None:
            col_supports = np.average(X_arr, axis=0, weights=sample_weight)
        else:
            col_supports = np.mean(X_arr, axis=0)

    # Level 1 frequent itemsets
    level_start = time.perf_counter()
    L1: Dict[Tuple[int], float] = {}
    for j in range(n_features):
        supp = float(col_supports[j])
        if supp >= min_support:
            L1[(j,)] = supp

    frequent_itemsets: Dict[FrozenSet[int], float] = {
        frozenset(k): v for k, v in L1.items()
    }

    if telemetry is not None:
        telemetry.record_level(
            level=1,
            candidate_count=n_features,
            frequent_count=len(L1),
            elapsed_level=time.perf_counter() - level_start,
        )

    current_L = sorted(L1.keys())
    k = 2

    # Pre-extract transaction sets for fast subset testing if sparse or dense
    if is_sparse:
        # Precompute row item sets for fast checking
        row_itemsets = [
            set(X_csr.indices[X_csr.indptr[i]: X_csr.indptr[i + 1]])
            for i in range(n_samples)
        ]
    else:
        row_itemsets = None

    while current_L and pruning_policy.should_continue_length(k - 1):
        level_start = time.perf_counter()
        # Generate candidates C_k from L_{k-1}
        candidates: List[Tuple[int, ...]] = []
        L_set = set(current_L)
        n_L = len(current_L)

        for i in range(n_L):
            for j in range(i + 1, n_L):
                l1 = current_L[i]
                l2 = current_L[j]
                # Join condition: share first k-2 elements
                if l1[:-1] == l2[:-1]:
                    c = l1 + (l2[-1],)
                    # Apriori pruning: all (k-1) subsets must be in L_{k-1}
                    is_valid = True
                    for drop_idx in range(len(c) - 2):  # last two are already checked by construction
                        sub = c[:drop_idx] + c[drop_idx + 1:]
                        if sub not in L_set:
                            is_valid = False
                            break
                    if is_valid:
                        candidates.append(c)
                else:
                    break

        candidate_count = len(candidates)
        can_proceed = pruning_policy.check_candidate_count(candidate_count, level=k)
        if not can_proceed:
            # Degraded mode triggered (NT-1, NT-6)
            break

        if not candidates:
            break

        # Truncate candidates if truncate policy
        if candidate_count > pruning_policy.max_candidates and pruning_policy.on_candidate_explosion == "truncate":
            candidates = candidates[: pruning_policy.max_candidates]

        # Count support for each candidate
        next_L: Dict[Tuple[int, ...], float] = {}
        for c in candidates:
            c_set = set(c)
            if is_sparse:
                if sample_weight is not None:
                    matched_weight = sum(
                        sample_weight[row_idx]
                        for row_idx, r_set in enumerate(row_itemsets)
                        if c_set.issubset(r_set)
                    )
                    supp = matched_weight / total_weight
                else:
                    matched = sum(
                        1 for r_set in row_itemsets if c_set.issubset(r_set)
                    )
                    supp = matched / total_weight
            else:
                cols = list(c)
                row_match = np.all(X_arr[:, cols], axis=1)
                if sample_weight is not None:
                    supp = float(np.sum(row_match * sample_weight) / total_weight)
                else:
                    supp = float(np.count_nonzero(row_match) / total_weight)

            if supp >= min_support:
                next_L[c] = supp
                frequent_itemsets[frozenset(c)] = supp

        if telemetry is not None:
            telemetry.record_level(
                level=k,
                candidate_count=candidate_count,
                frequent_count=len(next_L),
                elapsed_level=time.perf_counter() - level_start,
            )

        current_L = sorted(next_L.keys())
        k += 1

    if telemetry is not None:
        telemetry.stop({
            "algorithm": "apriori",
            "total_frequent_itemsets": len(frequent_itemsets),
            "is_degraded": pruning_policy.is_degraded,
            "peak_candidates": pruning_policy.peak_candidate_count,
        })

    return frequent_itemsets
