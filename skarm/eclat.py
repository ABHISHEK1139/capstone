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

    Uses vertical TID-bitset intersections (arbitrary-precision C bitvectors
    with hardware popcount) for unweighted workloads and vertical TID-set
    intersections for weighted or ultra-large sparse workloads.

    Parameters
    ----------
    X : np.ndarray or scipy.sparse.csr_matrix
        Binary transaction matrix.
    min_support : float, default=0.1
        Minimum support threshold.
    max_len : int or None, default=None
        Maximum length of itemsets.
    max_candidates : int, default=100000
        Upper bound on cumulative discovered frequent itemsets |F| from vertical
        TID-list intersections. In Eclat, equivalence classes are expanded recursively
        via pairwise TID-list intersections (T(i) ∩ T(j)). If |F| >= max_candidates,
        the configured candidate explosion policy ('raise', 'truncate', or 'degrade')
        is triggered.
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
    is_sparse = sparse.issparse(X)
    total_start = getattr(pruning_policy, "start_time", time.perf_counter())
    check_time_mem = (
        pruning_policy.max_time_seconds is not None
        or pruning_policy.max_memory_mb is not None
    )

    # Use native C bitset intersections for unweighted dense or moderate-size sparse matrices
    use_bitsets = (sample_weight is None) and (not is_sparse or n_samples <= 150_000)

    if use_bitsets:
        if is_sparse:
            X_csc = X.tocsc()
            if not X_csc.has_canonical_format:
                X_csc.sum_duplicates()
            X_csc.eliminate_zeros()
            col_counts = np.diff(X_csc.indptr)
            initial_items = [
                int(j) for j in np.flatnonzero(col_counts >= min_weight - 1e-12)
            ]
            if check_time_mem:
                elapsed = time.perf_counter() - total_start
                if not pruning_policy.check_budget(candidate_count=0, level=1, elapsed_time=elapsed):
                    return {}
            buf = np.zeros(n_samples, dtype=bool)
            init_bit_pairs: List[Tuple[int, int, int]] = []
            indptr = X_csc.indptr
            indices = X_csc.indices
            for idx_item, j in enumerate(initial_items):
                if check_time_mem and idx_item % 500 == 0:
                    elapsed = time.perf_counter() - total_start
                    if not pruning_policy.check_budget(candidate_count=0, level=1, elapsed_time=elapsed):
                        return {}
                rows = indices[indptr[j]: indptr[j + 1]]
                if rows.size:
                    buf[rows] = True
                    bits = int.from_bytes(np.packbits(buf, bitorder="little").tobytes(), "little")
                    buf[rows] = False
                else:
                    bits = 0
                init_bit_pairs.append((j, bits, int(col_counts[j])))
        else:
            X_arr = np.asarray(X, dtype=bool)
            col_counts = np.count_nonzero(X_arr, axis=0)
            initial_items = [
                int(j) for j in np.flatnonzero(col_counts >= min_weight - 1e-12)
            ]
            if check_time_mem:
                elapsed = time.perf_counter() - total_start
                if not pruning_policy.check_budget(candidate_count=0, level=1, elapsed_time=elapsed):
                    return {}
            init_bit_pairs = []
            if initial_items:
                packed = np.packbits(X_arr[:, initial_items].T, axis=1, bitorder="little")
                for idx_item, j in enumerate(initial_items):
                    bits = int.from_bytes(packed[idx_item].tobytes(), "little")
                    init_bit_pairs.append((j, bits, int(col_counts[j])))

        counters = {
            "intersections_performed": 0,
            "tid_elements_scanned": 0,
            "candidates_generated": len(initial_items),
            "support_evaluations": len(initial_items),
            "itemsets_discovered": 0,
        }
        frequent_itemsets: Dict[FrozenSet[int], float] = {}

        if check_time_mem:
            elapsed_init = time.perf_counter() - total_start
            if not pruning_policy.check_budget(candidate_count=0, level=1, elapsed_time=elapsed_init):
                pruning_policy.counters = counters
                return frequent_itemsets

        max_cands = pruning_policy.max_candidates

        for item, _, cnt in init_bit_pairs:
            if len(frequent_itemsets) >= max_cands:
                elapsed = time.perf_counter() - total_start
                can_proceed = pruning_policy.check_budget(
                    candidate_count=len(frequent_itemsets) + 1,
                    level=1,
                    elapsed_time=elapsed,
                )
                if not can_proceed:
                    break
                if pruning_policy.on_candidate_explosion == "truncate":
                    break
            frequent_itemsets[frozenset((item,))] = cnt / total_weight

        def extend_bits(prefix: Tuple[int, ...], items_with_bits: List[Tuple[int, int, int]]):
            n_items = len(items_with_bits)
            next_level = len(prefix) + 2
            can_recurse_next = pruning_policy.should_continue_length(next_level)
            for i in range(n_items):
                if pruning_policy.is_degraded:
                    return
                item_i, bits_i, cnt_i = items_with_bits[i]
                new_prefix = prefix + (item_i,)
                new_items: List[Tuple[int, int, int]] = []

                for j in range(i + 1, n_items):
                    if pruning_policy.is_degraded:
                        return
                    item_j, bits_j, cnt_j = items_with_bits[j]
                    inter_bits = bits_i & bits_j
                    inter_cnt = inter_bits.bit_count()
                    counters["intersections_performed"] += 1
                    counters["tid_elements_scanned"] += cnt_i + cnt_j
                    counters["candidates_generated"] += 1
                    counters["support_evaluations"] += 1

                    if check_time_mem and counters["intersections_performed"] % 100 == 0:
                        elapsed = time.perf_counter() - total_start
                        if not pruning_policy.check_budget(
                            candidate_count=0,
                            level=next_level,
                            elapsed_time=elapsed,
                        ):
                            return

                    if inter_cnt >= min_weight - 1e-12:
                        if len(frequent_itemsets) >= max_cands:
                            elapsed = time.perf_counter() - total_start
                            can_proceed = pruning_policy.check_budget(
                                candidate_count=len(frequent_itemsets) + 1,
                                level=next_level,
                                elapsed_time=elapsed,
                            )
                            if not can_proceed:
                                return
                            if pruning_policy.on_candidate_explosion == "truncate":
                                return

                        pair_itemset = frozenset(new_prefix + (item_j,))
                        frequent_itemsets[pair_itemset] = inter_cnt / total_weight

                        if check_time_mem:
                            elapsed = time.perf_counter() - total_start
                            if not pruning_policy.check_budget(
                                candidate_count=0,
                                level=next_level,
                                elapsed_time=elapsed,
                            ):
                                return

                        if can_recurse_next:
                            new_items.append((item_j, inter_bits, inter_cnt))

                if new_items and pruning_policy.should_continue_length(next_level):
                    extend_bits(new_prefix, new_items)
                    if pruning_policy.is_degraded:
                        return

        if init_bit_pairs and pruning_policy.should_continue_length(1):
            extend_bits((), init_bit_pairs)

        counters["itemsets_discovered"] = len(frequent_itemsets)
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

    # Fallback path: Python set TID-lists (for weighted transactions or huge ultra-sparse matrices)
    tid_lists: Dict[int, Set[int]] = {}
    if is_sparse:
        X_csc = X.tocsc()
        if not X_csc.has_canonical_format:
            X_csc.sum_duplicates()
        X_csc.eliminate_zeros()
        for j in range(n_features):
            if j % 1000 == 0 and check_time_mem:
                elapsed = time.perf_counter() - total_start
                if not pruning_policy.check_budget(candidate_count=0, level=1, elapsed_time=elapsed):
                    return {}
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
            if j % 1000 == 0 and check_time_mem:
                elapsed = time.perf_counter() - total_start
                if not pruning_policy.check_budget(candidate_count=0, level=1, elapsed_time=elapsed):
                    return {}
            tids = set(np.where(X_arr[:, j])[0])
            if sample_weight is not None:
                w_sum = sum(sample_weight[t] for t in tids)
            else:
                w_sum = len(tids)
            if w_sum >= min_weight - 1e-12:
                tid_lists[j] = tids

    initial_items = sorted(tid_lists.keys())
    counters = {
        "intersections_performed": 0,
        "tid_elements_scanned": 0,
        "candidates_generated": len(initial_items),
        "support_evaluations": len(initial_items),
        "itemsets_discovered": 0,
    }

    frequent_itemsets = {}

    if check_time_mem:
        elapsed_init = time.perf_counter() - total_start
        if not pruning_policy.check_budget(candidate_count=0, level=1, elapsed_time=elapsed_init):
            counters["itemsets_discovered"] = 0
            pruning_policy.counters = counters
            return frequent_itemsets

    def get_support(tids: Set[int]) -> float:
        if sample_weight is not None:
            return float(sum(sample_weight[t] for t in tids) / total_weight)
        return float(len(tids) / total_weight)

    # Initial frequent 1-itemsets with budget ceiling check
    for item in initial_items:
        if len(frequent_itemsets) >= pruning_policy.max_candidates:
            elapsed = time.perf_counter() - total_start
            can_proceed = pruning_policy.check_budget(
                candidate_count=len(frequent_itemsets) + 1,
                level=1,
                elapsed_time=elapsed,
            )
            if not can_proceed:
                break
            if pruning_policy.on_candidate_explosion == "truncate":
                break
        frequent_itemsets[frozenset([item])] = get_support(tid_lists[item])

    # Recursive equivalence class expansion
    def extend(prefix: Tuple[int, ...], items_with_tids: List[Tuple[int, Set[int]]]):
        n_items = len(items_with_tids)
        for i in range(n_items):
            if pruning_policy.is_degraded:
                return
            item_i, tids_i = items_with_tids[i]
            new_prefix = prefix + (item_i,)
            new_items_with_tids: List[Tuple[int, Set[int]]] = []

            for j in range(i + 1, n_items):
                if pruning_policy.is_degraded:
                    return
                item_j, tids_j = items_with_tids[j]
                inter_tids = tids_i.intersection(tids_j)
                counters["intersections_performed"] += 1
                counters["tid_elements_scanned"] += len(tids_i) + len(tids_j)
                counters["candidates_generated"] += 1
                counters["support_evaluations"] += 1

                if check_time_mem and counters["intersections_performed"] % 100 == 0:
                    elapsed = time.perf_counter() - total_start
                    if not pruning_policy.check_budget(
                        candidate_count=0,
                        level=len(new_prefix) + 1,
                        elapsed_time=elapsed,
                    ):
                        return

                inter_weight = (
                    sum(sample_weight[t] for t in inter_tids)
                    if sample_weight is not None
                    else len(inter_tids)
                )

                if inter_weight >= min_weight - 1e-12:
                    if len(frequent_itemsets) >= pruning_policy.max_candidates:
                        elapsed = time.perf_counter() - total_start
                        can_proceed = pruning_policy.check_budget(
                            candidate_count=len(frequent_itemsets) + 1,
                            level=len(new_prefix) + 1,
                            elapsed_time=elapsed,
                        )
                        if not can_proceed:
                            return
                        if pruning_policy.on_candidate_explosion == "truncate":
                            return

                    supp = float(inter_weight / total_weight)
                    pair_itemset = frozenset(new_prefix + (item_j,))
                    frequent_itemsets[pair_itemset] = supp

                    if check_time_mem:
                        elapsed = time.perf_counter() - total_start
                        if not pruning_policy.check_budget(
                            candidate_count=0,
                            level=len(pair_itemset),
                            elapsed_time=elapsed,
                        ):
                            return

                    new_items_with_tids.append((item_j, inter_tids))

            if new_items_with_tids and pruning_policy.should_continue_length(len(new_prefix) + 1):
                extend(new_prefix, new_items_with_tids)
                if pruning_policy.is_degraded:
                    return

    init_pairs = [(item, tid_lists[item]) for item in initial_items]
    if pruning_policy.should_continue_length(1):
        extend((), init_pairs)

    counters["itemsets_discovered"] = len(frequent_itemsets)

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
