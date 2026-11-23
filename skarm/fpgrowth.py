"""FP-Growth (Frequent Pattern Tree) algorithm implementation."""

import time
from typing import Dict, List, Tuple, Optional, Union, FrozenSet, Any
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


class FPNode:
    """A node in the FP-Tree."""
    __slots__ = ("item", "count", "parent", "children", "node_link")

    def __init__(self, item: int, count: float, parent: Optional["FPNode"]):
        self.item = item
        self.count = count
        self.parent = parent
        self.children: Dict[int, "FPNode"] = {}
        self.node_link: Optional["FPNode"] = None


class FPTree:
    """Frequent Pattern Tree data structure."""

    def __init__(self, counters: Optional[Dict[str, int]] = None):
        self.root = FPNode(-1, 0.0, None)
        # header_table maps item -> [head_node, tail_node, total_count]
        self.header_table: Dict[int, List[Any]] = {}
        self.counters = counters
        if self.counters is not None:
            self.counters["trees_constructed"] += 1
            self.counters["nodes_created"] += 1

    def add_transaction(self, items: List[int], count: float):
        """Insert a transaction into the FP-tree with O(1) linked list tail append."""
        curr = self.root
        header_table = self.header_table
        counters = self.counters
        for item in items:
            child = curr.children.get(item)
            if child is not None:
                child.count += count
                curr = child
            else:
                new_node = FPNode(item, count, curr)
                curr.children[item] = new_node
                if counters is not None:
                    counters["nodes_created"] += 1
                entry = header_table[item]
                if entry[0] is None:
                    entry[0] = new_node
                    entry[1] = new_node
                else:
                    entry[1].node_link = new_node
                    entry[1] = new_node
                curr = new_node


def _pack_rows_to_uint(X_bool: np.ndarray) -> np.ndarray:
    """Pack a boolean matrix with <= 64 columns into 1D unsigned integer row codes."""
    n_rows, _ = X_bool.shape
    packed = np.packbits(X_bool, axis=1, bitorder="little")
    n_bytes = packed.shape[1]
    if n_bytes == 1:
        return packed[:, 0]
    if n_bytes == 2:
        return np.ascontiguousarray(packed).view(np.uint16).ravel()
    if n_bytes <= 4:
        buf = np.zeros((n_rows, 4), dtype=np.uint8)
        buf[:, :n_bytes] = packed
        return buf.view(np.uint32).ravel()
    buf = np.zeros((n_rows, 8), dtype=np.uint8)
    buf[:, :n_bytes] = packed
    return buf.view(np.uint64).ravel()


def _insert_packed_transactions(
    tree: FPTree,
    X_sub: np.ndarray,
    col_items: List[int],
    sample_weight: Optional[np.ndarray],
    min_tx_len: int = 1,
):
    """Compress duplicate bit-packed transactions and insert into FP-Tree."""
    if min_tx_len > 1:
        row_sums = np.count_nonzero(X_sub, axis=1)
        keep_mask = row_sums >= min_tx_len
        if not np.any(keep_mask):
            return
        X_sub = X_sub[keep_mask]
        if sample_weight is not None:
            sample_weight = sample_weight[keep_mask]

    codes = _pack_rows_to_uint(X_sub)
    if min_tx_len <= 1:
        nz_mask = codes != 0
        if not np.any(nz_mask):
            return
        nz_codes = codes[nz_mask]
        nz_weights = sample_weight[nz_mask] if sample_weight is not None else None
    else:
        nz_codes = codes
        nz_weights = sample_weight

    if nz_weights is None:
        unique_codes, counts = np.unique(nz_codes, return_counts=True)
        unique_list = unique_codes.tolist()
        counts_list = counts.tolist()
    else:
        unique_codes, inv = np.unique(nz_codes, return_inverse=True)
        counts = np.bincount(inv, weights=nz_weights)
        unique_list = unique_codes.tolist()
        counts_list = counts.tolist()

    add_tx = tree.add_transaction
    for code, cnt in zip(unique_list, counts_list):
        if cnt == 0:
            continue
        items: List[int] = []
        c = code
        while c:
            low = c & -c
            items.append(col_items[low.bit_length() - 1])
            c ^= low
        add_tx(items, float(cnt))


def fpgrowth(
    X: Union[np.ndarray, sparse.csr_matrix],
    min_support: float = 0.1,
    max_len: Optional[int] = None,
    max_candidates: int = 100000,
    sample_weight: Optional[np.ndarray] = None,
    pruning_policy: Optional[PruningPolicy] = None,
    telemetry: Optional[TelemetryLogger] = None,
) -> Dict[FrozenSet[int], float]:
    """Mine frequent itemsets using the FP-Growth algorithm.

    Incorporates Grahne & Zhu's FP-Array co-occurrence pruning and
    bit-packed transaction compression before FP-Tree construction to
    eliminate redundant tree traversals while preserving exact supports.

    Parameters
    ----------
    X : np.ndarray or scipy.sparse.csr_matrix of shape (n_samples, n_features)
        Binary transaction matrix.
    min_support : float, default=0.1
        Minimum support threshold.
    max_len : int or None, default=None
        Maximum length of itemsets.
    max_candidates : int, default=100000
        Upper bound on cumulative discovered frequent itemsets |F| extracted
        from conditional pattern trees. In FP-Growth, frequent prefixes are projected
        recursively from conditional FP-trees without generating unverified candidate
        combinations. If |F| >= max_candidates, the configured candidate explosion
        policy ('raise', 'truncate', or 'degrade') is triggered.
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

    min_count = min_support * total_weight
    total_start = getattr(pruning_policy, "start_time", time.perf_counter())
    check_time_mem = (
        pruning_policy.max_time_seconds is not None
        or pruning_policy.max_memory_mb is not None
    )

    # Operation counters (R2 & R3)
    counters = {
        "trees_constructed": 0,
        "nodes_created": 0,
        "tree_nodes_created": 0,
        "paths_traversed": 0,
        "candidates_generated": 0,
        "support_evaluations": 0,
        "itemsets_discovered": 0,
    }

    # Step 1: Calculate 1-itemset frequencies
    is_sparse = sparse.issparse(X)
    if is_sparse:
        X_csr = X.tocsr()
        if not X_csr.has_canonical_format:
            X_csr.sum_duplicates()
        X_csr.eliminate_zeros()
        if X_csr.dtype != bool:
            X_csr = X_csr.astype(bool)
        if sample_weight is not None:
            item_counts = np.asarray(X_csr.T @ sample_weight, dtype=float).ravel()
        else:
            item_counts = np.bincount(X_csr.indices, minlength=n_features).astype(float)
    else:
        X_arr = np.asarray(X, dtype=bool)
        if sample_weight is not None:
            item_counts = np.asarray(X_arr.T @ sample_weight, dtype=float)
        else:
            item_counts = np.count_nonzero(X_arr, axis=0).astype(float)

    # Filter frequent 1-items and sort descending by frequency
    freq_indices = np.flatnonzero(item_counts >= min_count - 1e-12)
    frequent_1 = {int(j): float(item_counts[j]) for j in freq_indices}

    if not frequent_1:
        counters["trees_constructed"] = 1
        pruning_policy.counters = counters
        if telemetry is not None:
            telemetry.counters = counters
            telemetry.stop({"algorithm": "fpgrowth", "total_frequent_itemsets": 0, "counters": counters})
        return {}

    # Rank items: higher frequency first; break ties by item index
    rank_order = sorted(frequent_1.keys(), key=lambda x: (frequent_1[x], -x), reverse=True)
    n_freq = len(rank_order)
    rank_map = {item: idx for idx, item in enumerate(rank_order)}

    # Step 2: Grahne & Zhu (2003) FP-Array co-occurrence matrix when feasible (min_support > 0 and n_freq <= 256)
    fp_array: Optional[np.ndarray] = None
    cond_freq_by_rank: Optional[List[Dict[int, float]]] = None
    need_tree_for_depth3 = True

    if min_support > 0.0 and 2 <= n_freq <= 256 and pruning_policy.should_continue_length(1):
        if is_sparse:
            X_freq_csr = X_csr[:, rank_order]
            if sample_weight is None:
                mat_dtype = np.float32 if n_samples <= 16_000_000 else np.float64
                X_mat = X_freq_csr.astype(mat_dtype)
                fp_array = (X_mat.T @ X_mat).toarray()
                del X_mat
            else:
                X_f64 = X_freq_csr.astype(np.float64)
                X_w = X_f64.multiply(sample_weight[:, np.newaxis])
                fp_array = (X_w.T @ X_f64).toarray()
                del X_f64, X_w
        else:
            if sample_weight is None:
                block_size = 1024
                if n_samples <= block_size:
                    X_mat = X_arr[:, rank_order].astype(np.float32)
                    fp_array = (X_mat.T @ X_mat).astype(np.float64, copy=False)
                    del X_mat
                else:
                    fp_array = np.zeros((n_freq, n_freq), dtype=np.float64)
                    for start in range(0, n_samples, block_size):
                        X_blk = X_arr[start: start + block_size, rank_order].astype(np.float32)
                        fp_array += X_blk.T @ X_blk
            else:
                X_f64 = X_arr[:, rank_order].astype(np.float64)
                fp_array = (X_f64 * sample_weight[:, np.newaxis]).T @ X_f64
                del X_f64

        cond_freq_by_rank = []
        cond_has_triangle: List[bool] = []
        cond_tri_positions: List[List[int]] = []
        adj_masks: List[int] = [0] * n_freq
        any_triangle = False
        tree_needed_positions = set()
        thresh = min_count - 1e-12
        can_depth3 = pruning_policy.should_continue_length(2)

        for idx in range(n_freq):
            preds: Dict[int, float] = {}
            has_tri = False
            tri_pos_list: List[int] = []
            if idx > 0:
                col_vals = fp_array[:idx, idx]
                hits = np.flatnonzero(col_vals >= thresh)
                if hits.size > 0:
                    hits_list = hits.tolist()
                    mask_idx = 0
                    for h in hits_list:
                        preds[rank_order[h]] = float(col_vals[h])
                        mask_idx |= (1 << h)
                    adj_masks[idx] = mask_idx
                    if can_depth3 and hits.size >= 2:
                        # Bitmask triangle check: u < v < idx is a triangle iff
                        # bit u is set in adj_masks[v] and both u, v are in mask_idx.
                        tri_mask_bits = 0
                        for v in hits_list:
                            common = adj_masks[v] & mask_idx
                            if common:
                                tri_mask_bits |= common | (1 << v)
                        if tri_mask_bits:
                            has_tri = True
                            any_triangle = True
                            m = tri_mask_bits
                            while m:
                                low = m & -m
                                tri_pos_list.append(low.bit_length() - 1)
                                m ^= low
            cond_freq_by_rank.append(preds)
            cond_has_triangle.append(has_tri)
            cond_tri_positions.append(tri_pos_list)

        # Conditional FP-Array depth-3 check for unweighted dense matrices (Grahne & Zhu FP-Growth*):
        # verify whether triangle candidates actually yield >= 1 frequent 3-itemset before
        # allocating Python FPNode objects in the root FP-Tree.
        if any_triangle and not is_sparse and sample_weight is None and n_samples <= 16_000_000:
            any_verified_depth3 = False
            for idx in range(n_freq):
                if not cond_has_triangle[idx]:
                    continue
                tri_pos = cond_tri_positions[idx]
                tri_cols = [rank_order[p] for p in tri_pos]
                X_cond = X_arr[X_arr[:, rank_order[idx]]][:, tri_cols]
                if X_cond.shape[0] < thresh:
                    cond_has_triangle[idx] = False
                    continue
                X_c32 = X_cond.astype(np.float32)
                fp_c = X_c32.T @ X_c32
                del X_c32
                # Check strictly upper-triangular co-occurrences >= thresh
                k_tri = len(tri_pos)
                found_3 = False
                for u in range(k_tri):
                    row_u = fp_c[u]
                    for v in range(u + 1, k_tri):
                        if row_u[v] >= thresh:
                            found_3 = True
                            tree_needed_positions.add(idx)
                            tree_needed_positions.add(tri_pos[u])
                            tree_needed_positions.add(tri_pos[v])
                if found_3:
                    any_verified_depth3 = True
                else:
                    cond_has_triangle[idx] = False
            need_tree_for_depth3 = any_verified_depth3
        else:
            for idx in range(n_freq):
                if cond_has_triangle[idx]:
                    tree_needed_positions.add(idx)
                    tree_needed_positions.update(cond_tri_positions[idx])
            need_tree_for_depth3 = any_triangle

    # Step 3: Build initial FP-Tree (pruned to items participating in >=3-itemsets when FP-Array proves exact)
    tree = FPTree(counters=counters)
    if fp_array is not None and not need_tree_for_depth3:
        # FP-Array proved no item has >= 2 frequent predecessors (or max_len <= 2),
        # so no frequent itemset of cardinality >= 3 exists; header table holds L1 counts.
        for item in rank_order:
            tree.header_table[item] = [None, None, frequent_1[item]]
    elif fp_array is not None:
        active_pos = sorted(tree_needed_positions)
        active_items = [rank_order[p] for p in active_pos]
        for item in rank_order:
            tree.header_table[item] = [None, None, frequent_1[item]]

        if not is_sparse and len(active_items) <= 64:
            # FP-Array already holds exact 1- and 2-itemset counts, so the root tree
            # only needs transactions with >= 3 active items to mine >=3-itemsets.
            _insert_packed_transactions(
                tree, X_arr[:, active_items], active_items, sample_weight, min_tx_len=3
            )
        elif not is_sparse:
            X_act = X_arr[:, active_items]
            rows_nz, cols_nz = np.nonzero(X_act)
            if rows_nz.size > 0:
                boundaries = np.flatnonzero(np.diff(rows_nz)) + 1
                starts = np.concatenate(([0], boundaries))
                ends = np.concatenate((boundaries, [rows_nz.size]))
                add_tx = tree.add_transaction
                for seg_idx in range(starts.size):
                    if check_time_mem and seg_idx % 5000 == 0:
                        elapsed = time.perf_counter() - total_start
                        if not pruning_policy.check_budget(candidate_count=0, level=1, elapsed_time=elapsed):
                            pruning_policy.counters = counters
                            return {}
                    s, e = int(starts[seg_idx]), int(ends[seg_idx])
                    if e - s < 3:
                        continue
                    r_id = int(rows_nz[s])
                    w = float(sample_weight[r_id]) if sample_weight is not None else 1.0
                    add_tx([active_items[c] for c in cols_nz[s:e].tolist()], w)
        else:
            X_act_csr = X_csr[:, active_items]
            X_act_csr.sort_indices()
            indptr = X_act_csr.indptr
            indices = X_act_csr.indices
            add_tx = tree.add_transaction
            for i in range(n_samples):
                if check_time_mem and i % 5000 == 0:
                    elapsed = time.perf_counter() - total_start
                    if not pruning_policy.check_budget(candidate_count=0, level=1, elapsed_time=elapsed):
                        pruning_policy.counters = counters
                        return {}
                s, e = indptr[i], indptr[i + 1]
                if e - s >= 3:
                    w = float(sample_weight[i]) if sample_weight is not None else 1.0
                    add_tx([active_items[c] for c in indices[s:e].tolist()], w)
    else:
        for item in rank_order:
            tree.header_table[item] = [None, None, frequent_1[item]]

        if pruning_policy.should_continue_length(1):
            if not is_sparse and n_freq <= 64 and min_support > 0.0:
                _insert_packed_transactions(tree, X_arr[:, rank_order], rank_order, sample_weight)
            elif is_sparse:
                indptr = X_csr.indptr
                indices = X_csr.indices
                add_tx = tree.add_transaction
                for i in range(n_samples):
                    if check_time_mem and i % 5000 == 0:
                        elapsed = time.perf_counter() - total_start
                        if not pruning_policy.check_budget(candidate_count=0, level=1, elapsed_time=elapsed):
                            pruning_policy.counters = counters
                            return {}
                    w = float(sample_weight[i]) if sample_weight is not None else 1.0
                    cols = indices[indptr[i]: indptr[i + 1]]
                    filtered = [int(col) for col in cols if col in frequent_1]
                    if filtered:
                        filtered.sort(key=rank_map.__getitem__)
                        add_tx(filtered, w)
            else:
                X_freq_bool = X_arr[:, rank_order]
                rows_nz, cols_nz = np.nonzero(X_freq_bool)
                if rows_nz.size > 0:
                    boundaries = np.flatnonzero(np.diff(rows_nz)) + 1
                    starts = np.concatenate(([0], boundaries))
                    ends = np.concatenate((boundaries, [rows_nz.size]))
                    add_tx = tree.add_transaction
                    for seg_idx in range(starts.size):
                        if check_time_mem and seg_idx % 5000 == 0:
                            elapsed = time.perf_counter() - total_start
                            if not pruning_policy.check_budget(candidate_count=0, level=1, elapsed_time=elapsed):
                                pruning_policy.counters = counters
                                return {}
                        s, e = int(starts[seg_idx]), int(ends[seg_idx])
                        r_id = int(rows_nz[s])
                        w = float(sample_weight[r_id]) if sample_weight is not None else 1.0
                        add_tx([rank_order[c] for c in cols_nz[s:e].tolist()], w)

    frequent_itemsets: Dict[FrozenSet[int], float] = {}
    max_cands = pruning_policy.max_candidates

    # Check budget after initial FP-Tree construction
    if check_time_mem:
        elapsed_init = time.perf_counter() - total_start
        if not pruning_policy.check_budget(candidate_count=0, level=1, elapsed_time=elapsed_init):
            pruning_policy.counters = counters
            return frequent_itemsets

    def _record_itemset(prefix_tuple: Tuple[int, ...], item_count: float) -> bool:
        """Record a frequent itemset with candidate, time, and memory guardrails."""
        if len(frequent_itemsets) >= max_cands:
            elapsed = time.perf_counter() - total_start
            can_proceed = pruning_policy.check_budget(
                candidate_count=len(frequent_itemsets) + 1,
                level=len(prefix_tuple),
                elapsed_time=elapsed,
            )
            if not can_proceed:
                return False
            if pruning_policy.on_candidate_explosion == "truncate":
                return False

        frequent_itemsets[frozenset(prefix_tuple)] = float(item_count / total_weight)

        if check_time_mem:
            elapsed = time.perf_counter() - total_start
            if not pruning_policy.check_budget(
                candidate_count=0,
                level=len(prefix_tuple),
                elapsed_time=elapsed,
            ):
                return False
        return True

    # Recursive mining function
    def mine_tree(curr_tree: FPTree, prefix: Tuple[int, ...], is_root: bool = False):
        items_sorted = sorted(
            curr_tree.header_table.keys(),
            key=lambda x: (curr_tree.header_table[x][2], x),
        )

        for item in items_sorted:
            if pruning_policy.is_degraded:
                return

            entry = curr_tree.header_table[item]
            head_node, item_count = entry[0], entry[2]
            counters["candidates_generated"] += 1
            counters["support_evaluations"] += 1
            if item_count < min_count - 1e-12:
                continue

            new_prefix = prefix + (item,)
            if not _record_itemset(new_prefix, item_count):
                return

            if not pruning_policy.should_continue_length(len(new_prefix)):
                continue

            # Depth-1 FP-Array fast path: exact conditional item counts are already known
            if is_root and cond_freq_by_rank is not None:
                item_rank = rank_map[item]
                cond_frequent = cond_freq_by_rank[item_rank]
                if not cond_frequent:
                    continue
                if not cond_has_triangle[item_rank] or len(cond_frequent) == 1 or not pruning_policy.should_continue_length(2):
                    for p_item, p_cnt in sorted(cond_frequent.items(), key=lambda kv: (kv[1], kv[0])):
                        if pruning_policy.is_degraded:
                            return
                        counters["candidates_generated"] += 1
                        counters["support_evaluations"] += 1
                        if not _record_itemset(new_prefix + (p_item,), p_cnt):
                            return
                    continue

                # >= 2 frequent predecessors and max_len allows >= 3: extract filtered paths directly
                cond_rank_order = sorted(
                    cond_frequent.keys(),
                    key=lambda x: (cond_frequent[x], -x),
                    reverse=True,
                )
                cond_rank_map = {it: idx for idx, it in enumerate(cond_rank_order)}
                cond_tree = FPTree(counters=counters)
                for it in cond_rank_order:
                    cond_tree.header_table[it] = [None, None, cond_frequent[it]]

                node = head_node
                add_cond_tx = cond_tree.add_transaction
                while node is not None:
                    counters["paths_traversed"] += 1
                    path: List[int] = []
                    p = node.parent
                    while p is not None and p.item != -1:
                        if p.item in cond_frequent:
                            path.append(p.item)
                        p = p.parent
                    # A 1-item conditional path creates a root-child with no ancestors,
                    # contributing nothing to deeper extensions beyond header_table[p][2].
                    if len(path) >= 2:
                        path.sort(key=cond_rank_map.__getitem__)
                        add_cond_tx(path, node.count)
                    node = node.node_link

                mine_tree(cond_tree, new_prefix, is_root=False)
                if pruning_policy.is_degraded:
                    return
                continue

            # Standard conditional pattern base traversal
            cond_patterns: List[Tuple[List[int], float]] = []
            cond_item_counts: Dict[int, float] = {}
            node = head_node
            while node is not None:
                counters["paths_traversed"] += 1
                path = []
                p = node.parent
                nc = node.count
                while p is not None and p.item != -1:
                    pi = p.item
                    path.append(pi)
                    cond_item_counts[pi] = cond_item_counts.get(pi, 0.0) + nc
                    p = p.parent
                if path:
                    cond_patterns.append((path, nc))
                node = node.node_link

            cond_frequent = {
                it: cnt for it, cnt in cond_item_counts.items()
                if cnt >= min_count - 1e-12
            }

            if not cond_frequent:
                continue

            if len(cond_frequent) == 1 or not pruning_policy.should_continue_length(len(new_prefix) + 1):
                for p_item, p_cnt in sorted(cond_frequent.items(), key=lambda kv: (kv[1], kv[0])):
                    if pruning_policy.is_degraded:
                        return
                    counters["candidates_generated"] += 1
                    counters["support_evaluations"] += 1
                    if not _record_itemset(new_prefix + (p_item,), p_cnt):
                        return
                continue

            cond_rank_order = sorted(
                cond_frequent.keys(),
                key=lambda x: (cond_frequent[x], -x),
                reverse=True,
            )
            cond_rank_map = {it: idx for idx, it in enumerate(cond_rank_order)}

            cond_tree = FPTree(counters=counters)
            for it in cond_rank_order:
                cond_tree.header_table[it] = [None, None, cond_frequent[it]]

            add_cond_tx = cond_tree.add_transaction
            for path, count in cond_patterns:
                filtered_path = [it for it in path if it in cond_frequent]
                if len(filtered_path) >= 2:
                    filtered_path.sort(key=cond_rank_map.__getitem__)
                    add_cond_tx(filtered_path, count)

            mine_tree(cond_tree, new_prefix, is_root=False)
            if pruning_policy.is_degraded:
                return

    mine_tree(tree, (), is_root=True)

    counters["itemsets_discovered"] = len(frequent_itemsets)
    counters["tree_nodes_created"] = counters["nodes_created"]

    pruning_policy.counters = counters
    if telemetry is not None:
        telemetry.counters = counters
        telemetry.stop({
            "algorithm": "fpgrowth",
            "total_frequent_itemsets": len(frequent_itemsets),
            "is_degraded": pruning_policy.is_degraded,
            "peak_candidates": pruning_policy.peak_candidate_count,
            "counters": counters,
        })

    return frequent_itemsets
