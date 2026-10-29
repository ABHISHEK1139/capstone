"""FP-Growth (Frequent Pattern Tree) algorithm implementation."""

import time
from typing import Dict, List, Tuple, Optional, Union, FrozenSet
import numpy as np
from scipy import sparse

from skarm.pruning import PruningPolicy
from skarm.telemetry import TelemetryLogger


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

    def __init__(self):
        self.root = FPNode(-1, 0.0, None)
        # header_table maps item -> [head_node, total_count]
        self.header_table: Dict[int, List[Union[Optional[FPNode], float]]] = {}

    def add_transaction(self, items: List[int], count: float):
        """Insert a transaction into the FP-tree."""
        curr = self.root
        for item in items:
            if item in curr.children:
                curr.children[item].count += count
            else:
                new_node = FPNode(item, count, curr)
                curr.children[item] = new_node
                # Update header table linked list
                if self.header_table[item][0] is None:
                    self.header_table[item][0] = new_node
                else:
                    head = self.header_table[item][0]
                    while head.node_link is not None:
                        head = head.node_link
                    head.node_link = new_node
            curr = curr.children[item]


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

    Parameters
    ----------
    X : np.ndarray or scipy.sparse.csr_matrix of shape (n_samples, n_features)
        Binary transaction matrix.
    min_support : float, default=0.1
        Minimum support threshold.
    max_len : int or None, default=None
        Maximum length of itemsets.
    max_candidates : int, default=100000
        Maximum itemsets limit.
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

    min_count = min_support * total_weight

    # Step 1: Calculate 1-itemset frequencies
    is_sparse = sparse.issparse(X)
    if is_sparse:
        X_csr = X.tocsr()
        item_counts = np.zeros(n_features, dtype=float)
        if sample_weight is not None:
            for i in range(n_samples):
                cols = X_csr.indices[X_csr.indptr[i]: X_csr.indptr[i + 1]]
                item_counts[cols] += sample_weight[i]
        else:
            item_counts = np.array(X_csr.sum(axis=0)).flatten()
    else:
        X_arr = np.asarray(X, dtype=bool)
        if sample_weight is not None:
            item_counts = np.sum(X_arr * sample_weight[:, np.newaxis], axis=0)
        else:
            item_counts = np.sum(X_arr, axis=0)

    # Filter frequent 1-items and sort descending by frequency
    frequent_1 = {
        j: float(item_counts[j])
        for j in range(n_features)
        if item_counts[j] >= min_count - 1e-12
    }

    if not frequent_1:
        if telemetry is not None:
            telemetry.stop({"algorithm": "fpgrowth", "total_frequent_itemsets": 0})
        return {}

    # Rank items: higher frequency first; break ties by item index
    rank_order = sorted(frequent_1.keys(), key=lambda x: (frequent_1[x], -x), reverse=True)
    rank_map = {item: idx for idx, item in enumerate(rank_order)}

    # Step 2: Build initial FP-Tree
    tree = FPTree()
    for item in rank_order:
        tree.header_table[item] = [None, frequent_1[item]]

    if is_sparse:
        for i in range(n_samples):
            w = sample_weight[i] if sample_weight is not None else 1.0
            cols = X_csr.indices[X_csr.indptr[i]: X_csr.indptr[i + 1]]
            filtered = [col for col in cols if col in frequent_1]
            if filtered:
                filtered.sort(key=lambda x: rank_map[x])
                tree.add_transaction(filtered, w)
    else:
        for i in range(n_samples):
            w = sample_weight[i] if sample_weight is not None else 1.0
            row = X_arr[i]
            cols = np.where(row)[0]
            filtered = [col for col in cols if col in frequent_1]
            if filtered:
                filtered.sort(key=lambda x: rank_map[x])
                tree.add_transaction(filtered, w)

    frequent_itemsets: Dict[FrozenSet[int], float] = {}

    # Recursive mining function
    def mine_tree(curr_tree: FPTree, prefix: Tuple[int, ...]):
        # Items sorted ascending by frequency in this header table
        # Break ties deterministically by item index (NT-5)
        items_sorted = sorted(
            curr_tree.header_table.keys(),
            key=lambda x: (curr_tree.header_table[x][1], x)
        )

        for item in items_sorted:
            head_node, item_count = curr_tree.header_table[item]
            if item_count < min_count - 1e-12:
                continue

            new_prefix = prefix + (item,)
            new_itemset = frozenset(new_prefix)
            frequent_itemsets[new_itemset] = float(item_count / total_weight)

            # Check candidate/itemset explosion guardrail
            if not pruning_policy.check_candidate_count(len(frequent_itemsets), level=len(new_prefix)):
                return

            if not pruning_policy.should_continue_length(len(new_prefix)):
                continue

            # Find conditional pattern base
            cond_patterns: List[Tuple[List[int], float]] = []
            node = head_node
            while node is not None:
                path: List[int] = []
                p = node.parent
                while p is not None and p.item != -1:
                    path.append(p.item)
                    p = p.parent
                if path:
                    cond_patterns.append((path[::-1], node.count))
                node = node.node_link

            # Count frequency of items in the conditional pattern base
            cond_item_counts: Dict[int, float] = {}
            for path, count in cond_patterns:
                for p_item in path:
                    cond_item_counts[p_item] = cond_item_counts.get(p_item, 0.0) + count

            # Filter frequent items for conditional tree
            cond_frequent = {
                it: cnt for it, cnt in cond_item_counts.items()
                if cnt >= min_count - 1e-12
            }

            if not cond_frequent:
                continue

            # Build conditional tree
            cond_rank_order = sorted(cond_frequent.keys(), key=lambda x: (cond_frequent[x], -x), reverse=True)
            cond_rank_map = {it: idx for idx, it in enumerate(cond_rank_order)}

            cond_tree = FPTree()
            for it in cond_rank_order:
                cond_tree.header_table[it] = [None, cond_frequent[it]]

            for path, count in cond_patterns:
                filtered_path = [it for it in path if it in cond_frequent]
                if filtered_path:
                    filtered_path.sort(key=lambda x: cond_rank_map[x])
                    cond_tree.add_transaction(filtered_path, count)

            # Recurse
            mine_tree(cond_tree, new_prefix)
            if pruning_policy.is_degraded:
                return

    mine_tree(tree, ())

    if telemetry is not None:
        telemetry.stop({
            "algorithm": "fpgrowth",
            "total_frequent_itemsets": len(frequent_itemsets),
            "is_degraded": pruning_policy.is_degraded,
            "peak_candidates": pruning_policy.peak_candidate_count,
        })

    return frequent_itemsets
