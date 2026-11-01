"""Empirical Equivalence and Telemetry Audit Verification Script.

Executes standalone multi-dataset verification across Apriori, FP-Growth, and Eclat:
1. Result Determinism: Max support difference on Sparse Retail, Medium Dense, Weighted Transactions.
2. Rule Metric Equivalence: Support, confidence, lift, leverage, conviction, Zhang's metric.
3. Telemetry Counters: Non-zero positive counters per engine.
4. Ordering Determinism: Consecutive runs and canonical ordering.
"""

import math
import sys
import time
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from typing import Dict, Any

from skarm.estimator import AssociationRulesEstimator


def canonical_itemsets_df(df: pd.DataFrame) -> pd.DataFrame:
    """Sort itemsets DataFrame by canonical key: (-support, length, sorted_items)."""
    if df.empty:
        return df.copy()
    records = df.to_dict("records")
    records.sort(
        key=lambda r: (
            -round(r["support"], 9),
            r["length"],
            sorted(list(r["itemsets"])),
        )
    )
    return pd.DataFrame(records)


def run_verification():
    print("=" * 80)
    print("SKARM EMPIRICAL CHALLENGER: CROSS-ENGINE EQUIVALENCE & TELEMETRY AUDIT")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. Dataset Loading
    # -------------------------------------------------------------
    print("\n[1/4] Loading Test Datasets...")
    # (a) Sparse Retail (groceries.csv)
    with open("data/groceries.csv", "r", encoding="utf-8") as f:
        groceries = [line.strip().split(",") for line in f if line.strip()][:1500]
    print(f"  - Sparse Retail (groceries): {len(groceries)} transactions")

    # (b) Medium Dense (tic_tac_toe.dat)
    with open("data/tic_tac_toe.dat", "r", encoding="utf-8") as f:
        tic_tac_toe = [line.strip().split() for line in f if line.strip()]
    print(f"  - Medium Dense (tic_tac_toe): {len(tic_tac_toe)} transactions")

    # (c) Weighted Transactions
    rng = np.random.default_rng(2026)
    X_weighted = rng.choice([0, 1], size=(250, 14), p=[0.75, 0.25])
    weights = rng.uniform(0.5, 4.5, size=250)
    print(f"  - Weighted Transactions: {X_weighted.shape[0]} samples x {X_weighted.shape[1]} features, weights sum={weights.sum():.2f}")

    # -------------------------------------------------------------
    # 2. Result Determinism & Support Discrepancy (Delta_supp)
    # -------------------------------------------------------------
    print("\n[2/4] Testing Result Determinism (Delta_supp < 1e-9)...")
    datasets = [
        ("Sparse Retail (groceries)", groceries, 0.02, None, None),
        ("Medium Dense (tic_tac_toe)", tic_tac_toe, 0.15, 4, None),
        ("Weighted Transactions", X_weighted, 0.08, None, weights),
    ]

    for name, data, min_supp, max_len, w in datasets:
        print(f"\n  >> Dataset: {name} (min_support={min_supp})")
        ests = {}
        times = {}
        for algo in ["apriori", "fpgrowth", "eclat"]:
            t0 = time.perf_counter()
            ests[algo] = AssociationRulesEstimator(
                min_support=min_supp,
                max_len=max_len,
                algorithm=algo,
            ).fit(data, sample_weight=w)
            times[algo] = time.perf_counter() - t0

        keys_a = set(ests["apriori"].frequent_itemsets_.keys())
        keys_f = set(ests["fpgrowth"].frequent_itemsets_.keys())
        keys_e = set(ests["eclat"].frequent_itemsets_.keys())

        print(f"     Apriori:   {len(keys_a)} itemsets in {times['apriori']:.4f}s")
        print(f"     FP-Growth: {len(keys_f)} itemsets in {times['fpgrowth']:.4f}s")
        print(f"     Eclat:     {len(keys_e)} itemsets in {times['eclat']:.4f}s")

        assert keys_a == keys_f == keys_e, "Mismatch in discovered frequent itemset keys!"

        # Compute max support discrepancies
        max_diff_af = max(abs(ests["apriori"].frequent_itemsets_[k] - ests["fpgrowth"].frequent_itemsets_[k]) for k in keys_a)
        max_diff_ae = max(abs(ests["apriori"].frequent_itemsets_[k] - ests["eclat"].frequent_itemsets_[k]) for k in keys_a)
        max_diff_fe = max(abs(ests["fpgrowth"].frequent_itemsets_[k] - ests["eclat"].frequent_itemsets_[k]) for k in keys_a)

        print(f"     Max Delta_supp (Apriori vs FP-Growth): {max_diff_af:.3e}")
        print(f"     Max Delta_supp (Apriori vs Eclat):     {max_diff_ae:.3e}")
        print(f"     Max Delta_supp (FP-Growth vs Eclat):   {max_diff_fe:.3e}")
        assert max(max_diff_af, max_diff_ae, max_diff_fe) < 1e-9, "Delta_supp exceeded 1e-9 threshold!"
        print("     [VERDICT]: PASSED (Delta_supp < 1e-9 satisfied)")

    # -------------------------------------------------------------
    # 3. Association Rule Equivalence Across All Metrics
    # -------------------------------------------------------------
    print("\n[3/4] Testing Association Rule Equivalence across all 6 metrics...")
    rule_metrics = ["confidence", "lift", "support", "leverage", "conviction", "zhangs_metric"]

    for m in rule_metrics:
        thresh = 0.15 if m != "lift" else 1.05
        ests = {
            algo: AssociationRulesEstimator(
                min_support=0.08,
                metric=m,
                min_threshold=thresh,
                algorithm=algo,
            ).fit(X_weighted, sample_weight=weights)
            for algo in ["apriori", "fpgrowth", "eclat"]
        }

        def to_dict(rules_df):
            return {
                (frozenset(int(x) for x in r["antecedent"]), frozenset(int(x) for x in r["consequent"])): r
                for _, r in rules_df.iterrows()
            }

        ra_dict = to_dict(ests["apriori"].rules_)
        rf_dict = to_dict(ests["fpgrowth"].rules_)
        re_dict = to_dict(ests["eclat"].rules_)

        print(f"  >> Metric '{m}': rule count = {len(ra_dict)} (Apriori), {len(rf_dict)} (FP-Growth), {len(re_dict)} (Eclat)")
        assert set(ra_dict.keys()) == set(rf_dict.keys()) == set(re_dict.keys()), f"Rule sets differ under metric {m}!"

        # Numerical comparison across all columns
        max_diffs = {col: 0.0 for col in ["support", "confidence", "lift", "leverage", "conviction", "zhangs_metric"]}
        for r_key in ra_dict:
            for col in max_diffs:
                va = ra_dict[r_key][col]
                vf = rf_dict[r_key][col]
                ve = re_dict[r_key][col]
                if math.isinf(va) or math.isinf(vf) or math.isinf(ve):
                    assert math.isinf(va) and math.isinf(vf) and math.isinf(ve)
                else:
                    diff_af = abs(va - vf)
                    diff_ae = abs(va - ve)
                    max_diffs[col] = max(max_diffs[col], diff_af, diff_ae)

        diff_summary = ", ".join(f"{col}: {max_diffs[col]:.2e}" for col in max_diffs)
        print(f"     Max metric diffs: {diff_summary}")
        for col, max_d in max_diffs.items():
            assert max_d < 1e-9, f"Rule metric {col} exceeded tolerance 1e-9 with diff={max_d}"

    print("  [VERDICT]: PASSED (All 6 rule metrics strictly equivalent across engines)")

    # -------------------------------------------------------------
    # 4. Telemetry Auditability & Ordering Determinism
    # -------------------------------------------------------------
    print("\n[4/4] Testing Telemetry Auditability & Ordering Determinism...")
    est_a = AssociationRulesEstimator(min_support=0.02, algorithm="apriori").fit(groceries)
    est_f = AssociationRulesEstimator(min_support=0.02, algorithm="fpgrowth").fit(groceries)
    est_e = AssociationRulesEstimator(min_support=0.02, algorithm="eclat").fit(groceries)

    print("  >> Apriori Telemetry:")
    print("     status:", est_a.metadata_["status"], "| is_complete:", est_a.metadata_["is_complete"])
    print("     counters:", est_a.metadata_["counters"])
    assert est_a.metadata_["counters"]["candidates_generated"] > 0
    assert est_a.metadata_["counters"]["joins_attempted"] > 0
    assert est_a.metadata_["counters"]["scans_performed"] > 0

    print("  >> FP-Growth Telemetry:")
    print("     status:", est_f.metadata_["status"], "| is_complete:", est_f.metadata_["is_complete"])
    print("     counters:", est_f.metadata_["counters"])
    assert est_f.metadata_["counters"]["trees_constructed"] > 0
    assert est_f.metadata_["counters"]["nodes_created"] > 0
    assert est_f.metadata_["counters"]["paths_traversed"] > 0

    print("  >> Eclat Telemetry:")
    print("     status:", est_e.metadata_["status"], "| is_complete:", est_e.metadata_["is_complete"])
    print("     counters:", est_e.metadata_["counters"])
    assert est_e.metadata_["counters"]["intersections_performed"] > 0
    assert est_e.metadata_["counters"]["tid_elements_scanned"] > 0

    print("  [VERDICT]: PASSED (All telemetry counters strictly populated, valid, and non-zero)")

    # Consecutive run determinism
    print("\n  >> Consecutive Run Determinism:")
    for algo in ["apriori", "fpgrowth", "eclat"]:
        run1 = AssociationRulesEstimator(min_support=0.15, max_len=4, algorithm=algo).fit(tic_tac_toe)
        run2 = AssociationRulesEstimator(min_support=0.15, max_len=4, algorithm=algo).fit(tic_tac_toe)
        itemsets_match = run1.itemsets_.equals(run2.itemsets_)
        rules_match = run1.rules_.equals(run2.rules_)
        print(f"     {algo}: itemsets run1==run2? {itemsets_match} | rules run1==run2? {rules_match}")
        assert itemsets_match and rules_match

    # Canonical sorting cross-engine
    print("  >> Canonical Cross-Engine Ordering:")
    canon_a = canonical_itemsets_df(est_a.itemsets_)
    canon_f = canonical_itemsets_df(est_f.itemsets_)
    canon_e = canonical_itemsets_df(est_e.itemsets_)
    print("     canon_a equals canon_f?", canon_a.equals(canon_f))
    print("     canon_a equals canon_e?", canon_a.equals(canon_e))
    assert canon_a.equals(canon_f) and canon_a.equals(canon_e)
    print("  [VERDICT]: PASSED (Canonical cross-engine ordering verified)")

    print("\n" + "=" * 80)
    print("FINAL EMPIRICAL VERDICT: ALL TESTS PASSED -> APPROVE")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
