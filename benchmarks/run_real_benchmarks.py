"""End-to-End Real-World Dataset Empirical Evaluation for SKARM.
Datasets evaluated:
1. Groceries (N=9,835, M=169): Market Basket Analysis and Real Rules
2. Chess (N=3,196, M=75, density=49.33%): Combinatorial stress test and PruningPolicy
3. Retail (N=88,162, M=16,470, sparsity=99.94%): Large-scale scalability and throughput
"""
from __future__ import annotations

import sys
import os
import time
import json
import warnings
import psutil
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skarm.datasets import load_groceries, load_chess, load_retail, dataset_summary
from skarm.preprocessing import TransactionEncoder
from skarm.estimator import AssociationRulesEstimator
from skarm.pruning import PruningPolicy, DegradedModeWarning
from skarm.explorer import RuleExplorer

def get_peak_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_groceries_experiment():
    print("=" * 70)
    print("EXPERIMENT 1: GROCERIES MARKET BASKET ANALYSIS (N=9,835, M=169)")
    print("=" * 70)
    tx = load_groceries()
    stats = dataset_summary(tx)
    print(f"Dataset stats: {stats['n_transactions']} txns, {stats['n_unique_items']} items, {stats['avg_transaction_length']} avg len.")

    results = {}
    algorithms = ["fpgrowth", "eclat", "apriori"]
    min_supports = [0.02, 0.01]
    min_confidence = 0.25

    for min_sup in min_supports:
        results[f"min_sup_{min_sup}"] = {}
        for algo in algorithms:
            print(f"Running {algo.upper()} (min_support={min_sup}, min_confidence={min_confidence})...")
            m0 = get_peak_memory_mb()
            t0 = time.perf_counter()
            
            est = AssociationRulesEstimator(
                min_support=min_sup,
                metric="confidence",
                min_threshold=min_confidence,
                algorithm=algo,
                sparse_output=True,
            )
            est.fit(tx)
            elapsed = time.perf_counter() - t0
            m1 = get_peak_memory_mb()
            peak_ram = max(m1 - m0, 0.1)

            n_itemsets = len(est.frequent_itemsets_)
            n_rules = len(est.rules_)
            print(f"  -> Done in {elapsed:.3f}s | Peak RAM: {peak_ram:.1f} MB | Itemsets: {n_itemsets} | Rules: {n_rules}")

            results[f"min_sup_{min_sup}"][algo] = {
                "elapsed_sec": round(elapsed, 4),
                "peak_ram_mb": round(peak_ram, 2),
                "n_itemsets": n_itemsets,
                "n_rules": n_rules,
                "throughput_tx_per_sec": round(len(tx) / elapsed, 1) if elapsed > 0 else 0
            }

    # Extract top actionable rules from best model
    best_est = AssociationRulesEstimator(
        min_support=0.01,
        metric="lift",
        min_threshold=1.5,
        algorithm="fpgrowth",
        sparse_output=True
    ).fit(tx)
    
    explorer = best_est.explore()
    top_lift_rules = explorer.filter_rules(min_lift=1.5, sort_by="lift", ascending=False)
    
    print("\nTop 10 Actionable Association Rules in Groceries (Ranked by Lift):")
    rule_table = explorer.to_markdown(top_lift_rules.head(10))
    print(rule_table)

    # Convert sets to lists in top_rules for JSON serialization
    top_records = []
    for r in top_lift_rules.head(10).to_dict(orient="records"):
        r_clean = {}
        for k, v in r.items():
            if isinstance(v, (set, frozenset)):
                r_clean[k] = sorted(list(v))
            else:
                r_clean[k] = v
        top_records.append(r_clean)

    return results, stats, top_records

def run_chess_experiment():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: DENSE CHESS BENCHMARK & PRUNING POLICY STRESS TEST")
    print("=" * 70)
    tx = load_chess()
    stats = dataset_summary(tx)
    print(f"Dataset stats: {stats['n_transactions']} txns, {stats['n_unique_items']} items, density={stats['density_pct']}%.")

    results = {}
    
    # 1. Normal run at high support (0.85) comparing Apriori and FP-Growth
    for algo in ["fpgrowth", "eclat", "apriori"]:
        t0 = time.perf_counter()
        est = AssociationRulesEstimator(
            min_support=0.85,
            metric="confidence",
            min_threshold=0.90,
            algorithm=algo,
            sparse_output=True
        )
        est.fit(tx)
        elapsed = time.perf_counter() - t0
        print(f"Chess {algo.upper()} (min_sup=0.85): {elapsed:.3f}s | Itemsets: {len(est.frequent_itemsets_)} | Rules: {len(est.rules_)}")
        results[f"normal_{algo}"] = {
            "elapsed_sec": round(elapsed, 4),
            "n_itemsets": len(est.frequent_itemsets_),
            "n_rules": len(est.rules_)
        }

    # 2. PruningPolicy Stress Test: Apriori with max_candidates cap vs uncapped
    print("\nStress Testing Candidate Pruning Policy on Dense Data (min_sup=0.75)...")
    est_capped = AssociationRulesEstimator(
        min_support=0.75,
        algorithm="apriori",
        max_candidates=1000,
        max_len=4,
        on_candidate_explosion="degrade",
        sparse_output=True
    )
    
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        est_capped.fit(tx)
        elapsed_capped = time.perf_counter() - t0

    degraded_flag = est_capped.is_degraded_
    has_warning = any(issubclass(w.category, DegradedModeWarning) for w in caught_warnings)
    print(f"  -> Capped Apriori: {elapsed_capped:.3f}s | Itemsets: {len(est_capped.frequent_itemsets_)} | Degraded: {degraded_flag} | Warning Triggered: {has_warning}")

    results["pruning_stress_test"] = {
        "elapsed_sec": round(elapsed_capped, 4),
        "n_itemsets": len(est_capped.frequent_itemsets_),
        "is_degraded": degraded_flag,
        "warning_issued": has_warning
    }

    return results, stats

def run_retail_experiment():
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: LARGE-SCALE RETAIL BENCHMARK (N=88,162, M=16,470)")
    print("=" * 70)
    tx = load_retail()
    stats = dataset_summary(tx)
    print(f"Dataset stats: {stats['n_transactions']} txns, {stats['n_unique_items']} items, sparsity={stats['sparsity_pct']}%.")

    results = {}
    min_supports = [0.03, 0.02]

    for min_sup in min_supports:
        results[f"min_sup_{min_sup}"] = {}
        for algo in ["fpgrowth", "eclat"]:
            t0 = time.perf_counter()
            m0 = get_peak_memory_mb()
            est = AssociationRulesEstimator(
                min_support=min_sup,
                metric="confidence",
                min_threshold=0.30,
                algorithm=algo,
                sparse_output=True
            )
            est.fit(tx)
            elapsed = time.perf_counter() - t0
            m1 = get_peak_memory_mb()
            peak_ram = max(m1 - m0, 0.1)

            n_itemsets = len(est.frequent_itemsets_)
            n_rules = len(est.rules_)
            tput = len(tx) / elapsed if elapsed > 0 else 0
            print(f"Retail {algo.upper()} (min_sup={min_sup}): {elapsed:.3f}s | RAM: {peak_ram:.1f} MB | Itemsets: {n_itemsets} | Rules: {n_rules} | {tput:.1f} tx/s")
            
            results[f"min_sup_{min_sup}"][algo] = {
                "elapsed_sec": round(elapsed, 4),
                "peak_ram_mb": round(peak_ram, 2),
                "n_itemsets": n_itemsets,
                "n_rules": n_rules,
                "throughput_tx_per_sec": round(tput, 1)
            }

    # Run Apriori at min_sup=0.03 to measure comparative baseline runtime
    print("Running APRIORI on Retail (min_sup=0.03)...")
    t0 = time.perf_counter()
    est_apriori = AssociationRulesEstimator(
        min_support=0.03,
        metric="confidence",
        min_threshold=0.30,
        algorithm="apriori",
        sparse_output=True
    )
    est_apriori.fit(tx)
    elapsed_apriori = time.perf_counter() - t0
    print(f"Retail APRIORI (min_sup=0.03): {elapsed_apriori:.3f}s | Itemsets: {len(est_apriori.frequent_itemsets_)} | Rules: {len(est_apriori.rules_)}")
    results["min_sup_0.03"]["apriori"] = {
        "elapsed_sec": round(elapsed_apriori, 4),
        "peak_ram_mb": 0.0,
        "n_itemsets": len(est_apriori.frequent_itemsets_),
        "n_rules": len(est_apriori.rules_),
        "throughput_tx_per_sec": round(len(tx) / elapsed_apriori, 1)
    }

    return results, stats

def main():
    os.makedirs("benchmarks", exist_ok=True)
    full_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "Intel/AMD x86_64, Windows",
        "python_version": sys.version,
    }

    g_results, g_stats, g_rules = run_groceries_experiment()
    c_results, c_stats = run_chess_experiment()
    r_results, r_stats = run_retail_experiment()

    full_results["groceries"] = {"stats": g_stats, "benchmarks": g_results, "top_rules": g_rules}
    full_results["chess"] = {"stats": c_stats, "benchmarks": c_results}
    full_results["retail"] = {"stats": r_stats, "benchmarks": r_results}

    def json_serializer(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (set, frozenset)):
            return sorted(list(o))
        return str(o)

    out_file = "benchmarks/real_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, default=json_serializer)

    print(f"\n[COMPLETE] Real benchmark results saved to {out_file}.")

if __name__ == "__main__":
    main()
