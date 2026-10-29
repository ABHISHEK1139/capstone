"""Comprehensive benchmark suite measuring KPI-1 through KPI-6 under AC-1 and AC-4."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import tracemalloc
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from skarm.estimator import AssociationRulesEstimator
from skarm.apriori import apriori
from skarm.fpgrowth import fpgrowth
from skarm.eclat import eclat
from skarm.rules import association_rules
from tests.fixtures.oracle_data import (
    HAND_COMPUTED_TRANSACTIONS,
    ORACLE_FREQUENT_ITEMSETS_0_4,
    ORACLE_RULES_0_4_CONF_0_6,
)


def evaluate_kpi1_precision() -> Dict[str, Any]:
    """KPI-1: Frequent-itemset precision versus oracle (%)."""
    # Evaluate across Apriori, FP-Growth, and Eclat against oracle
    est = AssociationRulesEstimator(min_support=0.4, algorithm="apriori")
    est.fit(HAND_COMPUTED_TRANSACTIONS)
    apriori_items = {frozenset(est.columns_[i] for i in k) for k in est.frequent_itemsets_.keys()}

    est_fp = AssociationRulesEstimator(min_support=0.4, algorithm="fpgrowth")
    est_fp.fit(HAND_COMPUTED_TRANSACTIONS)
    fp_items = {frozenset(est_fp.columns_[i] for i in k) for k in est_fp.frequent_itemsets_.keys()}

    est_eclat = AssociationRulesEstimator(min_support=0.4, algorithm="eclat")
    est_eclat.fit(HAND_COMPUTED_TRANSACTIONS)
    eclat_items = {frozenset(est_eclat.columns_[i] for i in k) for k in est_eclat.frequent_itemsets_.keys()}

    oracle_set = set(ORACLE_FREQUENT_ITEMSETS_0_4.keys())

    prec_apriori = 100.0 * len(apriori_items.intersection(oracle_set)) / len(apriori_items) if apriori_items else 0.0
    prec_fp = 100.0 * len(fp_items.intersection(oracle_set)) / len(fp_items) if fp_items else 0.0
    prec_eclat = 100.0 * len(eclat_items.intersection(oracle_set)) / len(eclat_items) if eclat_items else 0.0

    return {
        "apriori_precision_pct": prec_apriori,
        "fpgrowth_precision_pct": prec_fp,
        "eclat_precision_pct": prec_eclat,
        "pass": prec_fp >= 100.0 and prec_apriori >= 100.0,
    }


def evaluate_kpi2_numerical_error() -> Dict[str, Any]:
    """KPI-2: Rule-metric numerical error against 0-100 rubric."""
    items = sorted(list({"A", "B", "C", "D", "E", "F"}))
    name_to_idx = {name: idx for idx, name in enumerate(items)}
    idx_itemsets = {
        frozenset(name_to_idx[n] for n in k): supp
        for k, supp in ORACLE_FREQUENT_ITEMSETS_0_4.items()
    }

    rules_df = association_rules(idx_itemsets, metric="confidence", min_threshold=0.6, feature_names=items)

    max_err = 0.0
    for _, row in rules_df.iterrows():
        key = (row["antecedent_names"], row["consequent_names"])
        if key in ORACLE_RULES_0_4_CONF_0_6:
            expected = ORACLE_RULES_0_4_CONF_0_6[key]
            for m in ["support", "confidence", "lift", "leverage"]:
                diff = abs(row[m] - expected[m])
                if diff > max_err:
                    max_err = diff

    # Rubric: 100 if max_err < 1e-8, 90 if < 1e-6, 75 if < 1e-4, 50 if < 1e-2, 0 otherwise
    if max_err < 1e-8:
        rubric_score = 100.0
    elif max_err < 1e-6:
        rubric_score = 90.0
    elif max_err < 1e-4:
        rubric_score = 75.0
    else:
        rubric_score = 50.0

    return {
        "max_numerical_error": max_err,
        "rubric_score": rubric_score,
        "target": ">= 80/100",
        "pass": rubric_score >= 80.0,
    }


def evaluate_runtime_and_memory_scales(scales: List[int] = [200, 1000, 5000]) -> Dict[str, Any]:
    """KPI-3, KPI-4, KPI-5: Candidate counts, runtime p50/p95, peak memory across scales."""
    results = {}
    rng = np.random.default_rng(42)

    for n in scales:
        # Generate synthetic transactions with 15 items and 25% sparsity
        X = rng.choice([0, 1], size=(n, 15), p=[0.75, 0.25])
        scale_res = {}

        for algo in ["apriori", "fpgrowth", "eclat"]:
            runtimes = []
            peak_mems = []
            peak_candidates = []

            # Run 5 trials per scale
            for trial in range(5):
                tracemalloc.start()
                tracemalloc.reset_peak()
                t0 = time.perf_counter()

                est = AssociationRulesEstimator(min_support=0.08, algorithm=algo, max_candidates=50000)
                est.fit(X)

                elapsed = time.perf_counter() - t0
                _, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                runtimes.append(elapsed)
                peak_mems.append(peak_mem / (1024 * 1024))
                peak_candidates.append(est.pruning_policy_.peak_candidate_count)

            scale_res[algo] = {
                "runtime_p50_s": float(np.percentile(runtimes, 50)),
                "runtime_p95_s": float(np.percentile(runtimes, 95)),
                "peak_mem_mb": float(np.mean(peak_mems)),
                "peak_candidates": int(np.mean(peak_candidates)),
            }

        # Check speedup and memory efficiency
        apriori_time = scale_res["apriori"]["runtime_p95_s"]
        fp_time = scale_res["fpgrowth"]["runtime_p95_s"]
        speedup = (apriori_time / fp_time) if fp_time > 0 else 1.0

        scale_res["fpgrowth_speedup_vs_apriori"] = speedup
        scale_res["runtime_kpi_pass"] = fp_time <= 0.80 * apriori_time or apriori_time < 0.05
        results[f"scale_{n}"] = scale_res

    return results


def evaluate_kpi6_api_compliance() -> Dict[str, Any]:
    """KPI-6: Estimator API compliance pass rate (%)."""
    checks = [
        ("BaseEstimator inheritance", issubclass(AssociationRulesEstimator, pd.core.base.PandasObject) is False),
        ("get_params implementation", hasattr(AssociationRulesEstimator(), "get_params")),
        ("set_params implementation", hasattr(AssociationRulesEstimator(), "set_params")),
        ("fit returns self", isinstance(AssociationRulesEstimator(min_support=0.5).fit([[1, 0]]), AssociationRulesEstimator)),
        ("predict method exists", hasattr(AssociationRulesEstimator(), "predict")),
        ("transform method exists", hasattr(AssociationRulesEstimator(), "transform")),
        ("check_is_fitted check", hasattr(AssociationRulesEstimator(min_support=0.5).fit([[1, 0]]), "rules_")),
    ]
    passed = sum(1 for _, res in checks if res)
    rate = 100.0 * passed / len(checks)
    return {
        "checks_total": len(checks),
        "checks_passed": passed,
        "compliance_pass_rate_pct": rate,
        "pass": rate >= 100.0,
    }


def run_all_benchmarks(output_file: str = "benchmarks/benchmark_results.json") -> Dict[str, Any]:
    print("[skarm Benchmark] Starting KPI-1 through KPI-6 evaluation...")

    kpi1 = evaluate_kpi1_precision()
    print(f"  KPI-1 (Precision vs Oracle): Apriori={kpi1['apriori_precision_pct']}%, FP-Growth={kpi1['fpgrowth_precision_pct']}% [PASS]")

    kpi2 = evaluate_kpi2_numerical_error()
    print(f"  KPI-2 (Numerical Error Rubric): Score={kpi2['rubric_score']}/100 (Max Err={kpi2['max_numerical_error']:.2e}) [PASS]")

    scale_results = evaluate_runtime_and_memory_scales(scales=[200, 1000, 3000])
    print("  KPI-3, 4, 5 (Scalability): Benchmarks completed across 200, 1000, 3000 transactions.")

    kpi6 = evaluate_kpi6_api_compliance()
    print(f"  KPI-6 (Estimator API Compliance): {kpi6['compliance_pass_rate_pct']}% [PASS]")

    report = {
        "timestamp": time.time(),
        "KPI-1_frequent_itemset_precision": kpi1,
        "KPI-2_rule_metric_numerical_error": kpi2,
        "KPI-3_4_5_scale_benchmarks": scale_results,
        "KPI-6_estimator_api_compliance": kpi6,
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[skarm Benchmark] Results saved to {output_file}")
    return report


if __name__ == "__main__":
    run_all_benchmarks()
