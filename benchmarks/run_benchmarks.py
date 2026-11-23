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

    def precision(found):
        return 100.0 * len(found.intersection(oracle_set)) / len(found) if found else 0.0

    def recall(found):
        return 100.0 * len(found.intersection(oracle_set)) / len(oracle_set) if oracle_set else 0.0

    prec_apriori, prec_fp, prec_eclat = precision(apriori_items), precision(fp_items), precision(eclat_items)
    rec_apriori, rec_fp, rec_eclat = recall(apriori_items), recall(fp_items), recall(eclat_items)

    return {
        "apriori_precision_pct": prec_apriori,
        "fpgrowth_precision_pct": prec_fp,
        "eclat_precision_pct": prec_eclat,
        "apriori_recall_pct": rec_apriori,
        "fpgrowth_recall_pct": rec_fp,
        "eclat_recall_pct": rec_eclat,
        # The old pass condition silently omitted Eclat, and precision alone
        # cannot catch a miner that misses oracle itemsets, so both
        # precision and recall of all three engines must be 100%.
        "pass": all(p >= 100.0 and r >= 100.0 for p, r in [
            (prec_apriori, rec_apriori), (prec_fp, rec_fp), (prec_eclat, rec_eclat),
        ]),
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
    elif max_err < 1e-2:
        rubric_score = 50.0
    else:
        rubric_score = 0.0

    return {
        "max_numerical_error": max_err,
        "rubric_score": rubric_score,
        "target": ">= 80/100",
        "pass": rubric_score >= 80.0,
    }


def evaluate_runtime_and_memory_scales(
    scales: List[int] = [200, 1000, 3000],
    n_warmup: int = 5,
    n_trials: int = 30,
) -> Dict[str, Any]:
    """KPI-3, KPI-4, KPI-5: Candidate counts, runtime p50/p95/p99, peak memory across scales.

    Follows the Charter KPI-4 measurement protocol: warm up each engine, run >= 30
    measured trials using a monotonic clock (time.perf_counter), and enforce strict
    p95 <= 0.80 * Apriori with zero small-runtime exemptions and exact equivalence.
    """
    engine_fns = {
        "apriori": apriori,
        "fpgrowth": fpgrowth,
        "eclat": eclat,
    }
    results: Dict[str, Any] = {
        "protocol": {
            "warmup_trials": n_warmup,
            "measured_trials": n_trials,
            "clock": "time.perf_counter",
            "small_runtime_exemptions": False,
        }
    }
    rng = np.random.default_rng(42)

    for n in scales:
        # Generate synthetic transactions with 15 items and 25% item density
        X = rng.choice([0, 1], size=(n, 15), p=[0.75, 0.25])
        scale_res: Dict[str, Any] = {}
        engine_outputs: Dict[str, Dict[Any, float]] = {}

        for algo in ["apriori", "fpgrowth", "eclat"]:
            fn = engine_fns[algo]

            # 1. Core mining engine timing (uninstrumented: tracemalloc stopped)
            if tracemalloc.is_tracing():
                tracemalloc.stop()

            for _ in range(n_warmup):
                last_engine_res = fn(X, min_support=0.08)

            engine_runtimes: List[float] = []
            engine_failures = 0
            for _ in range(n_trials):
                t0 = time.perf_counter()
                try:
                    last_engine_res = fn(X, min_support=0.08)
                    engine_runtimes.append(time.perf_counter() - t0)
                except Exception:
                    engine_failures += 1

            engine_outputs[algo] = last_engine_res

            # Dedicated core-engine tracemalloc peak memory measurement
            tracemalloc.start()
            fn(X, min_support=0.08)
            _, engine_peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # 2. End-to-end AssociationRulesEstimator.fit() timing (warmup + 30 trials)
            for _ in range(n_warmup):
                if tracemalloc.is_tracing():
                    tracemalloc.stop()
                AssociationRulesEstimator(min_support=0.08, algorithm=algo, max_candidates=50000).fit(X)

            est_runtimes: List[float] = []
            est_failures = 0
            peak_candidates: List[int] = []
            for _ in range(n_trials):
                if tracemalloc.is_tracing():
                    tracemalloc.stop()
                t0 = time.perf_counter()
                try:
                    est = AssociationRulesEstimator(min_support=0.08, algorithm=algo, max_candidates=50000)
                    est.fit(X)
                    est_runtimes.append(time.perf_counter() - t0)
                    peak_candidates.append(est.pruning_policy_.peak_candidate_count)
                except Exception:
                    est_failures += 1

            # Dedicated estimator peak memory measurement
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            tracemalloc.start()
            est_mem = AssociationRulesEstimator(min_support=0.08, algorithm=algo, max_candidates=50000)
            est_mem.fit(X)
            _, est_peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            scale_res[algo] = {
                "runtime_p50_s": float(np.percentile(est_runtimes, 50)) if est_runtimes else float("inf"),
                "runtime_p95_s": float(np.percentile(est_runtimes, 95)) if est_runtimes else float("inf"),
                "runtime_p99_s": float(np.percentile(est_runtimes, 99)) if est_runtimes else float("inf"),
                "failures": est_failures,
                "engine_runtime_p50_s": float(np.percentile(engine_runtimes, 50)) if engine_runtimes else float("inf"),
                "engine_runtime_p95_s": float(np.percentile(engine_runtimes, 95)) if engine_runtimes else float("inf"),
                "engine_runtime_p99_s": float(np.percentile(engine_runtimes, 99)) if engine_runtimes else float("inf"),
                "engine_failures": engine_failures,
                "peak_mem_mb": float(est_peak_bytes / (1024 * 1024)),
                "engine_peak_mem_mb": float(engine_peak_bytes / (1024 * 1024)),
                "peak_candidates": int(np.mean(peak_candidates)) if peak_candidates else 0,
                "n_itemsets": len(last_engine_res),
            }

        # Verify exact cross-engine functional equivalence
        ref_out = engine_outputs["apriori"]
        fp_out = engine_outputs["fpgrowth"]
        ec_out = engine_outputs["eclat"]
        keys_match = set(ref_out.keys()) == set(fp_out.keys()) == set(ec_out.keys())
        max_delta = (
            max(
                max((abs(ref_out[k] - fp_out[k]) for k in ref_out), default=0.0),
                max((abs(ref_out[k] - ec_out[k]) for k in ref_out), default=0.0),
            )
            if keys_match
            else float("inf")
        )
        exact_equivalence = bool(keys_match and max_delta < 1e-12)

        ap_est_p95 = scale_res["apriori"]["runtime_p95_s"]
        fp_est_p95 = scale_res["fpgrowth"]["runtime_p95_s"]
        ec_est_p95 = scale_res["eclat"]["runtime_p95_s"]

        ap_eng_p95 = scale_res["apriori"]["engine_runtime_p95_s"]
        fp_eng_p95 = scale_res["fpgrowth"]["engine_runtime_p95_s"]
        ec_eng_p95 = scale_res["eclat"]["engine_runtime_p95_s"]

        ap_eng_mem = scale_res["apriori"]["engine_peak_mem_mb"]
        fp_eng_mem = scale_res["fpgrowth"]["engine_peak_mem_mb"]
        ec_eng_mem = scale_res["eclat"]["engine_peak_mem_mb"]

        all_zero_failures = all(
            scale_res[a]["failures"] == 0 and scale_res[a]["engine_failures"] == 0
            for a in ("apriori", "fpgrowth", "eclat")
        )

        # Strict Charter KPI-4 checks (p95 <= 0.80 * Apriori) — NO small-runtime exemptions
        fp_est_pass = bool(all_zero_failures and exact_equivalence and fp_est_p95 <= 0.80 * ap_est_p95)
        ec_est_pass = bool(all_zero_failures and exact_equivalence and ec_est_p95 <= 0.80 * ap_est_p95)
        fp_eng_pass = bool(all_zero_failures and exact_equivalence and fp_eng_p95 <= 0.80 * ap_eng_p95)
        ec_eng_pass = bool(all_zero_failures and exact_equivalence and ec_eng_p95 <= 0.80 * ap_eng_p95)

        scale_res["exact_equivalence"] = exact_equivalence
        scale_res["max_delta_support"] = float(max_delta)
        scale_res["fpgrowth_p95_ratio_vs_apriori"] = float(fp_est_p95 / ap_est_p95) if ap_est_p95 > 0 else float("inf")
        scale_res["eclat_p95_ratio_vs_apriori"] = float(ec_est_p95 / ap_est_p95) if ap_est_p95 > 0 else float("inf")
        scale_res["fpgrowth_speedup_vs_apriori"] = float(ap_est_p95 / fp_est_p95) if fp_est_p95 > 0 else 0.0
        scale_res["eclat_speedup_vs_apriori"] = float(ap_est_p95 / ec_est_p95) if ec_est_p95 > 0 else 0.0
        scale_res["fpgrowth_engine_p95_ratio_vs_apriori"] = float(fp_eng_p95 / ap_eng_p95) if ap_eng_p95 > 0 else float("inf")
        scale_res["eclat_engine_p95_ratio_vs_apriori"] = float(ec_eng_p95 / ap_eng_p95) if ap_eng_p95 > 0 else float("inf")
        scale_res["fpgrowth_engine_speedup_vs_apriori"] = float(ap_eng_p95 / fp_eng_p95) if fp_eng_p95 > 0 else 0.0
        scale_res["eclat_engine_speedup_vs_apriori"] = float(ap_eng_p95 / ec_eng_p95) if ec_eng_p95 > 0 else 0.0
        scale_res["fpgrowth_engine_mem_ratio_vs_apriori"] = float(fp_eng_mem / ap_eng_mem) if ap_eng_mem > 0 else float("inf")
        scale_res["eclat_engine_mem_ratio_vs_apriori"] = float(ec_eng_mem / ap_eng_mem) if ap_eng_mem > 0 else float("inf")
        scale_res["fpgrowth_kpi4_pass"] = bool(fp_est_pass and fp_eng_pass)
        scale_res["eclat_kpi4_pass"] = bool(ec_est_pass and ec_eng_pass)
        scale_res["runtime_kpi_pass"] = bool(fp_est_pass and ec_est_pass and fp_eng_pass and ec_eng_pass)
        results[f"scale_{n}"] = scale_res

    return results


def evaluate_kpi6_api_compliance() -> Dict[str, Any]:
    """KPI-6: Estimator API compliance pass rate (%)."""
    from sklearn.base import BaseEstimator

    checks = [
        # The old check ("not a PandasObject") was vacuous: it passed for
        # any class that is not a pandas object and never tested sklearn
        # compliance at all.
        ("BaseEstimator inheritance", issubclass(AssociationRulesEstimator, BaseEstimator)),
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
    print("[skarm Benchmark] Starting KPI-1 through KPI-6 evaluation (strict Charter protocol)...")

    kpi1 = evaluate_kpi1_precision()
    print(
        f"  KPI-1 (Precision vs Oracle): Apriori={kpi1['apriori_precision_pct']}%, "
        f"FP-Growth={kpi1['fpgrowth_precision_pct']}%, Eclat={kpi1['eclat_precision_pct']}% "
        f"[{'PASS' if kpi1['pass'] else 'FAIL'}]"
    )

    kpi2 = evaluate_kpi2_numerical_error()
    print(
        f"  KPI-2 (Numerical Error Rubric): Score={kpi2['rubric_score']}/100 "
        f"(Max Err={kpi2['max_numerical_error']:.2e}) [{'PASS' if kpi2['pass'] else 'FAIL'}]"
    )

    scale_results = evaluate_runtime_and_memory_scales(scales=[200, 1000, 3000], n_warmup=5, n_trials=30)
    scale_keys = [k for k in scale_results if k.startswith("scale_")]
    all_scales_kpi4_pass = all(scale_results[k]["runtime_kpi_pass"] for k in scale_keys)
    for k in scale_keys:
        sr = scale_results[k]
        print(
            f"  KPI-4 ({k}, 30 trials): "
            f"engine p95 ratio FP={sr['fpgrowth_engine_p95_ratio_vs_apriori']:.3f}x, "
            f"Eclat={sr['eclat_engine_p95_ratio_vs_apriori']:.3f}x | "
            f"est p95 ratio FP={sr['fpgrowth_p95_ratio_vs_apriori']:.3f}x, "
            f"Eclat={sr['eclat_p95_ratio_vs_apriori']:.3f}x "
            f"[{'PASS' if sr['runtime_kpi_pass'] else 'FAIL'}]"
        )

    kpi6 = evaluate_kpi6_api_compliance()
    print(
        f"  KPI-6 (Estimator API Compliance): {kpi6['compliance_pass_rate_pct']}% "
        f"[{'PASS' if kpi6['pass'] else 'FAIL'}]"
    )

    overall_pass = bool(kpi1["pass"] and kpi2["pass"] and all_scales_kpi4_pass and kpi6["pass"])
    report = {
        "timestamp": time.time(),
        "overall_pass": overall_pass,
        "KPI-1_frequent_itemset_precision": kpi1,
        "KPI-2_rule_metric_numerical_error": kpi2,
        "KPI-3_4_5_scale_benchmarks": scale_results,
        "KPI-6_estimator_api_compliance": kpi6,
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[skarm Benchmark] Results saved to {output_file} (overall_pass={overall_pass})")
    return report


if __name__ == "__main__":
    res = run_all_benchmarks()
    sys.exit(0 if res["overall_pass"] else 1)
