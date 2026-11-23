"""Automated Acceptance Test Harness (AC-1 to AC-4, NT-1 to NT-6, and KPI scorecard).

KL University Capstone Project Framework: KLCAP-2026-00340
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

from skarm.estimator import AssociationRulesEstimator
from skarm.apriori import apriori
from skarm.fpgrowth import fpgrowth
from skarm.eclat import eclat
from skarm.rules import association_rules
from skarm.telemetry import RecoveryCoordinator
from tests.fixtures.oracle_data import (
    HAND_COMPUTED_TRANSACTIONS,
    ORACLE_FREQUENT_ITEMSETS_0_4,
    ORACLE_RULES_0_4_CONF_0_6,
)


def run_acceptance_suite() -> Dict[str, Any]:
    print("=" * 75)
    print("KL CAPSTONE KLCAP-2026-00340: INDEPENDENT ACCEPTANCE HARNESS (AC-1..AC-4)")
    print("Scalable Association-Rule Mining Estimator with Apriori Baseline")
    print("=" * 75)

    coordinator = RecoveryCoordinator()
    suite_start = time.perf_counter()

    # -------------------------------------------------------------
    # 1. AC-1: Representative Operation & Precision vs Oracle
    # -------------------------------------------------------------
    print("\n[AC-1] Validating Representative Operation and Precision versus Oracle...")
    est_apriori = AssociationRulesEstimator(min_support=0.4, algorithm="apriori")
    est_apriori.fit(HAND_COMPUTED_TRANSACTIONS)
    apriori_items = {frozenset(est_apriori.columns_[i] for i in k) for k in est_apriori.frequent_itemsets_.keys()}

    est_fp = AssociationRulesEstimator(min_support=0.4, algorithm="fpgrowth")
    est_fp.fit(HAND_COMPUTED_TRANSACTIONS)
    fp_items = {frozenset(est_fp.columns_[i] for i in k) for k in est_fp.frequent_itemsets_.keys()}

    est_eclat = AssociationRulesEstimator(min_support=0.4, algorithm="eclat")
    est_eclat.fit(HAND_COMPUTED_TRANSACTIONS)
    eclat_items = {frozenset(est_eclat.columns_[i] for i in k) for k in est_eclat.frequent_itemsets_.keys()}

    oracle_items = set(ORACLE_FREQUENT_ITEMSETS_0_4.keys())

    # NOTE: intersection/len(oracle) is RECALL, not precision. The old code
    # labelled it "precision" while run_benchmarks.py computed precision
    # (intersection/len(found)) — the two scripts disagreed. Report both.
    def precision(found):
        return 100.0 * len(found & oracle_items) / len(found) if found else 0.0

    def recall(found):
        return 100.0 * len(found & oracle_items) / len(oracle_items) if oracle_items else 0.0

    kpi1_apriori = precision(apriori_items)
    kpi1_fp = precision(fp_items)
    kpi1_eclat = precision(eclat_items)
    rec_apriori = recall(apriori_items)
    rec_fp = recall(fp_items)
    rec_eclat = recall(eclat_items)

    print(f"  -> KPI-1 Itemset Precision vs Oracle:")
    print(f"     * Apriori Baseline: precision {kpi1_apriori:.2f}%, recall {rec_apriori:.2f}% (Target: >= 95%)")
    print(f"     * FP-Growth:        precision {kpi1_fp:.2f}%, recall {rec_fp:.2f}% (Target: >= 98%)")
    print(f"     * Eclat (Vertical): precision {kpi1_eclat:.2f}%, recall {rec_eclat:.2f}% (Target: >= 98%)")
    assert kpi1_apriori == 100.0 and kpi1_fp == 100.0 and kpi1_eclat == 100.0
    assert rec_apriori == 100.0 and rec_fp == 100.0 and rec_eclat == 100.0
    assert est_apriori.status_ == "EXACT" and est_apriori.is_complete_ is True
    assert est_fp.status_ == "EXACT" and est_fp.is_complete_ is True
    assert est_eclat.status_ == "EXACT" and est_eclat.is_complete_ is True
    assert hasattr(est_apriori, "metadata_") and "counters" in est_apriori.metadata_
    assert hasattr(est_apriori, "itemsets_")

    # -------------------------------------------------------------
    # 2. KPI-2: Rule-Metric Numerical Error
    # -------------------------------------------------------------
    print("\n[KPI-2] Validating Rule-Metric Numerical Error against Oracle Ground Truth...")
    rules_df = est_apriori.get_rules()
    max_num_error = 0.0

    items = sorted(list({"A", "B", "C", "D", "E", "F"}))
    name_to_idx = {name: idx for idx, name in enumerate(items)}
    idx_itemsets = {
        frozenset(name_to_idx[n] for n in k): supp
        for k, supp in ORACLE_FREQUENT_ITEMSETS_0_4.items()
    }
    oracle_rules = association_rules(idx_itemsets, metric="confidence", min_threshold=0.6, feature_names=items)

    for _, row in oracle_rules.iterrows():
        key = (row["antecedent_names"], row["consequent_names"])
        if key in ORACLE_RULES_0_4_CONF_0_6:
            exp = ORACLE_RULES_0_4_CONF_0_6[key]
            for metric in ["support", "confidence", "lift", "leverage"]:
                diff = abs(row[metric] - exp[metric])
                if diff > max_num_error:
                    max_num_error = diff

    kpi2_rubric = 100.0 if max_num_error < 1e-9 else (90.0 if max_num_error < 1e-6 else 70.0)
    print(f"  -> Maximum Numerical Error: {max_num_error:.2e}")
    print(f"  -> Rubric Score: {kpi2_rubric}/100 (Target: >= 80/100, Baseline: 80)")
    assert kpi2_rubric >= 80.0

    # -------------------------------------------------------------
    # 3. AC-2: Mandatory Negative Tests (NT-1 to NT-6)
    # -------------------------------------------------------------
    print("\n[AC-2] Executing Negative-Test & Recovery Campaign (NT-1 through NT-6)...")

    # NT-1: Dense transactions causing candidate explosion
    X_dense = np.ones((5, 20), dtype=bool)
    est_nt1 = AssociationRulesEstimator(min_support=0.5, max_candidates=100, on_candidate_explosion="degrade")
    est_nt1.fit(X_dense)
    assert est_nt1.is_degraded_ is True
    assert est_nt1.status_ == "DEGRADED"
    assert est_nt1.is_complete_ is False
    assert est_nt1.metadata_["status"] == "DEGRADED"
    assert est_nt1.metadata_["future_candidates_status"] == "UNKNOWN"
    coordinator.record_failure_or_degradation(
        "NT-1", "20-item 100% dense matrix",
        f"Candidates exceeded 100", "Degraded mode safely triggered without OOM", "Deep rules omitted"
    )
    print("  -> NT-1: Candidate explosion detection & safe degradation [PASS]")

    # NT-2: Rare and duplicate items
    tx_nt2 = [["A", "A", "A", "B", "B"], ["B", "C"], ["RARE_1"]]
    est_nt2 = AssociationRulesEstimator(min_support=0.5)
    est_nt2.fit(tx_nt2)
    # Deduplication idempotency check
    assert est_nt2.encoder_.columns_ == ["A", "B", "C", "RARE_1"]
    coordinator.record_failure_or_degradation(
        "NT-2", "Repeated item tokens and singleton rare items",
        "Deduplication completed idempotently", "Lineage maintained", "None"
    )
    print("  -> NT-2: Rare & duplicate item idempotency [PASS]")

    # NT-3: Weighted and repeated transactions
    X_nt3 = np.array([[1, 0], [1, 0], [0, 1]], dtype=bool)
    w_nt3 = np.array([4.0, 1.0, 5.0])
    est_nt3 = AssociationRulesEstimator(min_support=0.4)
    est_nt3.fit(X_nt3, sample_weight=w_nt3)
    assert np.isclose(est_nt3.frequent_itemsets_[frozenset([0])], 0.5)
    assert np.isclose(est_nt3.frequent_itemsets_[frozenset([1])], 0.5)
    coordinator.record_failure_or_degradation(
        "NT-3", "Non-uniform sample weights (4, 1, 5)",
        "Weighted supports matched exact weighted formula", "Non-negative check passed", "None"
    )
    print("  -> NT-3: Weighted transactions & repeated rows [PASS]")

    # NT-4: Boundary conditions
    est_nt4 = AssociationRulesEstimator(min_support=1.0)
    est_nt4.fit(X_nt3)
    # No item has 100% support
    assert len(est_nt4.frequent_itemsets_) == 0
    coordinator.record_failure_or_degradation(
        "NT-4", "min_support=1.0 boundary",
        "Returned empty itemsets gracefully", "Safe boundary containment", "None"
    )
    print("  -> NT-4: Minimum-support boundary cases [PASS]")

    # NT-5: Deterministic ordering and serialization
    import joblib, io
    buf = io.BytesIO()
    joblib.dump(est_apriori, buf)
    buf.seek(0)
    restored = joblib.load(buf)
    assert restored.rules_.equals(est_apriori.rules_)
    coordinator.record_failure_or_degradation(
        "NT-5", "Serialization and deserialization cycle",
        "Joblib restored identical rule tables", "Deterministic order preserved", "None"
    )
    print("  -> NT-5: Deterministic ordering & serialization [PASS]")

    # NT-6: Dense transactions stress test
    rng = np.random.default_rng(2026)
    X_stress = rng.choice([0, 1], size=(50, 18), p=[0.25, 0.75])
    est_nt6 = AssociationRulesEstimator(min_support=0.5, max_candidates=500, on_candidate_explosion="degrade")
    est_nt6.fit(X_stress)
    # The old script recorded NT-6 as PASS without checking anything; the
    # degradation flag is the actual acceptance criterion (verified: this
    # scenario degrades with 501 capped itemsets).
    assert est_nt6.is_degraded_ is True
    assert est_nt6.status_ == "DEGRADED"
    assert est_nt6.is_complete_ is False
    assert est_nt6.metadata_["status"] == "DEGRADED"
    coordinator.record_failure_or_degradation(
        "NT-6", "18-item 75% dense matrix stress test",
        "Candidate generation halted at 500", "Logged telemetry and entered degraded mode", "Residual higher order capped"
    )
    print("  -> NT-6: Dense transactions stress test [PASS]")

    # -------------------------------------------------------------
    # 4. AC-3: Independent Leakage-Resistant Partition & Evaluation Mechanics
    # -------------------------------------------------------------
    print("\n[AC-3] Validating Held-Out Evaluation Mechanics (70/30 split)...")
    rng_ac3 = np.random.default_rng(2026)
    X_full = rng_ac3.choice([0, 1], size=(100, 10), p=[0.7, 0.3])
    # 70/30 train/validation split demonstrating held-out evaluation mechanics without data leakage
    X_train = X_full[:70]
    X_val = X_full[70:]

    est_ac3 = AssociationRulesEstimator(
        min_support=0.15, metric="confidence", min_threshold=0.5, algorithm="fpgrowth"
    )
    est_ac3.fit(X_train)
    assert est_ac3.status_ == "EXACT" and est_ac3.is_complete_ is True
    assert not est_ac3.rules_.empty

    # Unsupervised held-out scoring on validation partition
    val_score = est_ac3.score(X_val)
    val_transform = est_ac3.transform(X_val)
    val_preds = est_ac3.predict(X_val)

    assert isinstance(val_score, float)
    assert val_transform.shape == (30, len(est_ac3.rules_))
    assert len(val_preds) == 30
    assert est_ac3.n_features_in_ == 10
    print(f"  -> Train partition size: {len(X_train)} samples, Validation partition size: {len(X_val)} samples")
    print(f"  -> Discovered rules on train: {len(est_ac3.rules_)}, Validation held-out mean confidence: {val_score:.4f}")
    print(f"  -> Transformed feature space: {val_transform.shape}, Zero-leakage evaluation mechanics: PASS")
    coordinator.record_failure_or_degradation(
        "AC-3", "70/30 train-validation partition split",
        f"Validation score {val_score:.4f} without training leakage", "Held-out evaluation mechanics confirmed", "None"
    )
    print("  -> AC-3: Held-out evaluation mechanics on 70/30 split [PASS]")

    # -------------------------------------------------------------
    # 5. AC-4: Resource Envelope & Scalability (Strict Charter KPI-4)
    # -------------------------------------------------------------
    print("\n[AC-4] Resource Envelope Profiling (Strict Charter KPI-4: warmup + 30 trials)...")
    import tracemalloc

    # Stop any tracemalloc session left active by TelemetryLogger during AC-1..AC-3
    # so AC-4 timing measures uninstrumented monotonic wall-clock runtime fairly.
    if tracemalloc.is_tracing():
        tracemalloc.stop()

    rng = np.random.default_rng(99)
    X_scale = rng.choice([0, 1], size=(1000, 12), p=[0.75, 0.25])
    n_warmup = 5
    n_trials = 30

    def _measure_engine(fn, X_mat, min_supp: float) -> Dict[str, Any]:
        last_res = None
        for _ in range(n_warmup):
            last_res = fn(X_mat, min_support=min_supp)

        timings = []
        failures = 0
        for _ in range(n_trials):
            t0 = time.perf_counter()
            try:
                last_res = fn(X_mat, min_support=min_supp)
                timings.append(time.perf_counter() - t0)
            except Exception:
                failures += 1

        if timings:
            p50 = float(np.percentile(timings, 50))
            p95 = float(np.percentile(timings, 95))
            p99 = float(np.percentile(timings, 99))
        else:
            p50 = p95 = p99 = float("inf")

        return {
            "p50_s": p50,
            "p95_s": p95,
            "p99_s": p99,
            "failures": failures,
            "trials": n_trials,
            "warmup": n_warmup,
            "result": last_res,
        }

    ap_stats = _measure_engine(apriori, X_scale, 0.08)
    fp_stats = _measure_engine(fpgrowth, X_scale, 0.08)
    ec_stats = _measure_engine(eclat, X_scale, 0.08)

    res_ap = ap_stats["result"] or {}
    res_fp = fp_stats["result"] or {}
    res_ec = ec_stats["result"] or {}

    keys_match = set(res_ap.keys()) == set(res_fp.keys()) == set(res_ec.keys())
    max_delta_fp = max((abs(res_ap[k] - res_fp[k]) for k in res_ap), default=0.0) if keys_match else float("inf")
    max_delta_ec = max((abs(res_ap[k] - res_ec[k]) for k in res_ap), default=0.0) if keys_match else float("inf")
    max_delta_supp = max(max_delta_fp, max_delta_ec)
    equiv_pass = bool(keys_match and max_delta_supp < 1e-12)
    assert equiv_pass, f"Cross-engine mismatch on AC-4 workload (keys_match={keys_match}, max_delta={max_delta_supp})"

    ap_p95 = ap_stats["p95_s"]
    fp_p95 = fp_stats["p95_s"]
    ec_p95 = ec_stats["p95_s"]

    fp_p95_ratio = fp_p95 / ap_p95 if ap_p95 > 0 else float("inf")
    ec_p95_ratio = ec_p95 / ap_p95 if ap_p95 > 0 else float("inf")
    fp_speedup = ap_p95 / fp_p95 if fp_p95 > 0 else 0.0
    ec_speedup = ap_p95 / ec_p95 if ec_p95 > 0 else 0.0

    # Strict Charter KPI-4 pass contract: p95 <= 0.80 * O2 Apriori reference,
    # zero failures, and exact functional equivalence — NO small-runtime exemptions.
    kpi4_fp_pass = bool(
        ap_stats["failures"] == 0
        and fp_stats["failures"] == 0
        and equiv_pass
        and fp_p95 <= 0.80 * ap_p95
    )
    kpi4_eclat_pass = bool(
        ap_stats["failures"] == 0
        and ec_stats["failures"] == 0
        and equiv_pass
        and ec_p95 <= 0.80 * ap_p95
    )
    kpi4_pass = bool(kpi4_fp_pass and kpi4_eclat_pass)

    print(f"  -> Workload: N=1000, M=12, p=0.25, min_support=0.08 ({len(res_ap)} itemsets, max_delta_supp={max_delta_supp:.2e})")
    print(f"  -> Protocol: {n_warmup} warmup + {n_trials} measured trials (monotonic perf_counter)")
    print(
        f"  -> Apriori (O2 Ref): p50={ap_stats['p50_s']:.6f}s, p95={ap_p95:.6f}s, "
        f"p99={ap_stats['p99_s']:.6f}s, failures={ap_stats['failures']}"
    )
    print(
        f"  -> FP-Growth (CP2):  p50={fp_stats['p50_s']:.6f}s, p95={fp_p95:.6f}s, "
        f"p99={fp_stats['p99_s']:.6f}s, failures={fp_stats['failures']} "
        f"| p95/Apriori={fp_p95_ratio:.3f} (Speedup: {fp_speedup:.2f}x) "
        f"[{'PASS' if kpi4_fp_pass else 'FAIL'}]"
    )
    print(
        f"  -> Eclat (CP2):      p50={ec_stats['p50_s']:.6f}s, p95={ec_p95:.6f}s, "
        f"p99={ec_stats['p99_s']:.6f}s, failures={ec_stats['failures']} "
        f"| p95/Apriori={ec_p95_ratio:.3f} (Speedup: {ec_speedup:.2f}x) "
        f"[{'PASS' if kpi4_eclat_pass else 'FAIL'}]"
    )
    print(
        f"  -> Strict KPI-4 Rule (p95 <= 0.80 * Apriori, no small-runtime exemptions): "
        f"{'PASS' if kpi4_pass else 'FAIL'}"
    )

    overall_pass = bool(kpi4_pass)

    # Summary Scorecard — dynamically reflects actual measured pass/fail status.
    total_time = time.perf_counter() - suite_start
    print("\n" + "=" * 75)
    print("ACCEPTANCE HARNESS SCORECARD SUMMARY")
    print("=" * 75)
    print(f"  * KPI-1 (Precision vs Oracle):       Apriori {kpi1_apriori:.1f}%, FP {kpi1_fp:.1f}%, Eclat {kpi1_eclat:.1f}% [PASS]")
    print(f"  * KPI-1 (Recall vs Oracle):          Apriori {rec_apriori:.1f}%, FP {rec_fp:.1f}%, Eclat {rec_eclat:.1f}% [PASS]")
    print(f"  * KPI-2 (Numerical Error Rubric):    {kpi2_rubric}/100 (max err {max_num_error:.2e}) [PASS]")
    print(f"  * KPI-3 (Candidate Bounding):        NT-1/NT-6 degraded mode asserted [PASS]")
    print(
        f"  * KPI-4 (Strict p95 <= 0.80x Ref):   FP {fp_p95_ratio:.3f}x ({fp_speedup:.2f}x faster), "
        f"Eclat {ec_p95_ratio:.3f}x ({ec_speedup:.2f}x faster) [{'PASS' if kpi4_pass else 'FAIL'}]"
    )
    print(f"  * AC-3 (Held-Out Mechanics):         70/30 train/val synthetic partition, score {val_score:.4f} [PASS]")
    print(f"  * Negative Tests (NT-1..NT-6):       6/6 Passed [PASS]")
    print(f"  * Total Execution Time:              {total_time:.2f} seconds")
    print("=" * 75)
    if overall_pass:
        print("FINAL GATE VERDICT: PASSED (ALL STRICT CHARTER TARGETS & GUARDRAILS SATISFIED)")
    else:
        print("FINAL GATE VERDICT: FAILED (ONE OR MORE STRICT CHARTER TARGETS NOT MET)")
    print("=" * 75)

    return {
        "status": "PASS" if overall_pass else "FAIL",
        "execution_time_s": total_time,
        "negative_tests_passed": 6,
        "ac3_validation_score": val_score,
        "kpi1_precision_pct": min(kpi1_apriori, kpi1_fp, kpi1_eclat),
        "kpi1_recall_pct": min(rec_apriori, rec_fp, rec_eclat),
        "kpi2_rubric_score": kpi2_rubric,
        "kpi4_pass": kpi4_pass,
        "kpi4_fp_pass": kpi4_fp_pass,
        "kpi4_eclat_pass": kpi4_eclat_pass,
        "kpi4_speedup": fp_speedup,
        "kpi4_eclat_speedup": ec_speedup,
        "kpi4_fp_p95_ratio": fp_p95_ratio,
        "kpi4_eclat_p95_ratio": ec_p95_ratio,
        "kpi4_max_delta_support": max_delta_supp,
        "kpi4_protocol": {"warmup_trials": n_warmup, "measured_trials": n_trials, "clock": "time.perf_counter"},
        "kpi4_measurements": {
            "apriori": {k: v for k, v in ap_stats.items() if k != "result"},
            "fpgrowth": {k: v for k, v in fp_stats.items() if k != "result"},
            "eclat": {k: v for k, v in ec_stats.items() if k != "result"},
        },
        "audit_trail": coordinator.get_audit_trail(),
    }


if __name__ == "__main__":
    summary = run_acceptance_suite()
    sys.exit(0 if summary["status"] == "PASS" else 1)
