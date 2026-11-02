"""Empirical Challenger 2 Verification Oracle: Baselines & Three-Engine Determinism.

This script executes adversarial tests to verify:
1. Canonical Itemset and Support Comparison:
   - benchmarks/run_baselines.py helper functions:
     * parse_spmf_itemsets_file
     * extract_skarm_canonical_itemsets
     * compare_itemsets_exact
   - Synthetic end-to-end dataset through SKARM and SPMF (Java 21).
   - Strict perturbation oracle: verifies that perturbed support or missing/extra/swapped
     itemsets immediately report itemsets_identity_match = False.
2. Three-Engine Determinism:
   - Across varied random binary matrices (ultra-sparse, retail-like, medium, dense,
     high-density, wide, tall, all-ones column, all-zeros column, weighted transactions).
   - Verifies itemset identity and exact support match (Delta_supp == 0.0 on unweighted,
     < 1e-9 on weighted).
"""

import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Tuple

# Add root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from scipy import sparse

from benchmarks.run_baselines import (
    compare_itemsets_exact,
    export_transactions_to_spmf,
    extract_skarm_canonical_itemsets,
    parse_spmf_itemsets_file,
    run_skarm_algorithm,
    run_spmf_algorithm,
)
from skarm.estimator import AssociationRulesEstimator
from skarm.preprocessing import TransactionEncoder


def test_section_1_comparison_functions():
    print("\n" + "=" * 80)
    print("SECTION 1: CANONICAL ITEMSET AND SUPPORT COMPARISON FUNCTIONS")
    print("=" * 80)

    # 1.1 Test parse_spmf_itemsets_file
    print("\n[1.1] Testing parse_spmf_itemsets_file...")
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as tmp:
        tmp.write("1 2 #SUP: 15\n")
        tmp.write("3 #SUP: 20\n")
        tmp.write("1 2 3 #SUP: 10\n")
        tmp_name = tmp.name

    try:
        id_to_item = {1: "apple", 2: "banana", 3: "cherry"}
        n_transactions = 100
        parsed = parse_spmf_itemsets_file(tmp_name, id_to_item, n_transactions)

        expected = {
            frozenset(["apple", "banana"]): 0.15,
            frozenset(["cherry"]): 0.20,
            frozenset(["apple", "banana", "cherry"]): 0.10,
        }
        assert parsed == expected, f"Parsed SPMF file did not match expected! Got: {parsed}"
        print("  - Standard SPMF output file parsing: PASSED")

        # Test empty file
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as tmp_empty:
            tmp_empty_name = tmp_empty.name
        parsed_empty = parse_spmf_itemsets_file(tmp_empty_name, id_to_item, n_transactions)
        assert parsed_empty == {}, f"Empty file should return empty dict, got {parsed_empty}"
        os.remove(tmp_empty_name)
        print("  - Empty SPMF file handling: PASSED")

        # Test nonexistent file
        parsed_nonexistent = parse_spmf_itemsets_file("nonexistent_path_12345.txt", id_to_item, n_transactions)
        assert parsed_nonexistent == {}, f"Nonexistent file should return empty dict, got {parsed_nonexistent}"
        print("  - Nonexistent SPMF file handling: PASSED")

    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    # 1.2 Test extract_skarm_canonical_itemsets
    print("\n[1.2] Testing extract_skarm_canonical_itemsets...")
    synthetic_tx = [
        ["bread", "milk"],
        ["bread", "diaper", "beer", "eggs"],
        ["milk", "diaper", "beer", "cola"],
        ["bread", "milk", "diaper", "beer"],
        ["bread", "milk", "diaper", "cola"],
    ]
    enc = TransactionEncoder(sparse_output=True)
    X_syn = enc.fit_transform(synthetic_tx)
    est = AssociationRulesEstimator(min_support=0.4, algorithm="fpgrowth").fit(X_syn)

    canonical_skarm = extract_skarm_canonical_itemsets(est, enc.columns_)
    assert len(canonical_skarm) == len(est.frequent_itemsets_)
    for item_idx_tuple, supp in est.frequent_itemsets_.items():
        expected_items = frozenset(enc.columns_[i] for i in item_idx_tuple)
        assert expected_items in canonical_skarm
        assert canonical_skarm[expected_items] == float(supp)
    print(f"  - Canonical extraction matched all {len(canonical_skarm)} frequent itemsets: PASSED")

    # 1.3 Test compare_itemsets_exact and Perturbation Oracles
    print("\n[1.3] Testing compare_itemsets_exact Perturbation Oracles...")

    ref = {
        frozenset(["A"]): 0.60,
        frozenset(["B"]): 0.50,
        frozenset(["A", "B"]): 0.40,
        frozenset(["C"]): 0.30,
        frozenset(["A", "C"]): 0.25,
    }

    # Case A: Identical
    c_match, id_match, delta = compare_itemsets_exact(ref, ref.copy())
    assert c_match is True and id_match is True and delta == 0.0
    print("  - Case A (Identical): count_match=True, identity_match=True, delta=0.0 -> PASSED")

    # Case B: Tiny perturbation within tolerance (1e-8 <= 1e-7)
    candidate_tiny = ref.copy()
    candidate_tiny[frozenset(["A", "B"])] = 0.40 + 1e-8
    c_match, id_match, delta = compare_itemsets_exact(ref, candidate_tiny)
    assert c_match is True and id_match is True and abs(delta - 1e-8) < 1e-12
    print("  - Case B (Within tolerance 1e-8): count_match=True, identity_match=True -> PASSED")

    # Case C: Perturbation exceeding tolerance (delta = 1e-5 > 1e-7)
    candidate_pert = ref.copy()
    candidate_pert[frozenset(["A", "B"])] = 0.40 + 1e-5
    c_match, id_match, delta = compare_itemsets_exact(ref, candidate_pert)
    assert c_match is True, "Count should still match"
    assert id_match is False, "Identity match MUST be False when delta > tolerance"
    assert abs(delta - 1e-5) < 1e-9, f"Delta should be 1e-5, got {delta}"
    print(f"  - Case C (Perturbation 1e-5): count_match={c_match}, identity_match={id_match}, delta={delta:.1e} -> PASSED")

    # Case D: Large support perturbation (delta = 0.1)
    candidate_large = ref.copy()
    candidate_large[frozenset(["A"])] = 0.70
    c_match, id_match, delta = compare_itemsets_exact(ref, candidate_large)
    assert c_match is True and id_match is False and abs(delta - 0.1) < 1e-9
    print(f"  - Case D (Perturbation 0.1): count_match={c_match}, identity_match={id_match}, delta={delta:.1f} -> PASSED")

    # Case E: Missing itemset
    candidate_missing = {k: v for k, v in ref.items() if k != frozenset(["A", "C"])}
    c_match, id_match, delta = compare_itemsets_exact(ref, candidate_missing)
    assert c_match is False, "Count match MUST be False for missing itemset"
    assert id_match is False, "Identity match MUST be False for missing itemset"
    assert delta is None, "Delta should be None when keys do not match"
    print(f"  - Case E (Missing itemset): count_match={c_match}, identity_match={id_match}, delta={delta} -> PASSED")

    # Case F: Extra itemset
    candidate_extra = ref.copy()
    candidate_extra[frozenset(["D"])] = 0.20
    c_match, id_match, delta = compare_itemsets_exact(ref, candidate_extra)
    assert c_match is False and id_match is False and delta is None
    print(f"  - Case F (Extra itemset): count_match={c_match}, identity_match={id_match}, delta={delta} -> PASSED")

    # Case G: Swapped itemset (Same count, but one different itemset)
    candidate_swapped = {k: v for k, v in ref.items() if k != frozenset(["A", "C"])}
    candidate_swapped[frozenset(["B", "C"])] = 0.25
    c_match, id_match, delta = compare_itemsets_exact(ref, candidate_swapped)
    assert c_match is True, "Count match is True (same length)"
    assert id_match is False, "Identity match MUST be False for swapped itemset"
    assert delta is None, "Delta should be None when keys mismatch"
    print(f"  - Case G (Swapped itemset): count_match={c_match}, identity_match={id_match}, delta={delta} -> PASSED")

    # 1.4 End-to-End SKARM vs SPMF with Live Java Subprocess & Perturbation Invalidation
    print("\n[1.4] Testing Live SKARM vs SPMF Execution with Synthetic Transactions...")
    spmf_jar = str(ROOT_DIR / "benchmarks" / "spmf.jar")
    if not os.path.exists(spmf_jar):
        print("  [WARN] spmf.jar not found, skipping live SPMF subprocess test")
        return

    rng = np.random.default_rng(42)
    items_pool = [f"item_{i}" for i in range(1, 26)]
    synth_transactions = []
    for _ in range(300):
        tx_size = rng.integers(2, 8)
        chosen = list(rng.choice(items_pool, size=tx_size, replace=False))
        synth_transactions.append(chosen)

    # Encode for SKARM
    enc_live = TransactionEncoder(sparse_output=True)
    X_live = enc_live.fit_transform(synth_transactions)

    # Export for SPMF
    with tempfile.TemporaryDirectory() as tmp_dir:
        spmf_input_path = os.path.join(tmp_dir, "synth.spmf")
        _, id_to_item_live = export_transactions_to_spmf(synth_transactions, spmf_input_path)

        min_supp = 0.05
        # Run SPMF FPGrowth and FIN
        spmf_res_fp = run_spmf_algorithm(
            spmf_jar=spmf_jar,
            algorithm="FPGrowth_itemsets",
            spmf_input_file=spmf_input_path,
            min_support=min_supp,
            id_to_item=id_to_item_live,
            n_transactions=len(synth_transactions),
            trials=1,
        )

        spmf_res_fin = run_spmf_algorithm(
            spmf_jar=spmf_jar,
            algorithm="FIN",
            spmf_input_file=spmf_input_path,
            min_support=min_supp,
            id_to_item=id_to_item_live,
            n_transactions=len(synth_transactions),
            trials=1,
        )

        # Run SKARM engines
        skarm_res_fpg = run_skarm_algorithm("fpgrowth", X_live, min_supp, feature_names=enc_live.columns_, trials=1)
        skarm_res_eclat = run_skarm_algorithm("eclat", X_live, min_supp, feature_names=enc_live.columns_, trials=1)
        skarm_res_apriori = run_skarm_algorithm("apriori", X_live, min_supp, feature_names=enc_live.columns_, trials=1)

        skarm_canon = skarm_res_fpg["canonical_itemsets"]
        spmf_canon_fp = spmf_res_fp["canonical_itemsets"]
        spmf_canon_fin = spmf_res_fin["canonical_itemsets"]

        print(f"  - Discovered itemsets count: SKARM={len(skarm_canon)}, SPMF(FPGrowth)={len(spmf_canon_fp)}, SPMF(FIN)={len(spmf_canon_fin)}")
        assert len(skarm_canon) > 0, "Synthetic dataset should yield frequent itemsets"

        # Compare SKARM vs SPMF FPGrowth
        c_m, id_m, d = compare_itemsets_exact(skarm_canon, spmf_canon_fp)
        print(f"  - SKARM vs SPMF (FPGrowth): count_match={c_m}, identity_match={id_m}, Delta_supp={d}")
        assert c_m is True and id_m is True and d == 0.0, "SKARM and SPMF FPGrowth must produce identical itemsets!"

        # Compare SKARM vs SPMF FIN
        c_m, id_m, d = compare_itemsets_exact(skarm_canon, spmf_canon_fin)
        print(f"  - SKARM vs SPMF (FIN):      count_match={c_m}, identity_match={id_m}, Delta_supp={d}")
        assert c_m is True and id_m is True and d == 0.0, "SKARM and SPMF FIN must produce identical itemsets!"

        # Adversarial Perturbation: perturb one SPMF itemset support and verify rejection
        corrupted_spmf = spmf_canon_fp.copy()
        first_key = next(iter(corrupted_spmf))
        corrupted_spmf[first_key] += 0.01  # +1% support perturbation
        c_m, id_m, d = compare_itemsets_exact(skarm_canon, corrupted_spmf)
        print(f"  - Adversarial support perturbation (+0.01): count_match={c_m}, identity_match={id_m}, Delta_supp={d:.4f}")
        assert c_m is True and id_m is False and d > 0.009, "Perturbation must cause identity_match=False!"

        # Adversarial Removal: remove one itemset and verify rejection
        dropped_spmf = {k: v for k, v in spmf_canon_fp.items() if k != first_key}
        c_m, id_m, d = compare_itemsets_exact(skarm_canon, dropped_spmf)
        print(f"  - Adversarial itemset drop: count_match={c_m}, identity_match={id_m}, Delta_supp={d}")
        assert c_m is False and id_m is False and d is None, "Dropped itemset must cause identity_match=False!"

    print("  [VERDICT SECTION 1]: APPROVE (Canonical extraction, parsing, and exact comparison verified)")


def test_section_2_three_engine_determinism():
    print("\n" + "=" * 80)
    print("SECTION 2: THREE-ENGINE DETERMINISM ACROSS VARIED RANDOM BINARY MATRICES")
    print("=" * 80)

    matrix_configurations = [
        {
            "name": "Ultra-Sparse Matrix (N=200, M=50, density=2%)",
            "N": 200,
            "M": 50,
            "p": [0.98, 0.02],
            "min_support": 0.02,
            "weights": None,
        },
        {
            "name": "Correlated Multi-Item Clusters (N=400, M=25, length 2-4 itemsets)",
            "N": 400,
            "M": 25,
            "p": [0.85, 0.15],
            "min_support": 0.08,
            "special": "correlated_clusters",
            "weights": None,
        },
        {
            "name": "Sparse Retail Matrix (N=500, M=30, density=10%)",
            "N": 500,
            "M": 30,
            "p": [0.90, 0.10],
            "min_support": 0.03,
            "weights": None,
        },
        {
            "name": "Medium Density Matrix (N=300, M=20, density=25%)",
            "N": 300,
            "M": 20,
            "p": [0.75, 0.25],
            "min_support": 0.08,
            "weights": None,
        },
        {
            "name": "Dense Matrix (N=200, M=15, density=45%)",
            "N": 200,
            "M": 15,
            "p": [0.55, 0.45],
            "min_support": 0.25,
            "weights": None,
        },
        {
            "name": "High-Density Matrix (N=150, M=12, density=65%)",
            "N": 150,
            "M": 12,
            "p": [0.35, 0.65],
            "min_support": 0.40,
            "weights": None,
        },
        {
            "name": "Wide Matrix (N=100, M=80, density=5%)",
            "N": 100,
            "M": 80,
            "p": [0.95, 0.05],
            "min_support": 0.03,
            "weights": None,
        },
        {
            "name": "Tall Matrix (N=2000, M=10, density=20%)",
            "N": 2000,
            "M": 10,
            "p": [0.80, 0.20],
            "min_support": 0.05,
            "weights": None,
        },
        {
            "name": "Zero-Variance Columns (Empty Column & All-Ones Column)",
            "N": 200,
            "M": 15,
            "p": [0.75, 0.25],
            "min_support": 0.10,
            "special": "boundary_cols",
            "weights": None,
        },
        {
            "name": "Continuous Weighted Matrix (N=250, M=16, non-uniform weights)",
            "N": 250,
            "M": 16,
            "p": [0.80, 0.20],
            "min_support": 0.05,
            "weights": "random_uniform",
        },
    ]

    for idx, cfg in enumerate(matrix_configurations, 1):
        rng = np.random.default_rng(2026 + idx)
        print(f"\n[{idx}/{len(matrix_configurations)}] Testing {cfg['name']}...")

        X = rng.choice([0, 1], size=(cfg["N"], cfg["M"]), p=cfg["p"]).astype(np.int8)

        if cfg.get("special") == "boundary_cols":
            X[:, 0] = 0  # Item 0 never appears
            X[:, 1] = 1  # Item 1 appears in every transaction
        elif cfg.get("special") == "correlated_clusters":
            # Plant frequent co-occurring patterns
            # Cluster 1: items 0, 1, 2 co-occur in 30% of rows
            mask1 = rng.random(cfg["N"]) < 0.30
            X[mask1, 0] = 1
            X[mask1, 1] = 1
            X[mask1, 2] = 1
            # Cluster 2: items 5, 6, 7, 8 co-occur in 25% of rows
            mask2 = rng.random(cfg["N"]) < 0.25
            X[mask2, 5] = 1
            X[mask2, 6] = 1
            X[mask2, 7] = 1
            X[mask2, 8] = 1

        w = None
        is_weighted = False
        if cfg.get("weights") == "random_uniform":
            w = rng.uniform(0.1, 5.0, size=cfg["N"])
            is_weighted = True

        actual_density = float(np.count_nonzero(X)) / (cfg["N"] * cfg["M"])
        print(f"    Dimensions: {X.shape[0]}x{X.shape[1]} | Empirical Density: {actual_density:.2%}")

        # Run all three engines
        ests = {}
        itemsets_dicts = {}
        for algo in ["apriori", "fpgrowth", "eclat"]:
            t0 = time.perf_counter()
            est = AssociationRulesEstimator(
                min_support=cfg["min_support"],
                algorithm=algo,
                max_candidates=10_000_000,
                on_candidate_explosion="raise",
            ).fit(X, sample_weight=w)
            elapsed = time.perf_counter() - t0
            ests[algo] = est
            # Extract frozenset of integer indices -> support float
            itemsets_dicts[algo] = {frozenset(k): float(v) for k, v in est.frequent_itemsets_.items()}
            print(f"    Engine {algo:8s}: {len(itemsets_dicts[algo]):5d} itemsets in {elapsed:.4f}s")

        keys_a = set(itemsets_dicts["apriori"].keys())
        keys_f = set(itemsets_dicts["fpgrowth"].keys())
        keys_e = set(itemsets_dicts["eclat"].keys())

        # 1. Exact Set Identity Check
        assert keys_a == keys_f, f"Key mismatch between Apriori and FP-Growth! (A={len(keys_a)}, F={len(keys_f)})"
        assert keys_a == keys_e, f"Key mismatch between Apriori and Eclat! (A={len(keys_a)}, E={len(keys_e)})"
        print(f"    Itemset Identity: STRICT MATCH across Apriori, FP-Growth, Eclat ({len(keys_a)} itemsets)")

        if len(keys_a) == 0:
            print("    [WARN] Zero itemsets mined with current min_support; skipping delta check")
            continue

        # 2. compare_itemsets_exact Evaluation
        c_m_af, id_m_af, d_af = compare_itemsets_exact(itemsets_dicts["apriori"], itemsets_dicts["fpgrowth"], tolerance=1e-7 if is_weighted else 0.0)
        c_m_ae, id_m_ae, d_ae = compare_itemsets_exact(itemsets_dicts["apriori"], itemsets_dicts["eclat"], tolerance=1e-7 if is_weighted else 0.0)
        c_m_fe, id_m_fe, d_fe = compare_itemsets_exact(itemsets_dicts["fpgrowth"], itemsets_dicts["eclat"], tolerance=1e-7 if is_weighted else 0.0)

        assert c_m_af is True and id_m_af is True, f"compare_itemsets_exact failed between Apriori and FP-Growth! (delta={d_af})"
        assert c_m_ae is True and id_m_ae is True, f"compare_itemsets_exact failed between Apriori and Eclat! (delta={d_ae})"
        assert c_m_fe is True and id_m_fe is True, f"compare_itemsets_exact failed between FP-Growth and Eclat! (delta={d_fe})"

        overall_max_delta = max(d_af if d_af else 0.0, d_ae if d_ae else 0.0, d_fe if d_fe else 0.0)
        print(f"    compare_itemsets_exact(A, F): match={id_m_af}, Delta_supp={d_af:.2e}")
        print(f"    compare_itemsets_exact(A, E): match={id_m_ae}, Delta_supp={d_ae:.2e}")
        print(f"    compare_itemsets_exact(F, E): match={id_m_fe}, Delta_supp={d_fe:.2e}")

        if not is_weighted:
            assert overall_max_delta == 0.0, f"Unweighted Delta_supp must be strictly 0.0, got {overall_max_delta}"
            print("    Verdict: PASSED (Delta_supp == 0.0 EXACT ZERO verified by compare_itemsets_exact)")
        else:
            assert overall_max_delta < 1e-9, f"Weighted Delta_supp exceeded 1e-9, got {overall_max_delta}"
            print("    Verdict: PASSED (Delta_supp < 1e-9 machine epsilon verified by compare_itemsets_exact)")

        # 3. Rule Generation Equivalence Check
        print("    Checking Association Rules Equivalence (confidence, lift, zhangs_metric)...")
        rules_a = ests["apriori"].rules_
        rules_f = ests["fpgrowth"].rules_
        rules_e = ests["eclat"].rules_

        assert len(rules_a) == len(rules_f) == len(rules_e), (
            f"Rule count mismatch! A={len(rules_a)}, F={len(rules_f)}, E={len(rules_e)}"
        )

        def rule_dict(df):
            return {
                (frozenset(r["antecedent"]), frozenset(r["consequent"])): r
                for _, r in df.iterrows()
            }

        ra_d = rule_dict(rules_a)
        rf_d = rule_dict(rules_f)
        re_d = rule_dict(rules_e)

        assert set(ra_d.keys()) == set(rf_d.keys()) == set(re_d.keys()), "Rule keys differ across engines!"

        for r_k in ra_d:
            row_a, row_f, row_e = ra_d[r_k], rf_d[r_k], re_d[r_k]
            for metric_col in ["support", "confidence", "lift", "leverage", "conviction", "zhangs_metric"]:
                va, vf, ve = row_a[metric_col], row_f[metric_col], row_e[metric_col]
                if math.isinf(va) or math.isinf(vf) or math.isinf(ve):
                    assert math.isinf(va) and math.isinf(vf) and math.isinf(ve)
                else:
                    assert abs(va - vf) < 1e-9 and abs(va - ve) < 1e-9

        print(f"    Rule Equivalence: {len(rules_a)} rules strictly equivalent across all metrics")

    print("\n" + "=" * 80)
    print("FINAL EMPIRICAL VERDICT: ALL TESTS PASSED -> APPROVE")
    print("=" * 80)


if __name__ == "__main__":
    t_start = time.perf_counter()
    test_section_1_comparison_functions()
    test_section_2_three_engine_determinism()
    print(f"\nTotal empirical verification run completed in {time.perf_counter() - t_start:.2f} seconds.")
