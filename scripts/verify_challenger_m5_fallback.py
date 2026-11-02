"""Empirical Challenger 1 Verification Oracle: Baseline Runner Fallback and Metadata Verification.

This script executes empirical tests to verify:
1. Exact Fallback Logic in benchmarks/run_baselines.py lines 524-556:
   - When canonical_itemsets is None: exact_match is None, itemsets_identity_match is None,
     max_delta_support is None, itemsets_match is None.
   - itemsets_count_match accurately reflects (n_itemsets == expected_itemsets).
   - Formatter output verification for True, False, and None exact_match values.
2. End-to-end mock execution simulating missing canonical itemsets through run_comparative_baselines.
3. Edge cases and stress testing for compare_itemsets_exact.
4. Schema and field integrity verification of benchmarks/baseline_results.json.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import benchmarks.run_baselines as rb
from benchmarks.run_baselines import (
    compare_itemsets_exact,
    run_comparative_baselines,
)


def simulate_run_baselines_verification_block(dataset_records, ref_itemsets, expected_itemsets, verbose=False):
    """Exact reproduction of lines 524-556 of benchmarks/run_baselines.py."""
    output_lines = []
    cfg = {"expected_itemsets": expected_itemsets}

    for rec in dataset_records:
        if rec.get("status") == "OK" and rec.get("canonical_itemsets") is not None:
            cnt_match, id_match, max_delta = compare_itemsets_exact(ref_itemsets, rec["canonical_itemsets"])
            rec["itemsets_count_match"] = cnt_match
            rec["itemsets_identity_match"] = id_match
            rec["max_delta_support"] = round(float(max_delta), 8) if max_delta is not None else None
            rec["exact_match"] = (cnt_match and id_match)
            rec["itemsets_match"] = rec["exact_match"]
            del rec["canonical_itemsets"]
        elif rec.get("status") == "OK":
            rec["itemsets_count_match"] = (rec["n_itemsets"] == cfg["expected_itemsets"])
            rec["itemsets_identity_match"] = None
            rec["max_delta_support"] = None
            rec["exact_match"] = None
            rec["itemsets_match"] = None
        else:
            rec["itemsets_count_match"] = None
            rec["itemsets_identity_match"] = None
            rec["max_delta_support"] = None
            rec["exact_match"] = None
            rec["itemsets_match"] = None

        if verbose and rec.get("status") == "OK":
            if rec["exact_match"] is True:
                delta_str = f"Delta_supp={rec.get('max_delta_support', 0.0)}"
                match_str = f"EXACT MATCH ({delta_str})"
            elif rec["exact_match"] is False:
                match_str = "MISMATCH"
            else:
                count_status = "COUNT MATCH" if rec["itemsets_count_match"] else "COUNT MISMATCH"
                match_str = f"{count_status} (unverified identity)"
            line = f"  Verified {rec['framework']:7s} {rec['algorithm']:16s}: {rec['n_itemsets']:5d} itemsets [{match_str}]"
            output_lines.append(line)

    return dataset_records, output_lines


def test_fallback_logic_unit():
    print("\n[1] Testing fallback logic unit matrix...")
    ref = {frozenset(["A"]): 0.5, frozenset(["B"]): 0.3}

    # Case A: Exact Match
    recs, lines = simulate_run_baselines_verification_block(
        [{"framework": "skarm", "algorithm": "fpgrowth", "status": "OK", "n_itemsets": 2, "canonical_itemsets": ref}],
        ref, 2, verbose=True
    )
    assert recs[0]["exact_match"] is True
    assert recs[0]["itemsets_match"] is True
    assert recs[0]["itemsets_identity_match"] is True
    assert recs[0]["itemsets_count_match"] is True
    assert recs[0]["max_delta_support"] == 0.0
    assert "EXACT MATCH (Delta_supp=0.0)" in lines[0]
    print("  - Case A (Exact Match): PASSED")

    # Case B: Identity Mismatch
    mismatch_cand = {frozenset(["A"]): 0.5, frozenset(["C"]): 0.3}
    recs, lines = simulate_run_baselines_verification_block(
        [{"framework": "spmf", "algorithm": "eclat", "status": "OK", "n_itemsets": 2, "canonical_itemsets": mismatch_cand}],
        ref, 2, verbose=True
    )
    assert recs[0]["exact_match"] is False
    assert recs[0]["itemsets_match"] is False
    assert recs[0]["itemsets_identity_match"] is False
    assert recs[0]["itemsets_count_match"] is True
    assert recs[0]["max_delta_support"] is None
    assert "MISMATCH" in lines[0]
    print("  - Case B (Identity Mismatch): PASSED")

    # Case C: None canonical_itemsets with Count Match
    recs, lines = simulate_run_baselines_verification_block(
        [{"framework": "external", "algorithm": "miner", "status": "OK", "n_itemsets": 100, "canonical_itemsets": None}],
        ref, 100, verbose=True
    )
    assert recs[0]["exact_match"] is None
    assert recs[0]["itemsets_match"] is None
    assert recs[0]["itemsets_identity_match"] is None
    assert recs[0]["max_delta_support"] is None
    assert recs[0]["itemsets_count_match"] is True
    assert "COUNT MATCH (unverified identity)" in lines[0]
    print("  - Case C (None canonical with Count Match): PASSED")

    # Case D: Missing canonical_itemsets with Count Mismatch
    recs, lines = simulate_run_baselines_verification_block(
        [{"framework": "external", "algorithm": "miner", "status": "OK", "n_itemsets": 95}],
        ref, 100, verbose=True
    )
    assert recs[0]["exact_match"] is None
    assert recs[0]["itemsets_match"] is None
    assert recs[0]["itemsets_identity_match"] is None
    assert recs[0]["max_delta_support"] is None
    assert recs[0]["itemsets_count_match"] is False
    assert "COUNT MISMATCH (unverified identity)" in lines[0]
    print("  - Case D (Missing canonical with Count Mismatch): PASSED")

    # Case E: Non-OK status (SKIP / ERROR)
    recs, lines = simulate_run_baselines_verification_block(
        [
            {"framework": "mlxtend", "algorithm": "apriori", "status": "SKIP", "n_itemsets": 0},
            {"framework": "spmf", "algorithm": "failed", "status": "ERROR", "n_itemsets": 0},
        ],
        ref, 100, verbose=True
    )
    for r in recs:
        assert r["exact_match"] is None
        assert r["itemsets_match"] is None
        assert r["itemsets_identity_match"] is None
        assert r["itemsets_count_match"] is None
        assert r["max_delta_support"] is None
    assert len(lines) == 0
    print("  - Case E (Non-OK status handling): PASSED")


def test_compare_itemsets_exact_boundaries():
    print("\n[2] Testing compare_itemsets_exact boundary conditions...")
    # Empty vs empty
    cnt, id_m, delta = compare_itemsets_exact({}, {})
    assert cnt is True and id_m is False and delta is None
    print("  - Empty vs Empty: PASSED")

    # Empty vs non-empty
    cnt, id_m, delta = compare_itemsets_exact({}, {frozenset(["A"]): 0.5})
    assert cnt is False and id_m is False and delta is None
    print("  - Empty vs Non-Empty: PASSED")

    # Boundary tolerance 1e-7
    ref = {frozenset([1]): 0.5000000000}
    cnt1, id1, d1 = compare_itemsets_exact(ref, {frozenset([1]): 0.5000000999}, tolerance=1e-7)
    assert cnt1 is True and id1 is True and abs(d1 - 0.0000000999) < 1e-12
    print("  - Within tolerance (1e-7): PASSED")

    cnt2, id2, d2 = compare_itemsets_exact(ref, {frozenset([1]): 0.5000002000}, tolerance=1e-7)
    assert cnt2 is True and id2 is False and abs(d2 - 0.0000002) < 1e-12
    print("  - Outside tolerance (1e-7): PASSED")


def test_baseline_results_json_schema():
    print("\n[3] Testing benchmarks/baseline_results.json schema and metadata...")
    json_path = ROOT_DIR / "benchmarks" / "baseline_results.json"
    assert json_path.exists(), f"File not found: {json_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    for k in ["measurement_commit", "measurement_timestamp", "verification_commit", "verification_timestamp", "verification_method"]:
        assert k in meta and isinstance(meta[k], str) and len(meta[k].strip()) > 0
        print(f"  - Metadata field '{k}': {meta[k]}")

    assert meta["verification_method"] == "canonical_itemsets_cross_comparison"

    for ds in ["chess", "adult_census", "bms_pos"]:
        assert ds in data["datasets"]
        records = data["datasets"][ds]["records"]
        for r in records:
            if r["status"] == "SKIP":
                assert r["exact_match"] is None
                assert r["itemsets_count_match"] is None
            elif r["status"] == "OK" and r["exact_match"] is True:
                assert r["itemsets_count_match"] is True
                assert r["itemsets_identity_match"] is True
                assert r["max_delta_support"] == 0.0

    print("  - baseline_results.json schema & metadata verification: PASSED")


def test_live_mock_run_baselines():
    print("\n[4] Testing live mock execution of run_comparative_baselines with missing canonical itemsets...")
    orig_mlx = rb.run_mlxtend_algorithm

    def mock_mlx(algorithm, X_sparse, min_support, n_transactions, n_items, feature_names=None, trials=3):
        res = orig_mlx(algorithm, X_sparse, min_support, n_transactions, n_items, feature_names, trials=1)
        if algorithm == "apriori":
            res["canonical_itemsets"] = None
        elif algorithm == "fpgrowth":
            res.pop("canonical_itemsets", None)
        return res

    rb.run_mlxtend_algorithm = mock_mlx

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
        tmp_out = tmp.name

    try:
        results = run_comparative_baselines(
            datasets=["chess"],
            trials=1,
            output_path=tmp_out,
            verbose=False,
        )

        with open(tmp_out, "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        records = saved_data["datasets"]["chess"]["records"]
        mlx_apriori = next(r for r in records if r["algorithm"] == "mlxtend_apriori")
        mlx_fpgrowth = next(r for r in records if r["algorithm"] == "mlxtend_fpgrowth")
        skarm_apriori = next(r for r in records if r["algorithm"] == "apriori" and r["framework"] == "skarm")

        # Verified fallback on mlxtend_apriori (canonical_itemsets was None)
        assert mlx_apriori["status"] == "OK"
        assert mlx_apriori["exact_match"] is None
        assert mlx_apriori["itemsets_identity_match"] is None
        assert mlx_apriori["max_delta_support"] is None
        assert mlx_apriori["itemsets_match"] is None
        assert mlx_apriori["itemsets_count_match"] is True
        assert mlx_apriori["n_itemsets"] == 2669
        print("  - Intercepted mlxtend_apriori (canonical_itemsets=None): exact_match=None, count_match=True: PASSED")

        # Verified fallback on mlxtend_fpgrowth (canonical_itemsets was missing)
        assert mlx_fpgrowth["status"] == "OK"
        assert mlx_fpgrowth["exact_match"] is None
        assert mlx_fpgrowth["itemsets_identity_match"] is None
        assert mlx_fpgrowth["max_delta_support"] is None
        assert mlx_fpgrowth["itemsets_match"] is None
        assert mlx_fpgrowth["itemsets_count_match"] is True
        assert mlx_fpgrowth["n_itemsets"] == 2669
        print("  - Intercepted mlxtend_fpgrowth (canonical_itemsets omitted): exact_match=None, count_match=True: PASSED")

        # Verified normal path on unpatched SKARM
        assert skarm_apriori["status"] == "OK"
        assert skarm_apriori["exact_match"] is True
        assert skarm_apriori["itemsets_identity_match"] is True
        assert skarm_apriori["max_delta_support"] == 0.0
        assert skarm_apriori["itemsets_match"] is True
        assert skarm_apriori["itemsets_count_match"] is True
        print("  - Unpatched SKARM (canonical_itemsets present): exact_match=True, Delta_supp=0.0: PASSED")

    finally:
        rb.run_mlxtend_algorithm = orig_mlx
        if os.path.exists(tmp_out):
            os.remove(tmp_out)


def main():
    print("=" * 80)
    print("CHALLENGER 1 EMPIRICAL VERIFICATION: BASELINE RUNNER FALLBACK & METADATA")
    print("=" * 80)
    test_fallback_logic_unit()
    test_compare_itemsets_exact_boundaries()
    test_baseline_results_json_schema()
    test_live_mock_run_baselines()
    print("\n" + "=" * 80)
    print("FINAL EMPIRICAL VERDICT: ALL TESTS PASSED -> APPROVE")
    print("=" * 80)


if __name__ == "__main__":
    main()
