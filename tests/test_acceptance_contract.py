"""Regression tests enforcing the strict Charter KPI-4 contract and exact equivalence.

Guards against small-runtime exemptions in acceptance/benchmark harnesses and
verifies that both CP2 engines (FP-Growth and Eclat) achieve bit-exact support
identity and strict p95 <= 0.80 * Apriori under warmup + 30-trial measurement.
"""

from pathlib import Path
import re
import numpy as np
from scipy import sparse

from skarm.apriori import apriori
from skarm.fpgrowth import fpgrowth
from skarm.eclat import eclat
from tests.run_acceptance import run_acceptance_suite

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_no_small_runtime_exemptions_in_harnesses():
    """Ensure neither acceptance nor benchmark harness hides KPI-4 behind small-runtime exemptions."""
    acceptance_src = (ROOT_DIR / "tests" / "run_acceptance.py").read_text(encoding="utf-8")
    benchmark_src = (ROOT_DIR / "benchmarks" / "run_benchmarks.py").read_text(encoding="utf-8")

    exemption_pattern = re.compile(r"or\s+apriori_time\s*<\s*0\.\d+")
    assert not exemption_pattern.search(acceptance_src), (
        "Small-runtime exemption found in tests/run_acceptance.py"
    )
    assert not exemption_pattern.search(benchmark_src), (
        "Small-runtime exemption found in benchmarks/run_benchmarks.py"
    )


def test_strict_kpi4_acceptance_harness_passes():
    """Verify tests/run_acceptance.py passes strict 30-trial KPI-4 for both FP-Growth and Eclat."""
    summary = run_acceptance_suite()
    assert summary["status"] == "PASS"
    assert summary["kpi4_pass"] is True
    assert summary["kpi4_fp_pass"] is True
    assert summary["kpi4_eclat_pass"] is True
    assert summary["kpi4_protocol"]["measured_trials"] >= 30
    assert summary["kpi4_protocol"]["warmup_trials"] >= 1
    assert summary["kpi4_fp_p95_ratio"] <= 0.80
    assert summary["kpi4_eclat_p95_ratio"] <= 0.80
    assert summary["kpi4_max_delta_support"] == 0.0
    for engine in ("apriori", "fpgrowth", "eclat"):
        stats = summary["kpi4_measurements"][engine]
        assert stats["failures"] == 0
        assert stats["p50_s"] <= stats["p95_s"] <= stats["p99_s"]


def test_cross_engine_exact_equivalence_depth_regimes():
    """Verify bit-exact support identity across Apriori, FP-Growth, and Eclat across depth regimes."""
    rng = np.random.default_rng(12345)
    # 1. Depth-2 regime with triangles in pairwise co-occurrence (like scale_200)
    X_tri = rng.choice([0, 1], size=(200, 15), p=[0.75, 0.25])
    # 2. Deep dense regime with depth >= 4 frequent itemsets
    X_deep = rng.choice([0, 1], size=(120, 10), p=[0.45, 0.55])
    # 3. Sparse CSR regime
    X_sparse = sparse.csr_matrix(X_deep)
    # 4. Weighted dense regime
    weights = rng.uniform(0.5, 3.0, size=120)

    for label, X_mat, ms, w in [
        ("dense_tri", X_tri, 0.08, None),
        ("dense_deep", X_deep, 0.18, None),
        ("sparse_deep", X_sparse, 0.18, None),
        ("dense_weighted", X_deep, 0.18, weights),
    ]:
        r_ap = apriori(X_mat, min_support=ms, sample_weight=w)
        r_fp = fpgrowth(X_mat, min_support=ms, sample_weight=w)
        r_ec = eclat(X_mat, min_support=ms, sample_weight=w)

        assert set(r_ap.keys()) == set(r_fp.keys()) == set(r_ec.keys()), f"Key mismatch on {label}"
        for k in r_ap:
            if w is None:
                assert r_fp[k] == r_ap[k], f"FP-Growth support mismatch on {label} key={k}"
                assert r_ec[k] == r_ap[k], f"Eclat support mismatch on {label} key={k}"
            else:
                assert abs(r_fp[k] - r_ap[k]) < 1e-12, f"Weighted FP-Growth mismatch on {label} key={k}"
                assert abs(r_ec[k] - r_ap[k]) < 1e-12, f"Weighted Eclat mismatch on {label} key={k}"
