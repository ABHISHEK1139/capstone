"""Self-contained runner for the SKARM Independent E2E Verification Suite.

KL University Capstone Project Framework: KLCAP-2026-00340
Executes all 5 E2E Test Tiers:
- Tier 1: Feature Coverage (Input formats, Core miners, Rule metrics, Result states, Estimator API)
- Tier 2: Boundary & Corner Cases (Transaction geometry, Token idempotency, Extreme thresholds, Missing values, Budgets)
- Tier 3: Cross-Feature Combinations (Pairwise matrix/engine/pipeline/CV interactions)
- Tier 4: Real-World Application Scenarios (Market basket, E-commerce pipeline, Clickstream, Census, Fault-tolerant ETL)
- Tier 5: Adversarial Coverage Hardening (Boundary matrices, Budget transitions, Persistence, Numerical stability, Pandas set_output)

Outputs structured terminal scorecard and machine-readable JSON summary.
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List

import pytest


TIER_CONFIG = [
    {
        "tier": "Tier 1",
        "name": "Feature Coverage",
        "file": "test_tier1_features.py",
        "min_tests": 25,
        "description": "Input formats, core miners vs oracles, rule metrics, result states, estimator contracts",
    },
    {
        "tier": "Tier 2",
        "name": "Boundary & Corner Cases",
        "file": "test_tier2_boundaries.py",
        "min_tests": 25,
        "description": "Empty/single items, token idempotency, extreme thresholds, missing values, candidate budgets",
    },
    {
        "tier": "Tier 3",
        "name": "Cross-Feature Combinations",
        "file": "test_tier3_pairwise.py",
        "min_tests": 5,
        "description": "Pairwise interactions: CSR+Eclat, DataFrame+FP-Growth+Pipeline, GridSearchCV, weights equivalence",
    },
    {
        "tier": "Tier 4",
        "name": "Real-World Application Scenarios",
        "file": "test_tier4_scenarios.py",
        "min_tests": 5,
        "description": "Retail market basket, e-commerce intent pipeline, clickstream sparsity, census survey, batch ETL",
    },
    {
        "tier": "Tier 5",
        "name": "Adversarial Coverage Hardening",
        "file": "test_tier5_adversarial.py",
        "min_tests": 10,
        "description": "Extreme boundary matrices, multi-dimensional budgets, persistence, numerical stability, pandas set_output",
    },
]


class PytestSummaryCollector:
    """Pytest plugin to capture exact test execution metrics per tier."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = 0
        self.total = 0
        self.failures_details: List[str] = []

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.total += 1
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1
                self.failures_details.append(f"{report.nodeid}: {report.longreprtext[:200]}")
            elif report.skipped:
                self.skipped += 1
        elif report.when == "setup" and report.failed:
            self.errors += 1
            self.failures_details.append(f"SETUP ERROR {report.nodeid}: {report.longreprtext[:200]}")


def run_e2e_suite(verbose: bool = True, output_json: str = "tests/e2e/e2e_results.json") -> int:
    """Execute all E2E test tiers and output structured verification report."""
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent.parent

    # Ensure project root is in sys.path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print("=" * 85)
    print("  SKARM INDEPENDENT END-TO-END (E2E) VERIFICATION SUITE")
    print("  KL University Capstone Framework: KLCAP-2026-00340")
    print("  Opaque-Box Requirement-Driven Multi-Tier Test Campaign")
    print("=" * 85)
    print(f"  Working Directory: {project_root}")
    print(f"  Python Interpreter: {sys.executable}")
    print(f"  Test Directory:    {base_dir}")
    print("-" * 85)

    suite_start = time.perf_counter()
    tier_results = []
    all_passed = True

    for item in TIER_CONFIG:
        test_file_path = str(base_dir / item["file"])
        tier_title = f"{item['tier']}: {item['name']}"
        print(f"\n>> Executing {tier_title} ({item['file']})...")
        print(f"   Scope: {item['description']}")

        collector = PytestSummaryCollector()
        t0 = time.perf_counter()

        # Run pytest programmatically for isolation
        ret = pytest.main([
            test_file_path,
            "-q",
            "--tb=short",
        ], plugins=[collector])

        duration = time.perf_counter() - t0
        passed = (ret == pytest.ExitCode.OK) and (collector.failed == 0) and (collector.errors == 0)
        if not passed:
            all_passed = False

        status_str = "PASS" if passed else "FAIL"
        print(f"   Result: [{status_str}] in {duration:.2f}s | Passed: {collector.passed}/{collector.total} | Failed: {collector.failed} | Errors: {collector.errors}")

        tier_results.append({
            "tier": item["tier"],
            "name": item["name"],
            "file": item["file"],
            "description": item["description"],
            "status": status_str,
            "duration_s": round(duration, 3),
            "passed": collector.passed,
            "failed": collector.failed,
            "errors": collector.errors,
            "skipped": collector.skipped,
            "total": collector.total,
            "failures": collector.failures_details,
        })

    total_time = time.perf_counter() - suite_start

    # Print Scorecard
    print("\n" + "=" * 85)
    print("  E2E MULTI-TIER EXECUTION SCORECARD")
    print("=" * 85)
    print(f"  {'Tier':<8} | {'Feature Category':<30} | {'Tests':<7} | {'Passed':<6} | {'Failed':<6} | {'Status':<6} | {'Time (s)':<8}")
    print("  " + "-" * 81)

    total_tests = sum(r["total"] for r in tier_results)
    total_passed = sum(r["passed"] for r in tier_results)
    total_failed = sum(r["failed"] for r in tier_results)
    total_errors = sum(r["errors"] for r in tier_results)

    for r in tier_results:
        print(f"  {r['tier']:<8} | {r['name']:<30} | {r['total']:<7} | {r['passed']:<6} | {r['failed']:<6} | {r['status']:<6} | {r['duration_s']:<8.2f}")

    print("  " + "-" * 81)
    overall_status = "ALL TESTS PASSED" if all_passed else "FAILURES DETECTED"
    print(f"  TOTAL    | {'ALL 5 TIERS':<30} | {total_tests:<7} | {total_passed:<6} | {total_failed:<6} | {overall_status:<6} | {total_time:<8.2f}")
    print("=" * 85)

    # Save JSON summary
    summary_data = {
        "framework": "SKARM E2E Verification Suite",
        "capstone_id": "KLCAP-2026-00340",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overall_status": overall_status,
        "total_time_seconds": round(total_time, 3),
        "total_tests": total_tests,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "total_errors": total_errors,
        "tiers": tier_results,
    }

    try:
        out_path = Path(output_json)
        if not out_path.is_absolute():
            out_path = project_root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        print(f"\n[INFO] Machine-readable test report written to: {out_path}")
    except Exception as e:
        print(f"[WARN] Failed to write JSON summary: {e}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = run_e2e_suite()
    sys.exit(exit_code)
