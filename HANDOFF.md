# HANDOFF — skarm Capstone Project & Research Submission (KLCAP-2026-00340)

**Date**: 2026-11-02  
**Project**: `skarm` — Scalable Association-Rule Mining Estimator with Apriori Baseline  
**Repository**: `ABHISHEK1139/capstone` (main branch)  
**Authors**: Abhishek Kumar (Lead Implementer), Gontla Venkata Anvitha, Kamma Giridhar, Paladagula Venkata Rajeev  
**Supervisor / Guide**: Dr. Dudigam Ramya  
**Institution**: Koneru Lakshmaiah Education Foundation (KL University), Department of Computer Science & Engineering  
**Scope & Mandate**: Complete independent verification, comparative benchmarking, scikit-learn compatibility certification, and dual-manuscript readiness for research publication and capstone degree defense.

---

## 1. Executive Status (as of November 2, 2026)

Technical verification complete; publication decisions pending (target venue selection and author/supervisor sign-off):

| Component | Status | Key Metric / Verification Evidence |
|---|---|---|
| **Automated Test Suite** | **PASS (100%)** | **257 passed out of 257 collected** (169 unit/integration/adversarial + 88 5-tier E2E; zero failures) across Python 3.10, 3.11, 3.12 |
| **Independent Acceptance Harness** | **PASS** | AC-1 to AC-4 and NT-1 to NT-6 verified; 100.0/100 rubric score; max numerical error $\le 2.22 \times 10^{-16}$ |
| **Primary Benchmark Campaign (25 Datasets)** | **COMPLETE** | 25 benchmark datasets, 42 evaluation cells, 1,650 trial executions, 157 engine pairs evaluated for strict itemset & support identity ($\Delta_{\mathrm{supp}} = 0.0$, 0 mismatches); `benchmarks/campaign_results.json` |
| **Modern Baseline Campaign (Chess, Adult, BMS-POS)** | **COMPLETE** | 27 cells across SKARM (Apriori, FP-Growth, Eclat), mlxtend (v0.25.0), and SPMF Java 21 (FIN, PrePost+, LCMFreq, FP-Growth); 25 completed runs show $\Delta_{\mathrm{supp}} = 0.0$; 2 BMS-POS mlxtend runs skipped due to 814.8 MB dense-matrix limit; `benchmarks/baseline_results.json` |
| **Scikit-Learn Compatibility Audit** | **CERTIFIED** | 14 passed core checks, 39 expected domain-specific boundary incompatibilities (37 unique expectation keys, 2 dual-executed checks), 1 environment-skipped check, 0 unexpected failures under `check_estimator`. Distinct from institutional KPI-6 (7/7, 100% API compliance) |
| **Pruning & Memory Governor Ablation** | **COMPLETE** | Bounded degradation verified under severe candidate explosion; up to 17.20x speedups under auditable truncation (Chess, 0.3485s vs 5.9941s exact baseline); `benchmarks/ablation_results.json` |
| **LaTeX Research Manuscripts** | **COMPILED** | Dual camera-ready manuscripts (`paper/paper.pdf` and `paper/paper-ieee.pdf`) compiled via pdfTeX with 0 fatal errors, 0 undefined citations, and 0 overfull hboxes |
| **Institutional Deliverables** | **SUBMITTED** | Complete deliverables D1 through D7 submitted in `docs/` |

---

## 2. Repository Evidence & Benchmark Provenance

### 2.1 Commit Provenance
- **Primary Campaign Measurement Commit**: `ea5bff5d657639298b197074ab35bd4bb7bb7a37` (2026-11-01T18:59:27Z; 25 datasets, 1,650 records).
- **Current Verified HEAD Commit**: `420cffbb` (full commit: `420cffbb956a28c74a667f6b1ddda74798ade0bf`).
- **Baseline Re-Verification Commit**: `564a7063e74224252eaa7ae9e2eb8152c5819e5a` (2026-11-02T14:35:02Z; verification method: `canonical_itemsets_cross_comparison`).
- **Code Fix Commits for Acceptance Defects**: `ff03b451`, `dd22782f`, `904b19b3`.

### 2.2 Benchmark Artifacts
- `benchmarks/campaign_results.json`: Full 25-dataset primary PC campaign results (1,650 trial records across 42 cells, `incomplete: false`).
- `benchmarks/cells/*.json`: 25 per-dataset checkpoint files containing peak RSS memory, exact itemset counts, rule counts, candidate counts, and degradation flags.
- `benchmarks/thresholds.json`: Matched support ladder across all 25 benchmark datasets.
- `benchmarks/baseline_results.json`: External comparative benchmarks against mlxtend 0.25.0 and SPMF Java 21 modern miners across Chess, Adult Census, and BMS-POS.
- `benchmarks/ablation_results.json`: Systematic candidate limit and memory governor ablation data.
- `DATASET_INVENTORY.md`: Comprehensive manifest of all 25 dataset files in `data/`, detailing access dates (2026-10-29), source URLs, transaction counts, item counts, density, and licences.

---

## 3. Completed System Components

1. **Algorithmic Mining Engines (`skarm/`)**:
   - `skarm/apriori.py`: Canonical level-wise Apriori with support counting and anti-monotonic pruning.
   - `skarm/fpgrowth.py`: Prefix-tree FP-Growth with conditional FP-tree projection, header table links, and $O(1)$ tail-pointer optimizations.
   - `skarm/eclat.py`: Vertical TID-list intersection Eclat with vertical TID-set intersections.
2. **Resource Governance & Memory Telemetry (`skarm/pruning.py`, `skarm/telemetry.py`)**:
   - Dynamic candidate ceiling (`max_candidates`), maximum pattern length (`max_len`), and absolute process RSS memory governor budget ceiling (`max_memory_mb`). Note: `max_memory_mb` enforces an absolute process RSS budget limit, distinct from the real-time 10 ms background RSS sampler which measures resident set size delta above baseline.
   - Auditable degradation (`is_degraded` flag, `DegradedModeWarning`) when search boundaries are reached.
   - Real-time peak RSS sampling via `psutil`.
3. **Scikit-Learn Estimator (`skarm/estimator.py`)**:
   - `AssociationRulesEstimator` inheriting from `(TransformerMixin, BaseEstimator)`.
   - Full implementation of `fit()`, `transform()`, `predict()` (for transactional basket recommendation), `get_params()`, and `set_params()`.
   - Native support for scikit-learn `Pipeline`, `GridSearchCV`, and `pickle`/`joblib` serialization.
4. **Command-Line Interface (`skarm/cli.py`)**:
   - Comprehensive CLI supporting `--dataset`, `--algorithm`, `--min-support`, `--metric`, `--top-n`, and `--demo`.

---

## 4. Certified Scikit-Learn Compatibility

An automated audit against `sklearn.utils.estimator_checks.check_estimator` was executed and documented in `docs/SKLEARN_COMPATIBILITY_REPORT.md` and verified in `tests/test_check_estimator_compatibility.py`:
- **Total Checks Yielded**: 54 checks (53 executed + 1 environment-skipped).
- **Core Structural API (Passed)**: **14 checks** (cloning, estimator tags, repr, get/set_params invariance, constructor purity, unfitted checks, mixin inheritance).
- **Expected Domain Boundary Incompatibilities**: **39 check executions** (37 unique expectation keys, with 2 dual-executed checks: `check_estimators_pickle` and `check_transformer_general` under standard and `readonly_memmap=True`), strictly justified by the mathematical formulation of association-rule mining:
  - *Continuous / Non-Binary Float Inputs* (26 checks): Pattern mining requires discrete binary incidence matrices $X \in \{0, 1\}^{N \times M}$. Continuous Gaussian floats synthesized by generic test generators are rejected by design with `ValueError`.
  - *Supervised 1D $y$ Target Contracts* (4 checks): Pattern mining is unsupervised pattern discovery; `predict(X)` produces transactional basket recommendations (`List[List[str]]`) rather than a 1D supervised target vector $y$.
  - *Synthetic Mock Containers* (2 checks): Rejects `_NotAnArray` mock objects that cannot represent transaction streams.
  - *Custom Exception Formatting* (8 checks): Guardrails raise dedicated `BoundaryConditionError` rather than generic `check_array` messages.
- **Environment-Skipped**: **1 check** (`check_array_api_input`, skipped due to `SCIPY_ARRAY_API` unset).
- **Unexpected Failures**: **0**.
- **Institutional KPI-6 Distinction**: This generic 14/39/1 audit is strictly separate from institutional KPI-6, which evaluates institutional API acceptance criteria across seven mandatory contracts (`BaseEstimator` inheritance, `get_params`, `set_params`, `fit`, `predict`, `transform`, and `rules_`), achieving 7/7 (100%) compliance.

---

## 5. Known Limitations & Domain Boundaries

1. **Discrete Binary Incidence Requirement**: Input data must be strictly binary (0/1 or boolean indicator matrices) or transaction token streams. Continuous, negative, or complex data are rejected by design.
2. **Dense Memory Footprint in External Baselines**: Dense one-hot encoding over sparse high-dimensional data incurs excessive memory in naive implementations. For example, `mlxtend` required 814.8 MB for BMS-POS and was safely skipped by the 500 MB governor, while `skarm`'s native sparse CSR support mined BMS-POS in 8.10 s with minimal memory overhead.
3. **Support-Range Boundaries**: Results hold strictly at primary and matched-ladder thresholds documented in `benchmarks/thresholds.json`. Thresholds below the minimum support boundary on dense datasets trigger candidate explosion and were excluded after failed probes.
4. **Hardware and Environment Transferability**: Empirical execution timings are benchmarked on single-machine environments (AMD64 Windows 11 workstation, Python 3.12.10; and Linux test environments); timings are not directly comparable to compiled C/Java miners without accounting for process spawning and JVM overhead.
5. **Non-Production Scope**: As established in project charters, SKARM is designed as an auditable scientific and educational research framework; no unattended high-stakes mission-critical deployment is claimed.

---

## 6. Independent Reproduction Commands

To independently reproduce all tests, benchmarks, and manuscripts from a fresh clone:

```bash
# 1. Activate project virtual environment
.venv\Scripts\activate   # On Windows
# source .venv/bin/activate  # On Linux/macOS

# 2. Run the complete 257-test Pytest suite
pytest -v
# Expected: 257 passed with 0 failures

# 3. Run the 5-Tier E2E test runner
python tests/e2e/run_e2e.py
# Expected: 88 passed with 0 failures

# 4. Run the independent acceptance harness (AC-1..AC-4, NT-1..NT-6)
python tests/run_acceptance.py
# Expected: FINAL GATE VERDICT: PASSED (ALL TARGETS & GUARDRAILS SATISFIED)

# 5. Run the adversarial M5 challenge suite
pytest tests/test_adversarial_m5_challenge.py -v
# Expected: 41 passed with 0 failures

# 6. Run the certified scikit-learn check_estimator compatibility audit
pytest tests/test_check_estimator_compatibility.py -v
# Expected: 1 passed (14 passed, 39 expected xfail executions [37 unique keys], 1 skipped, 0 unexpected)

# 7. Verify challenger M5 baselines and cross-engine determinism
python scripts/verify_challenger_m5_baselines.py
# Expected: FINAL EMPIRICAL VERDICT: ALL TESTS PASSED -> APPROVE

# 8. Re-run baseline benchmark subset on Chess, Adult Census, and BMS-POS
python benchmarks/run_baselines.py --datasets chess,adult_census,bms_pos --trials 3
# Expected: 25 completed runs report EXACT MATCH (Delta_supp=0.0); 2 BMS-POS mlxtend runs skipped safely

# 9. Recompile LaTeX manuscripts to PDF
python scripts/compile_paper.py
# Expected: 0 fatal errors, 0 undefined citations, 0 overfull hboxes for both paper.pdf and paper-ieee.pdf
```

---

## 7. Remaining External Decisions

The following items are reserved for author team and faculty supervisor decision:
1. **Target Submission Venue**:
   - Primary options: IEEE Transactions on Knowledge and Data Engineering (TKDE), ACM Transactions on Knowledge Discovery from Data (TKDD), Journal of Open Source Software (JOSS), SoftwareX, or PeerJ Computer Science.
   - If a venue with a strict 12-page limit is selected (e.g., standard IEEE conference), Appendix B (cell-level timing table) can be moved to an online supplementary repository or condensed.
2. **Author and Supervisor Sign-Off**:
   - Formal review and sign-off by Faculty Guide Dr. Dudigam Ramya and the student author team (Abhishek Kumar, Gontla Venkata Anvitha, Kamma Giridhar, Paladagula Venkata Rajeev).
